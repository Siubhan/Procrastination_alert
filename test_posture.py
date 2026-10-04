"""Подстройка под позу и ложные тревоги: регрессия на главную причину "плохого трекинга".

Сигнал глаз здесь - ОЦЕНКА из геометрии (амплитуды), шум - измерен на настоящем FaceMesh.
"""
import numpy as np
import pytest

import config as C
import gaze
from logic import AlertLogic

NOISE = np.array([0.0045, 0.0029, 0.0066, 0.0055, 0.0103])


def world(x, y):
    gx, gy = x - 0.5, y - 0.5
    return np.array([0.5 + 0.10 * gx, 0.5 + 0.06 * gy, 0.16 * gx,
                     0.08 * gy - 0.12 * max(gy, 0) ** 2,
                     0.28 - 0.12 * max(gy, 0) + 0.02 * max(-gy, 0)])


class Harness:
    """Гоняет настоящий GazeTracker.update, подставляя готовые признаки вместо картинки."""

    def __init__(self, calib):
        self.t = gaze.GazeTracker(mesh=object())
        self.t.calib = calib
        self.cur = None
        self.t._landmarks = lambda frame: np.zeros((15, 2)) if self.cur is not None else None
        self.t.extractor = lambda P, smooth=True: self.cur.copy()

    def step(self, feats, now):
        self.cur = feats
        return self.t.update(None, now)


def calibrated(seed=0):
    rng = np.random.default_rng(seed)
    targets = gaze.snake_targets(C.CALIB_COLS, C.CALIB_ROWS)
    samples = [np.array([world(x, y) + rng.normal(0, NOISE) for _ in range(20)]) for x, y in targets]
    return gaze.build_calibration(samples, targets)[0], rng


def alerts_in_session(calib, rng, posture, seconds=180, drift=True, fps=30):
    """Сколько тревог "away" услышит человек, который всё время смотрит на экран (логика как в приложении)."""
    h = Harness(calib)
    if not drift:
        h.t.drift.apply = lambda f, t: f
    logic = AlertLogic(C.AWAY_SECONDS, C.COOLDOWN_SECONDS)
    t, alerts, fix = 0.0, 0, 0
    x = y = 0.5
    for _ in range(int(seconds * fps)):
        if fix <= 0:
            fix = int(rng.uniform(0.3, 1.5) * fps)
            x, y = np.clip(rng.normal(0.5, 0.25, 2), 0.03, 0.97)
        fix -= 1
        f = world(x, y) + rng.normal(0, NOISE)
        f[0] += posture[0]
        f[1] += posture[1]
        f[4] += posture[2]
        st = h.step(f, t)
        alerts += logic.update(t, st.away, False) == "away"
        t += 1 / fps
    return alerts


# ---------------- PostureDrift ----------------
def test_drift_absorbs_small_posture_shift_slowly():
    d = gaze.PostureDrift([0.5, 0.5, 0.28])
    f = np.array([0.52, 0.53, 0.0, 0.0, 0.28])
    out = f
    for i in range(30 * 60):
        out = d.apply(f, i / 30)
    assert out[0] == pytest.approx(0.5, abs=0.002) and out[1] == pytest.approx(0.5, abs=0.002)
    first = gaze.PostureDrift([0.5, 0.5, 0.28]).apply(f, 0.0)
    assert first[0] == pytest.approx(0.52)                          # мгновенно ничего не меняется


def test_drift_does_not_absorb_large_deviation_like_looking_away():
    d = gaze.PostureDrift([0.5, 0.5, 0.28])
    away = np.array([0.5, 0.65, 0.0, 0.0, 0.17])                    # голова вниз + глаза прикрыты: телефон
    for i in range(30 * 120):                                      # 2 минуты смотрит на телефон
        out = d.apply(away, i / 30)
    assert out[1] == pytest.approx(0.65, abs=1e-9) and out[4] == pytest.approx(0.17, abs=1e-9)


def test_drift_is_clamped_to_limit():
    d = gaze.PostureDrift([0.5, 0.5, 0.28])
    f = np.array([0.5, 0.5 + C.DRIFT_LIMIT[1] * 1.4, 0.0, 0.0, 0.28])  # внутри "захвата", но больше предела
    for i in range(30 * 300):
        out = d.apply(f, i / 30)
    assert 0.5 + C.DRIFT_LIMIT[1] * 0.4 - 1e-6 <= out[1] <= 0.5 + C.DRIFT_LIMIT[1] * 0.4 + 1e-6


def test_drift_ignores_huge_time_gaps_and_leaves_gaze_features_alone():
    d = gaze.PostureDrift([0.5, 0.5, 0.28])
    f = np.array([0.52, 0.5, 0.07, -0.03, 0.28])
    d.apply(f, 0.0)
    out = d.apply(f, 1000.0)                                        # лицо пропадало на 1000 с
    assert out[0] > 0.5                                             # за один "прыжок" не подстроились
    assert out[2] == pytest.approx(0.07) and out[3] == pytest.approx(-0.03)   # зрачок не трогаем


def test_recenter_takes_current_pose_instantly():
    d = gaze.PostureDrift([0.5, 0.5, 0.28])
    f = np.array([0.53, 0.56, 0.0, 0.0, 0.30])
    d.recenter(f)
    assert d.apply(f, 0.0)[:2] == pytest.approx([0.5, 0.5], abs=1e-9)


# ---------------- ложные тревоги ----------------
def test_posture_change_causes_few_alerts_with_drift_and_many_without():
    calib, rng = calibrated(0)
    posture = (0.03, 0.05, 0.0)                                      # сел заметно иначе, чем при калибровке
    with_drift = alerts_in_session(calib, np.random.default_rng(1), posture, drift=True)
    without = alerts_in_session(calib, np.random.default_rng(1), posture, drift=False)
    assert without >= 8                                              # раньше: десятки тревог
    assert with_drift <= 2
    assert without >= 5 * max(with_drift, 1)


def test_no_alerts_when_posture_is_unchanged():
    calib, rng = calibrated(1)
    assert alerts_in_session(calib, rng, (0, 0, 0)) == 0


def test_long_stare_at_phone_is_never_absorbed():
    calib, rng = calibrated(2)
    h = Harness(calib)
    flagged = []
    for i in range(30 * 120):
        t = i / 30
        target = (0.5, 0.5) if t < 20 else (0.5, 1.45)               # 20 с в экран, потом 100 с на телефон
        st = h.step(world(*target) + rng.normal(0, NOISE), t)
        if t > 90:
            flagged.append(st.away)
    assert np.mean(flagged) > 0.95


def test_looking_sideways_is_still_detected_after_posture_change():
    calib, rng = calibrated(3)
    h = Harness(calib)
    for i in range(30 * 60):                                         # минута в новой позе
        h.step(world(0.5, 0.5) + np.array([0.02, 0.03, 0, 0, 0]) + rng.normal(0, NOISE), i / 30)
    flagged = [h.step(world(1.7, 0.5) + np.array([0.02, 0.03, 0, 0, 0]) + rng.normal(0, NOISE), 60 + i / 30).away
               for i in range(30 * 3)]
    assert np.mean(flagged[45:]) > 0.9


# ---------------- допуск, доверие к карте, причина ----------------
def test_untrusted_mapper_falls_back_to_head_only():
    calib, rng = calibrated(4)
    calib.mapper.cv_error = C.MAPPER_MAX_CV + 0.05
    assert not calib.mapper_trusted
    h = Harness(calib)
    st = None
    for i in range(60):
        st = h.step(world(0.5, 1.8) + rng.normal(0, NOISE), i / 30)   # далеко ниже экрана по глазам
    assert not st.screen_out                                          # карте не верим
    st = h.step(np.array([0.9, 0.5, 0, 0, 0.28]), 3.0)                # а голова повёрнута - тревога
    assert st.head_out and st.away


def test_tolerance_keys_change_decision_and_are_bounded():
    calib, rng = calibrated(5)
    h = Harness(calib)
    for i in range(60):
        h.step(world(0.5, 0.5) + rng.normal(0, NOISE), i / 30)
    near_edge = world(0.5, 1.2)                                       # чуть ниже экрана
    base = [h.step(near_edge + rng.normal(0, NOISE), 2 + i / 30) for i in range(30)][-1]
    h.t.adjust_tolerance(0.3)
    loose = [h.step(near_edge + rng.normal(0, NOISE), 3 + i / 30) for i in range(30)][-1]
    assert not loose.screen_out
    assert h.t.adjust_tolerance(5.0) == pytest.approx(0.3) and h.t.adjust_tolerance(-5.0) == pytest.approx(-0.1)
    assert base.pt == pytest.approx(loose.pt, abs=0.05)


def test_reason_names_the_cause():
    calib, rng = calibrated(6)
    h = Harness(calib)
    for i in range(60):
        st = h.step(world(0.5, 0.5) + rng.normal(0, NOISE), i / 30)
    assert st.reason == ""
    st = h.step(np.array([0.95, 0.5, 0, 0, 0.28]), 3.0)
    assert st.reason.startswith("голова") and "yaw" in st.reason
    h2 = Harness(calib)
    for i in range(60):
        st = h2.step(world(0.5, 1.9) + rng.normal(0, NOISE), i / 30)
    assert "низ" in st.reason or st.reason.startswith("голова")


def test_tracker_recenter_requires_a_face_and_calibration():
    t = gaze.GazeTracker(mesh=object())
    assert not t.recenter()                                           # нет калибровки и кадра
    calib, rng = calibrated(7)
    h = Harness(calib)
    assert not h.t.recenter()                                         # ещё не было кадров
    h.step(world(0.5, 0.5) + np.array([0.02, 0.03, 0, 0, 0]), 0.0)
    assert h.t.recenter()
