"""Наклон головы к плечу: точка взгляда не должна уезжать, пока глаза остаются на экране."""
import numpy as np
import pytest

import config as C
import gaze
from helpers import FakeMesh, blank_frame, face_for_gaze, legacy_head_axis_features, nodded, tilted


def calibrations(seed=1):
    rng = np.random.default_rng(seed)
    targets = gaze.snake_targets(C.CALIB_COLS, C.CALIB_ROWS)
    faces = [[face_for_gaze(x, y, rng, 0.15) for _ in range(20)] for x, y in targets]
    new = gaze.build_calibration([np.array([gaze.compute_features(P) for P in fs]) for fs in faces], targets)[0]
    old = gaze.build_calibration([np.array([legacy_head_axis_features(P) for P in fs]) for fs in faces], targets)[0]
    return new, old, rng


def mean_error(calib, features_fn, deg, rng, n=200):
    errs = []
    for _ in range(n):
        x, y = rng.uniform(0.1, 0.9, 2)
        P = tilted(x, y, deg * rng.choice([-1, 1]), rng, 0.15)
        errs.append(np.hypot(*(calib.mapper.predict(features_fn(P)) - (x, y))))
    return float(np.mean(errs))


@pytest.mark.parametrize("deg", [10, 20, 30, 45])
def test_tilt_does_not_move_the_gaze_point(deg):
    new, _, rng = calibrations()
    base = mean_error(new, gaze.compute_features, 0, rng)
    assert mean_error(new, gaze.compute_features, deg, rng) < base + 0.02


def test_regression_old_head_axis_features_failed_under_tilt():
    """Было: смещение зрачка считалось в осях головы и поворачивалось вместе с наклоном
    (ошибка точки ~8% при 10°, ~16% при 20°, ~24% при 30°). Тест фиксирует и баг, и исправление."""
    new, old, rng = calibrations()
    old_err = mean_error(old, legacy_head_axis_features, 20, rng)
    new_err = mean_error(new, gaze.compute_features, 20, rng)
    assert old_err > 0.10 and new_err < 0.03


@pytest.mark.parametrize("x,y", [(0.5, 0.5), (0.15, 0.2), (0.85, 0.2), (0.2, 0.9), (0.8, 0.9), (0.5, 0.95)])
@pytest.mark.parametrize("deg", [-28, 28])
def test_tracker_stays_on_screen_when_head_is_tilted(x, y, deg):
    """Весь конвейер: наклонил голову к плечу, но смотрю в экран (в том числе в нижний край) - тревоги нет."""
    new, _, _ = calibrations()
    mesh = FakeMesh()
    t = gaze.GazeTracker(mesh=mesh)
    t.calib = new
    mesh.P = tilted(x, y, deg)
    st = None
    for i in range(60):
        st = t.update(blank_frame(), i / 30)
    assert not st.away
    assert st.pt == pytest.approx([x, y], abs=0.07)
    assert st.roll == pytest.approx(deg, abs=0.5)


def test_roll_is_reported_and_does_not_create_a_profile():
    mesh = FakeMesh()
    t = gaze.GazeTracker(mesh=mesh)
    new, _, _ = calibrations()
    t.calib = new
    mesh.P = tilted(0.5, 0.5, 45)
    for i in range(30):
        st = t.update(blank_frame(), i / 30)
    assert st.kind != "head" and st.roll == pytest.approx(45, abs=0.5)


def nod_calibrations():
    rng = np.random.default_rng(3)
    targets = gaze.snake_targets(C.CALIB_COLS, C.CALIB_ROWS)
    eyes = [np.array([gaze.compute_features(face_for_gaze(x, y, rng, 0.15)) for _ in range(20)]) for x, y in targets]
    sweep_targets = list(C.CALIB_HEAD_TARGETS)
    sweeps = [np.array([gaze.compute_features(nodded(x, y, 0.9 * np.sin(2 * np.pi * i / 90 + k), rng, 0.15))
                        for i in range(150)]) for k, (x, y) in enumerate(sweep_targets)]
    return (gaze.build_calibration(eyes, targets)[0],
            gaze.build_calibration(eyes + sweeps, targets + sweep_targets, head_phase=True)[0], rng)


def test_nodding_needs_the_head_step_of_calibration():
    """Наклон головы вперёд-назад при взгляде на экран: без шага 2 калибровки ошибка точки 30-70% экрана,
    с шагом 2 (голова ходит, глаза на точке) 1-5%."""
    eyes_only, with_head, rng = nod_calibrations()
    for p in (-1.0, -0.5, 0.5, 1.0):
        e1, e2 = [], []
        for _ in range(150):
            x, y = rng.uniform(0.1, 0.9, 2)
            f = gaze.compute_features(nodded(x, y, p, rng, 0.15))
            e1.append(np.hypot(*(eyes_only.mapper.predict(f) - (x, y))))
            e2.append(np.hypot(*(with_head.mapper.predict(f) - (x, y))))
        assert np.mean(e1) > 0.2 and np.mean(e2) < 0.06


@pytest.mark.parametrize("p", [-1.0, 1.0])
def test_tracker_stays_on_screen_when_nodding(p):
    _, with_head, _ = nod_calibrations()
    mesh = FakeMesh()
    t = gaze.GazeTracker(mesh=mesh)
    t.calib = with_head
    for x, y in ((0.5, 0.5), (0.3, 0.9), (0.7, 0.15)):
        t.reset()
        mesh.P = nodded(x, y, p)
        for i in range(60):
            st = t.update(blank_frame(), i / 30)
        assert not st.away and st.pt == pytest.approx([x, y], abs=0.08)


def test_head_phase_flag_is_saved_and_loaded(tmp_path):
    _, with_head, _ = nod_calibrations()
    eyes_only, _, _ = nod_calibrations()
    p = str(tmp_path / "c.json")
    gaze.save_calibration(0, with_head, p)
    gaze.save_calibration(1, eyes_only, p)
    assert gaze.load_calibration(0, p).head_phase is True
    assert gaze.load_calibration(1, p).head_phase is False
