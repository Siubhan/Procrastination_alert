"""Оценка взгляда по лэндмаркам лица: признаки, фильтры, отображение на экран, калибровка."""
import json
import math
from dataclasses import dataclass

import cv2
import numpy as np

import config as C

CALIB_VERSION = 3  # при смене набора признаков старые калибровки считаются недействительными

# Индексы точек FaceMesh (refine_landmarks=True добавляет радужку 468-477).
# Берём только нужные 15 точек, а не все 478: быстрее.
LM_IDS = (468, 473, 33, 133, 159, 145, 362, 263, 386, 374, 1, 234, 454, 10, 152)
(I_R_IRIS, I_L_IRIS, I_R_OUT, I_R_IN, I_R_UP, I_R_LOW,
 I_L_IN, I_L_OUT, I_L_UP, I_L_LOW, I_NOSE, I_CHEEK_L, I_CHEEK_R,
 I_FOREHEAD, I_CHIN) = range(len(LM_IDS))
N_FEATURES = len(C.FEATURE_NAMES)


class CalibrationError(Exception):
    pass


# ======================================================================
# Признаки
# ======================================================================
def landmark_array(landmarks, w, h):
    """Лэндмарки MediaPipe -> массив (15, 2) в пикселях."""
    return np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in LM_IDS])


def roll_degrees(P):
    """Наклон головы к плечу в градусах по линии внешних уголков глаз (0 = ровно, + = правая сторона кадра ниже)."""
    d = P[I_L_OUT] - P[I_R_OUT]
    return math.degrees(math.atan2(d[1], d[0]))


def _eye(P, left, right, iris, up, low):
    """Положение зрачка и раскрытие глаза, всё в долях ширины глаза.

    Зрачок меряем относительно УГОЛКОВ глаза, а не век: при взгляде вниз веки опускаются вместе со
    зрачком, и отношение "зрачок между веками" почти не меняется.

    Смещение берём в осях КАМЕРЫ, а не головы. Когда человек наклоняет голову к плечу, но продолжает
    смотреть в ту же точку экрана, глаза доворачиваются, и в кадре смещение зрачка остаётся прежним,
    а в осях головы оно бы поворачивалось вместе с наклоном и сбивало карту экрана.
    Раскрытие глаза (высота / ширина) от наклона не зависит и считается вдоль перпендикуляра к линии уголков.
    """
    a, b = P[left], P[right]
    u = b - a
    width = math.hypot(u[0], u[1])
    if width < 1e-6:
        return None
    e2 = np.array([-u[1], u[0]]) / width
    d = P[iris] - (a + b) / 2
    return d[0] / width, d[1] / width, (P[low] - P[up]) @ e2 / width


def _nose_offset(P):
    """Поворот и наклон головы вперёд-назад: нос относительно центра лица в осях камеры, в долях ширины/высоты лица."""
    c = (P[I_CHEEK_L] + P[I_CHEEK_R] + P[I_FOREHEAD] + P[I_CHIN]) / 4
    sx = float(np.linalg.norm(P[I_CHEEK_R] - P[I_CHEEK_L]))
    sy = float(np.linalg.norm(P[I_CHIN] - P[I_FOREHEAD]))
    if sx < 1e-6 or sy < 1e-6:
        return 0.5, 0.5
    n = P[I_NOSE] - c
    return 0.5 + n[0] / sx, 0.5 + n[1] / sy


def compute_features(P):
    """(15, 2) -> [yaw, pitch, iris_h, iris_v, eye_open] или None, если геометрия вырожденная."""
    r = _eye(P, I_R_OUT, I_R_IN, I_R_IRIS, I_R_UP, I_R_LOW)
    l = _eye(P, I_L_IN, I_L_OUT, I_L_IRIS, I_L_UP, I_L_LOW)
    if r is None or l is None:
        return None
    h, v, op = (np.array(r) + np.array(l)) / 2
    yaw, pitch = _nose_offset(P)
    return np.array([yaw, pitch, h, v, op])


class FeatureExtractor:
    """Признаки + защита от моргания + лёгкое сглаживание."""

    def __init__(self, alpha=C.SMOOTH_ALPHA):
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.smoothed = None
        self.last_eye = None   # последние "хорошие" iris_h, iris_v, eye_open
        self.open_base = None  # медленно подстраивающееся обычное раскрытие глаз
        self.blink = False

    def forget_smoothing(self):
        self.smoothed = None

    def __call__(self, P, smooth=True):
        f = compute_features(P)
        if f is None:
            return None
        op = f[4]
        if self.open_base is None:
            self.open_base = op
        # моргание = резкое схлопывание глаз. Взгляд вниз тоже прикрывает глаза, но заметно слабее
        self.blink = op < max(C.BLINK_ABS, C.BLINK_REL * self.open_base)
        if self.blink and self.last_eye is not None:
            f[2:] = self.last_eye
        elif not self.blink:
            self.last_eye = f[2:].copy()
            self.open_base += 0.02 * (op - self.open_base)
        if smooth:
            self.smoothed = f if self.smoothed is None else self.alpha * f + (1 - self.alpha) * self.smoothed
            f = self.smoothed
        return f


# ======================================================================
# Фильтр точки взгляда (One Euro: гладко в покое, быстро при резком движении)
# ======================================================================
class OneEuro:
    def __init__(self, min_cutoff=C.POINT_MIN_CUTOFF, beta=C.POINT_BETA, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.t_prev = self.x_prev = self.dx_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, float)
        if self.t_prev is None:
            self.t_prev, self.x_prev, self.dx_prev = t, x, np.zeros_like(x)
            return x
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(dx_hat))
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.t_prev, self.x_prev, self.dx_prev = t, x_hat, dx_hat
        return x_hat


# ======================================================================
# Подстройка под позу
# ======================================================================
class PostureDrift:
    """Медленно вычитает отклонение позы (yaw, pitch, eye_open) от калибровочной.

    Без этого достаточно сесть чуть иначе, и карта экрана "уезжает": симуляция давала 30-70 ложных
    тревог за 10 минут. Подстройка ограничена пределами и не трогает большие отклонения,
    поэтому взгляд на телефон или в сторону по-прежнему ловится.
    """
    IDX = (0, 1, 4)

    def __init__(self, ref):
        self.ref = np.asarray(ref, float)
        self.limit = np.array(C.DRIFT_LIMIT)
        self.capture = self.limit * C.DRIFT_CAPTURE
        self.off = np.zeros(len(self.IDX))
        self.t_prev = None

    def apply(self, feats, t):
        cur = feats[list(self.IDX)] - self.ref
        dt = 0.0 if self.t_prev is None else min(max(t - self.t_prev, 0.0), 0.5)
        self.t_prev = t
        if dt > 0 and np.all(np.abs(cur - self.off) <= self.capture):
            self.off += (1 - math.exp(-dt / C.DRIFT_TAU)) * (cur - self.off)
            self.off = np.clip(self.off, -self.limit, self.limit)
        out = feats.copy()
        out[list(self.IDX)] -= self.off
        return out

    def recenter(self, feats):
        """Мгновенно принять текущую позу за нормальную (клавиша z, когда смотришь в центр экрана)."""
        self.off = np.clip(feats[list(self.IDX)] - self.ref, -self.limit, self.limit)


# ======================================================================
# Отображение признаков на точку экрана
# ======================================================================
def _poly(Xs, degree):
    cols = [Xs]
    if degree >= 2:
        n = Xs.shape[1]
        cols.append(np.stack([Xs[:, i] * Xs[:, j] for i in range(n) for j in range(i, n)], axis=1))
    cols.append(np.ones((len(Xs), 1)))
    return np.hstack(cols)


def _ridge(A, Y, lam):
    R = lam * np.eye(A.shape[1])
    R[-1, -1] = 0  # свободный член не штрафуем
    return np.linalg.solve(A.T @ A + R, A.T @ Y)


@dataclass
class FitReport:
    cv_error: float   # средняя ошибка на "невиданных" точках экрана (доля экрана)
    degree: int
    lam: float
    groups: list      # [(target_x, target_y, error)] по точкам калибровки (кросс-валидация)


class GazeMapper:
    """Признаки -> (x, y) на экране в долях 0..1. Модель (линейная или квадратичная) и
    регуляризация выбираются кросс-валидацией "оставь одну точку экрана за бортом"."""

    OUT_LIMIT = 2.0  # предсказание обрезается до [-2, 3] экранов: квадратичная модель не должна улетать в бесконечность

    def __init__(self):
        self.degree = self.lam = self.mu = self.sd = self.W = None
        self.cv_error = float("inf")

    @staticmethod
    def _cv(A, Y, groups, lam):
        res = []
        for g in np.unique(groups):
            te = groups == g
            W = _ridge(A[~te], Y[~te], lam)
            p = np.clip(A[te] @ W, -GazeMapper.OUT_LIMIT, 1 + GazeMapper.OUT_LIMIT)
            res.append((float(Y[te][0][0]), float(Y[te][0][1]), float(np.linalg.norm(p - Y[te], axis=1).mean())))
        return res

    def fit(self, X, Y, groups):
        X, Y, groups = np.asarray(X, float), np.asarray(Y, float), np.asarray(groups)
        self.mu = X.mean(axis=0)
        self.sd = np.maximum(X.std(axis=0), 1e-3)
        Xs = (X - self.mu) / self.sd
        best = None
        for degree in (1, 2):
            A = _poly(Xs, degree)
            for lam in C.RIDGE_LAMBDAS:
                per_group = self._cv(A, Y, groups, lam)
                err = float(np.mean([e for _, _, e in per_group]))
                if best is None or err < best[0]:
                    best = (err, degree, lam, per_group)
        self.cv_error, self.degree, self.lam, per_group = best
        self.W = _ridge(_poly(Xs, self.degree), Y, self.lam)
        return FitReport(self.cv_error, self.degree, self.lam, per_group)

    def predict(self, feats):
        xs = (np.asarray(feats, float) - self.mu) / self.sd
        p = _poly(xs[None], self.degree)[0] @ self.W
        return np.clip(p, -self.OUT_LIMIT, 1 + self.OUT_LIMIT)

    def to_dict(self):
        return {"degree": self.degree, "lam": self.lam, "cv_error": self.cv_error,
                "mu": self.mu.tolist(), "sd": self.sd.tolist(), "W": self.W.tolist()}

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.degree, m.lam, m.cv_error = int(d["degree"]), float(d["lam"]), float(d["cv_error"])
        m.mu, m.sd, m.W = np.array(d["mu"]), np.array(d["sd"]), np.array(d["W"])
        return m


# ======================================================================
# Калибровка
# ======================================================================
@dataclass
class Calibration:
    mapper: GazeMapper
    head_phase: bool = False  # был ли шаг 2 (голова ходит, глаза на точке). Без него наклоны и повороты определяются плохо

    def tolerances(self, extra=0.0):
        """Допуск выхода точки взгляда за каждый край экрана (доли экрана).
        Хуже калибровка = мягче (не меньше K * ошибка калибровки)."""
        floor = C.TOL_CV_K * self.mapper.cv_error
        return {side: max(tol + extra, floor, 0.02) for side, tol in C.GAZE_TOL.items()}

    @property
    def mapper_trusted(self):
        """Слишком неточная карта экрана не должна вызывать тревоги."""
        return self.mapper.cv_error <= C.MAPPER_MAX_CV

    @property
    def pose_ref(self):
        """Средняя поза (yaw, pitch, eye_open) за время калибровки."""
        return self.mapper.mu[list(PostureDrift.IDX)]


def reject_outliers(samples, k=3.5, floor=0.002):
    """Выбрасывает кадры, сильно выбивающиеся из медианы (моргание, скачок взгляда)."""
    s = np.asarray(samples, float)
    if len(s) < 8:
        return s
    med = np.median(s, axis=0)
    mad = np.maximum(np.median(np.abs(s - med), axis=0) * 1.4826, floor)
    keep = (np.abs(s - med) <= k * mad).all(axis=1)
    return s[keep] if keep.sum() >= 5 else s


def build_calibration(point_samples, targets, head_phase=False):
    """point_samples: список массивов (n_i, 5) - признаки при взгляде в targets[i] (x, y в долях экрана).
    Точки могут быть двух видов: голова неподвижна (глаза ведут взгляд) и голова движется при взгляде
    в одну точку: так модель отделяет вклад глаз от вклада головы."""
    X, Y, G = [], [], []
    for gi, (s, t) in enumerate(zip(point_samples, targets)):
        if s is None or len(s) < 5:
            continue
        s = reject_outliers(s)
        X.append(s)
        Y.append(np.tile(np.asarray(t, float), (len(s), 1)))
        G.append(np.full(len(s), gi))
    if len(X) < C.CALIB_MIN_POINTS:
        raise CalibrationError(f"Мало удачных точек ({len(X)} из {len(targets)}): держи лицо в кадре")
    mapper = GazeMapper()
    report = mapper.fit(np.vstack(X), np.vstack(Y), np.concatenate(G))
    return Calibration(mapper, head_phase), report


def snake_targets(cols, rows):
    """Точки обхода "змейкой": взгляд перелетает на соседнюю точку, а не через весь экран."""
    out = []
    for i, y in enumerate(rows):
        xs = cols if i % 2 == 0 else tuple(reversed(cols))
        out += [(x, y) for x in xs]
    return out


def zone_errors(items):
    """items: [(target_x, target_y, error)] -> средняя ошибка по зонам экрана {"верх", "середина", "низ"}."""
    zones = {"верх": [], "середина": [], "низ": []}
    for _, ty, e in items:
        zones["верх" if ty < 0.34 else "середина" if ty < 0.67 else "низ"].append(e)
    return {k: float(np.mean(v)) for k, v in zones.items() if v}


def save_calibration(cam_index, calib, path=None):
    path = path or C.CALIB_PATH
    data = {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        pass
    if data.get("version") != CALIB_VERSION:
        data = {"version": CALIB_VERSION}
    data[str(cam_index)] = {"mapper": calib.mapper.to_dict(), "head_phase": calib.head_phase}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_calibration(cam_index, path=None):
    path = path or C.CALIB_PATH
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != CALIB_VERSION:
            return None
        d = data[str(cam_index)]
        return Calibration(GazeMapper.from_dict(d["mapper"]), bool(d.get("head_phase", False)))
    except (OSError, ValueError, KeyError, TypeError):
        return None


# ======================================================================
# Трекер
# ======================================================================
@dataclass
class GazeState:
    feats: np.ndarray
    pt: object          # np.ndarray (x, y) в долях экрана или None, если нет калибровки
    kind: str           # "" - на экране/рядом; иначе куда ушёл взгляд: "down", "up", "side", "head"
    iris_px: np.ndarray  # положение зрачков в кадре (для отрисовки)
    reason: str = ""    # человекочитаемая причина
    roll: float = 0.0   # наклон головы к плечу, градусы (для отладки и журнала)

    @property
    def away(self):
        return bool(self.kind)


class GazeTracker:
    def __init__(self, mesh=None):
        self.mesh = mesh if mesh is not None else self._make_mesh()
        self.extractor = FeatureExtractor()
        self.filter = OneEuro()
        self.drift = None
        self.tol_extra = 0.0   # поправка допуска, меняется клавишами [ и ] (-0.15..+0.3)
        self._calib = None
        self._last_feats = None

    @property
    def calib(self):
        return self._calib

    @calib.setter
    def calib(self, value):
        self._calib = value
        self.drift = PostureDrift(value.pose_ref) if value is not None else None

    @staticmethod
    def _make_mesh():
        import mediapipe as mp
        return mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def reset(self):
        self.extractor.reset()
        self.filter.reset()
        if self._calib is not None:
            self.drift = PostureDrift(self._calib.pose_ref)

    def adjust_tolerance(self, delta):
        self.tol_extra = min(max(self.tol_extra + delta, -0.1), 0.3)
        return self.tol_extra

    def recenter(self):
        """Принять текущую позу за нормальную (смотришь в центр экрана и нажимаешь z)."""
        if self.drift is not None and self._last_feats is not None:
            self.drift.recenter(self._last_feats)
            return True
        return False

    def _landmarks(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False  # MediaPipe не копирует буфер
        res = self.mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        return landmark_array(res.multi_face_landmarks[0].landmark, w, h)

    def raw_features(self, frame_bgr):
        """Сырые признаки без сглаживания и подстройки (для калибровки): (feats | None, моргание)."""
        P = self._landmarks(frame_bgr)
        if P is None:
            return None, False
        f = self.extractor(P, smooth=False)
        return f, self.extractor.blink

    def update(self, frame_bgr, t):
        """Один кадр -> GazeState или None, если лица нет."""
        P = self._landmarks(frame_bgr)
        feats = self.extractor(P) if P is not None else None
        if feats is None:
            self.extractor.forget_smoothing()
            self.filter.reset()
            return None
        self._last_feats = feats
        iris = P[[I_R_IRIS, I_L_IRIS]]
        c = self._calib
        if c is None:
            return GazeState(feats, None, "", iris, "", roll_degrees(P))
        feats = self.drift.apply(feats, t)          # поправка на позу, отличную от калибровочной
        pt = self.filter(c.mapper.predict(feats), t)
        kind, reason = self._classify(c, feats, pt)
        return GazeState(feats, pt, kind, iris, reason, roll_degrees(P))

    def _classify(self, c, feats, pt):
        """Куда смотрит человек: "" (на экране или рядом), "head" (профиль), "down", "up", "side"."""
        lo, hi = C.HEAD_YAW_LIMITS
        if not lo <= feats[0] <= hi:
            return "head", f"голова почти в профиль (yaw {feats[0]:.2f})"
        if not c.mapper_trusted:
            return "", ""                           # карте не верим: решают только профиль, нет лица и телефон
        tol = c.tolerances(self.tol_extra)
        x, y = pt
        if y > 1 + tol["bottom"]:
            return "down", f"взгляд ниже экрана: y {y:.2f} (допуск {tol['bottom']:.2f})"
        if y < -tol["top"]:
            return "up", f"взгляд выше экрана: y {y:.2f} (допуск {tol['top']:.2f})"
        if x < -tol["left"] or x > 1 + tol["right"]:
            return "side", f"взгляд в сторону от экрана: x {x:.2f} (допуск {tol['left' if x < 0 else 'right']:.2f})"
        return "", ""
