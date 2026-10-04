"""Окна и отрисовка: оверлей, карта взгляда, тревоги, калибровка, тест точности, отладка."""
import csv
import os
import sys
import time

import cv2
import numpy as np

import config as C
from gaze import CalibrationError, build_calibration, save_calibration, snake_targets, zone_errors


# ======================================================================
# Экран и картинки
# ======================================================================
def detect_screen_size():
    """(ширина, высота) основного монитора в реальных пикселях; None, если не Windows."""
    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # без этого Windows врёт про размер при масштабировании
        except Exception:
            pass
        u = ctypes.windll.user32
        return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    except Exception:
        return None


def load_image(name):
    img = cv2.imread(os.path.join(C.ASSETS, name))
    if img is None:
        raise FileNotFoundError(f"Нет картинки assets/{name}")
    return img


def fit_to_screen(img, size=None):
    """Вписывает картинку в экран size=(w, h) с чёрными полями; без size возвращает как есть."""
    if size is None:
        return img
    sw, sh = size
    h, w = img.shape[:2]
    s = min(sw / w, sh / h)
    img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((sh, sw, 3), np.uint8)
    y, x = (sh - img.shape[0]) // 2, (sw - img.shape[1]) // 2
    canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
    return canvas


class AlertPresenter:
    """Показывает и убирает картинку тревоги. Подгонка под экран делается один раз, а не каждый кадр.
    Когда убирать, решает ActiveAlert (logic.py)."""
    WIN = "ALERT"

    def __init__(self, images, size=None):
        self.frames = {k: fit_to_screen(v, size) for k, v in images.items()}
        self.fullscreen = size is not None
        self.visible = False

    def show(self, kind):
        cv2.namedWindow(self.WIN, cv2.WND_PROP_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty(self.WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty(self.WIN, cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow(self.WIN, self.frames[kind])
        self.visible = True

    def hide(self):
        if not self.visible:
            return
        self.visible = False
        try:
            cv2.destroyWindow(self.WIN)
        except cv2.error:
            pass


# ======================================================================
# Карта взгляда и оверлей
# ======================================================================
# cv2.applyColorMap строит таблицу цветов на каждый вызов (~10 мс), поэтому берём её один раз
_JET = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1), cv2.COLORMAP_JET).reshape(256, 3)


def draw_gaze_map(pt, trail, heat, size=C.MAP_SIZE):
    """Экран в миниатюре: тепловая карта, след и текущая точка взгляда."""
    W, H = size
    canvas = np.full((H, W, 3), 28, np.uint8)
    h = cv2.GaussianBlur(heat, (0, 0), 1.5)  # всё тяжёлое делаем на маленькой сетке, растягиваем один раз
    m = float(h.max())
    if m > 0:
        h8 = np.clip(h * (255.0 / max(m, C.HEAT_MIN_SCALE)), 0, 255).astype(np.uint8)
        color = cv2.resize(_JET[h8], (W, H), interpolation=cv2.INTER_LINEAR)
        _, mask = cv2.threshold(cv2.resize(h8, (W, H), interpolation=cv2.INTER_LINEAR), 20, 255, cv2.THRESH_BINARY)
        cv2.copyTo(cv2.addWeighted(canvas, 0.35, color, 0.65, 0), mask, canvas)
    cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), (200, 200, 200), 2)
    if len(trail) > 1:
        pts = np.array([(int(min(max(x, 0), 1) * (W - 1)), int(min(max(y, 0), 1) * (H - 1)))
                        for x, y in trail], np.int32)
        cv2.polylines(canvas, [pts], False, (255, 255, 255), 1, cv2.LINE_AA)
    if pt is not None:
        inside = 0 <= pt[0] <= 1 and 0 <= pt[1] <= 1
        c = (int(min(max(pt[0], 0), 1) * (W - 1)), int(min(max(pt[1], 0), 1) * (H - 1)))
        col = (0, 255, 0) if inside else (0, 0, 255)
        cv2.circle(canvas, c, 10, col, 2, cv2.LINE_AA)
        cv2.circle(canvas, c, 3, col, -1, cv2.LINE_AA)
        cv2.putText(canvas, "on screen" if inside else "OFF SCREEN", (10, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    else:
        cv2.putText(canvas, "no face", (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    return canvas


class DesktopOverlay:
    """Прозрачное окно поверх рабочего стола с точкой взгляда (только Windows, через tkinter)."""

    def __init__(self, size):
        import tkinter as tk

        self.W, self.H = size
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-transparentcolor", "black")
        self.root.geometry(f"{self.W}x{self.H}+0+0")
        self.canvas = tk.Canvas(self.root, width=self.W, height=self.H, bg="black", highlightthickness=0)
        self.canvas.pack()
        self.ring = self.canvas.create_oval(0, 0, 0, 0, outline="#00ff66", width=4)
        self.dot = self.canvas.create_oval(0, 0, 0, 0, fill="#00ff66", outline="")
        self.root.update()
        self._make_click_through()

    def _make_click_through(self):
        """Win32: фон окна полностью прозрачен, точка полупрозрачна (OVERLAY_OPACITY), клики проходят сквозь."""
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        user32.GetAncestor.restype = wintypes.HWND
        user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.GetWindowLongW.restype = ctypes.c_long
        user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
        user32.SetWindowLongW.restype = ctypes.c_long
        user32.SetLayeredWindowAttributes.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_ubyte, wintypes.DWORD]
        user32.SetLayeredWindowAttributes.restype = wintypes.BOOL

        self.root.update_idletasks()
        hwnd = user32.GetAncestor(self.root.winfo_id(), 2)  # GA_ROOT: настоящее окно верхнего уровня
        style = user32.GetWindowLongW(hwnd, -20)             # GWL_EXSTYLE
        # LAYERED | TRANSPARENT | TOOLWINDOW (не в панели задач) | NOACTIVATE (не забирает фокус)
        user32.SetWindowLongW(hwnd, -20, style | 0x00080000 | 0x20 | 0x80 | 0x08000000)
        alpha = int(max(0.05, min(1.0, C.OVERLAY_OPACITY)) * 255)
        # чёрный (0,0,0) = полностью прозрачный фон, остальное (точка) с прозрачностью alpha
        if not user32.SetLayeredWindowAttributes(hwnd, 0x000000, alpha, 0x1 | 0x2):  # COLORKEY | ALPHA
            print("Не удалось настроить прозрачность оверлея")

    def update(self, pt):
        """pt: (x, y) в долях экрана или None (скрыть)."""
        if pt is None:
            self.canvas.coords(self.ring, 0, 0, 0, 0)
            self.canvas.coords(self.dot, 0, 0, 0, 0)
        else:
            x = int(min(max(pt[0], 0.0), 1.0) * self.W)
            y = int(min(max(pt[1], 0.0), 1.0) * self.H)
            col = "#00ff66" if 0 <= pt[0] <= 1 and 0 <= pt[1] <= 1 else "#ff3030"
            self.canvas.itemconfig(self.ring, outline=col)
            self.canvas.itemconfig(self.dot, fill=col)
            self.canvas.coords(self.ring, x - 28, y - 28, x + 28, y + 28)
            self.canvas.coords(self.dot, x - 6, y - 6, x + 6, y + 6)
        self.root.update()

    def close(self):
        try:
            self.root.destroy()
        except Exception:
            pass


# ======================================================================
# Калибровка и тест точности
# ======================================================================
def run_targets(cap, targets, step_fn, screen_size=None, title="Follow the dot with your eyes",
                settle=None, collect=None, show=True, subtitle=""):
    """Показывает точку в каждой цели по очереди.

    step_fn(frame, now, collecting, target) вызывается на каждом кадре; во время `collecting`
    непустые результаты записываются. Возвращает список списков или None, если нажали Esc.
    """
    settle = C.CALIB_SETTLE if settle is None else settle
    collect = C.CALIB_COLLECT if collect is None else collect
    W, H = screen_size or (1280, 720)
    win = "Calibration"
    canvas = np.zeros((H, W, 3), np.uint8)
    if show:
        cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    results = []
    try:
        for idx, target in enumerate(targets):
            t0, samples = time.monotonic(), []
            while time.monotonic() - t0 < settle + collect:
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = cv2.flip(frame, 1)
                el = time.monotonic() - t0
                collecting = el > settle
                s = step_fn(frame, time.monotonic(), collecting, target)
                if collecting and s is not None:
                    samples.append(s)
                if show:
                    canvas.fill(0)
                    c = (int(target[0] * W), int(target[1] * H))
                    r = int(18 + 6 * np.sin(el * 8))
                    cv2.circle(canvas, c, r, (0, 255, 0) if collecting else (0, 255, 255), -1)
                    cv2.circle(canvas, c, 3, (0, 0, 0), -1)
                    cv2.putText(canvas, f"{title} ({idx + 1}/{len(targets)}). Esc - cancel",
                                (int(W * 0.2), int(H * 0.46)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
                    if subtitle:
                        cv2.putText(canvas, subtitle, (int(W * 0.2), int(H * 0.54)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 200, 255), 2)
                    cv2.imshow(win, canvas)
                    if cv2.waitKey(1) & 0xFF == 27:
                        return None
            results.append(samples)
    finally:
        if show:
            cv2.destroyWindow(win)
    return results


def print_zone_report(title, errors):
    parts = ", ".join(f"{k} {v * 100:.1f}%" for k, v in errors.items())
    print(f"{title}: {parts} (доля экрана)")


def calibrate(cap, tracker, cam_index, screen_size=None):
    """Шаг 1: голова неподвижна, глаза ходят по 15 точкам. Шаг 2: глаза на точке, голова медленно ходит.
    Вместе это учит модель отделять вклад глаз от вклада головы (вид 3/4, взгляд краем глаза)."""
    targets = snake_targets(C.CALIB_COLS, C.CALIB_ROWS)

    def step(frame, now, collecting, target):
        f, blink = tracker.raw_features(frame)
        return f if (collecting and f is not None and not blink) else None

    data = run_targets(cap, targets, step, screen_size, title="Step 1: head still, follow the dot with your eyes")
    if data is None:
        print("Калибровка отменена")
        return False
    if C.CALIB_HEAD_PHASE:
        head_targets = list(C.CALIB_HEAD_TARGETS)
        head_data = run_targets(cap, head_targets, step, screen_size,
                                title="Step 2: eyes on the dot, slowly turn your head",
                                subtitle="turn left-right, nod up-down, tilt ear to shoulder. Eyes stay on the dot",
                                settle=0.8, collect=C.CALIB_HEAD_SECONDS)
        if head_data is None:
            print("Калибровка отменена")
            return False
        data, targets = data + head_data, targets + head_targets
    try:
        calib, report = build_calibration([np.array(d) for d in data], targets, head_phase=C.CALIB_HEAD_PHASE)
    except CalibrationError as e:
        print("Калибровка не удалась:", e)
        return False
    tracker.calib = calib
    tracker.reset()
    save_calibration(cam_index, calib)
    model = "квадратичная" if report.degree == 2 else "линейная"
    print(f"Калибровка готова: модель {model}, lambda={report.lam}, "
          f"ошибка по кросс-валидации {report.cv_error * 100:.1f}% экрана"
          + ("" if calib.mapper_trusted else " (ВЫСОКАЯ: карте экрана не верю, перекалибруйся)"))
    if not calib.head_phase:
        print("  ВНИМАНИЕ: без шага 2 повороты и наклоны головы определяются плохо")
    t = calib.tolerances()
    print(f"  допуск за краем экрана: слева/справа {t['left'] * 100:.0f}%, сверху {t['top'] * 100:.0f}%, "
          f"снизу {t['bottom'] * 100:.0f}%")
    print_zone_report("  по зонам", zone_errors(report.groups))
    return True


def evaluate_points(targets, predictions):
    """Ошибки по точкам теста: [(tx, ty, err)] для тех целей, где есть предсказания."""
    items = []
    for (tx, ty), preds in zip(targets, predictions):
        if len(preds) >= 3:
            p = np.median(np.array(preds), axis=0)
            items.append((tx, ty, float(np.hypot(p[0] - tx, p[1] - ty))))
    return items


def accuracy_test(cap, tracker, cam_index, screen_size=None):
    """Тест на точках, которых не было в калибровке: показывает реальную точность по зонам экрана."""
    if tracker.calib is None:
        print("Сначала откалибруйся (клавиша c)")
        return None
    targets = snake_targets(C.TEST_COLS, C.TEST_ROWS)
    tracker.filter.reset()

    def step(frame, now, collecting, target):
        st = tracker.update(frame, now)  # весь боевой конвейер: признаки, карта, фильтр
        return st.pt if (st is not None and st.pt is not None) else None

    data = run_targets(cap, targets, step, screen_size, title="Accuracy test: look at the dot")
    if data is None:
        print("Тест отменён")
        return None
    items = evaluate_points(targets, data)
    if not items:
        print("Тест не удался: лицо не найдено")
        return None
    zones = zone_errors(items)
    mean = float(np.mean([e for _, _, e in items]))
    print(f"Тест точности: средняя ошибка {mean * 100:.1f}% экрана")
    print_zone_report("  по зонам", zones)
    _log_accuracy(cam_index, len(items), mean, zones, tracker.calib.mapper.cv_error)
    return mean, zones


def _log_accuracy(cam_index, n, mean, zones, cv_error):
    new = not os.path.exists(C.ACCURACY_LOG)
    try:
        with open(C.ACCURACY_LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "camera", "points", "mean_err", "top", "middle", "bottom", "cv_error"])
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), cam_index, n, f"{mean:.4f}",
                        *(f"{zones.get(k, float('nan')):.4f}" for k in ("верх", "середина", "низ")),
                        f"{cv_error:.4f}"])
    except OSError:
        pass


# ======================================================================
# Отладочная информация
# ======================================================================
def draw_debug(frame, state, tracker, phone, perf, cam_index, away_kind):
    for (x1, y1, x2, y2) in phone.boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, "phone", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if state is not None:
        for p in state.iris_px:
            cv2.circle(frame, tuple(p.astype(int)), 3, (0, 255, 0), -1)

    def put(txt, y, col, sc=0.7, th=2):  # тёмная обводка, чтобы читалось и на светлом фоне
        cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 2)
        cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, th)

    put(f"gaze: {away_kind.upper() if away_kind else 'OK'}  (cam {cam_index})", 25,
        (0, 0, 255) if away_kind else (0, 200, 0))
    phone_on = phone.present
    put(f"phone: {'YES' if phone_on else 'no'}", 50, (0, 0, 255) if phone_on else (0, 200, 0))
    put("c calib  t test  z recenter  [ ] tolerance  n cam  g map  o overlay  d debug", 75, (255, 255, 0), 0.4, 1)
    ms = perf.ms
    put(f"fps {perf.fps:.0f} | gaze {ms.get('gaze', 0):.1f}ms | phone {phone.last_ms:.0f}ms (bg)",
        100, (200, 200, 200), 0.5, 1)
    c = tracker.calib
    if c is not None:
        t = c.tolerances(tracker.tol_extra)
        trust = "" if c.mapper_trusted else "  MAP NOT TRUSTED: recalibrate (c)"
        if not c.head_phase:
            trust += "  NO HEAD STEP: tilts unreliable (c)"
        put(f"tol L/R {t['left']:.2f} top {t['top']:.2f} bottom {t['bottom']:.2f} (cv err {c.mapper.cv_error * 100:.0f}%){trust}",
            120, (0, 200, 255) if c.mapper_trusted else (0, 0, 255), 0.5, 1)
    if state is not None:
        if state.reason:
            put(state.reason, 140, (0, 0, 255), 0.5, 1)
        if state.pt is not None:
            put(f"gaze point: {state.pt[0]:.2f}, {state.pt[1]:.2f}   roll {state.roll:+.0f} deg", 160, (255, 255, 255), 0.5, 1)
        for i, name in enumerate(C.FEATURE_NAMES):
            put(f"{name}: {state.feats[i]:.3f}", frame.shape[0] - 12 - 20 * (len(C.FEATURE_NAMES) - 1 - i),
                (255, 255, 255), 0.5, 1)
