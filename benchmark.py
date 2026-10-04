"""
Замеры производительности по этапам.

  python benchmark.py                      - на тестовом фото лица (нужен scikit-image)
  python benchmark.py --image face.jpg     - на своём фото
  python benchmark.py --camera 0           - на живой камере (без окон), покажет реальный FPS
  python benchmark.py --json result.json   - сохранить результаты

Цифры зависят от процессора: запусти у себя и сравни до/после изменений.
"""
import argparse
import json
import os
import platform
import time
import warnings

warnings.filterwarnings("ignore", message=r".*GetPrototype.*")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np

import config as C
import gaze
from ui import draw_gaze_map


def letterbox(img, w, h):
    s = min(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
    out = np.zeros((h, w, 3), np.uint8)
    y, x = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def load_photo(path):
    if path:
        img = cv2.imread(path)
        if img is None:
            raise SystemExit(f"Не удалось прочитать {path}")
        return img
    try:
        from skimage import data
        return cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    except ImportError:
        raise SystemExit("Нужен --image face.jpg (или pip install scikit-image для тестового фото)")


def synthetic_calibration(f0, seed=0):
    """Калибровка из случайных данных: нужна только чтобы прогнать карту экрана (для замера времени)."""
    rng = np.random.default_rng(seed)
    targets = gaze.snake_targets(C.CALIB_COLS, C.CALIB_ROWS)
    samples = [f0 + np.array([0.2 * (x - .5), 0.15 * (y - .5), 0.3 * (x - .5), 0.05 * (y - .5), -0.04 * (y - .5)])
               + rng.normal(0, 0.003, (20, 5)) for x, y in targets]
    return gaze.build_calibration(samples, targets)[0]


def bench(fn, n, warmup=10):
    for _ in range(warmup):
        fn()
    ms = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ms.append((time.perf_counter() - t0) * 1000)
    a = np.array(ms)
    return {"mean": float(a.mean()), "median": float(np.median(a)), "p95": float(np.percentile(a, 95))}


def row(name, r):
    print(f"  {name:<46}{r['mean']:>8.2f}{r['median']:>9.2f}{r['p95']:>8.2f}")


def run_stages(photo, n):
    res = {}
    frame = letterbox(photo, 640, 480)
    tracker = gaze.GazeTracker()
    h, w = frame.shape[:2]

    # подготовка: настоящие лэндмарки и калибровка
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out = tracker.mesh.process(rgb)
    if not out.multi_face_landmarks:
        raise SystemExit("На картинке не найдено лицо: бенчмарку нужно фото лица")
    lms = out.multi_face_landmarks[0].landmark
    P = gaze.landmark_array(lms, w, h)
    f0 = gaze.compute_features(P)
    tracker.calib = synthetic_calibration(f0)
    mapper, filt, ex = tracker.calib.mapper, gaze.OneEuro(), gaze.FeatureExtractor()
    heat = np.random.default_rng(0).random((90, 160)).astype(np.float32) * 20
    trail = [(0.5 + 0.2 * np.sin(i / 5), 0.5 + 0.2 * np.cos(i / 5)) for i in range(40)]
    clock = iter(range(10 ** 9))

    print(f"\nЭтапы на кадре 640x480 ({n} повторов), мс:")
    print(f"  {'этап':<46}{'среднее':>8}{'медиана':>9}{'p95':>8}")
    res["flip"] = bench(lambda: cv2.flip(frame, 1), n)
    res["cvtColor BGR->RGB"] = bench(lambda: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), n)
    res["FaceMesh.process"] = bench(lambda: tracker.mesh.process(rgb), n)
    res["landmark_array (15 точек из 478)"] = bench(lambda: gaze.landmark_array(lms, w, h), n)
    res["признаки + моргание + сглаживание"] = bench(lambda: ex(P), n)
    res["карта экрана + One Euro"] = bench(lambda: filt(mapper.predict(f0), next(clock) / 30), n)
    res["GazeTracker.update (весь взгляд)"] = bench(lambda: tracker.update(frame, next(clock) / 30), n)
    res["draw_gaze_map (окно карты)"] = bench(lambda: draw_gaze_map(np.array([0.4, 0.6]), trail, heat), n)
    for k, v in res.items():
        row(k, v)

    # --- масштабирование по разрешению камеры ---
    print("\nВлияние разрешения камеры на GazeTracker.update, мс:")
    print(f"  {'разрешение':<46}{'среднее':>8}{'медиана':>9}{'p95':>8}")
    scale = {}
    for (W_, H_) in ((640, 480), (1280, 720), (1920, 1080)):
        fr = letterbox(photo, W_, H_)
        t = gaze.GazeTracker()
        t.calib = tracker.calib
        scale[f"{W_}x{H_}"] = bench(lambda: t.update(fr, next(clock) / 30), max(n // 2, 20))
        row(f"{W_}x{H_}", scale[f"{W_}x{H_}"])
    res["_resolution"] = scale

    # --- YOLO (если установлен) ---
    try:
        from phone import PhoneDetector
        det = PhoneDetector()
        det._load()
        print(f"\nYOLO (imgsz={C.PHONE_IMGSZ}), мс:")
        res["YOLO yolov8n"] = bench(lambda: det._infer(frame), max(n // 4, 20), warmup=3)
        row("PhoneDetector._infer", res["YOLO yolov8n"])
    except ImportError:
        print("\nYOLO: ultralytics не установлен, этап пропущен")
        res["YOLO yolov8n"] = None
    return res


def summarize(res):
    gaze_ms = res["GazeTracker.update (весь взгляд)"]["mean"]
    draw_ms = res["draw_gaze_map (окно карты)"]["mean"]
    print("\nИтог для цикла камеры (без чтения кадра и imshow):")
    loop = gaze_ms
    print(f"  взгляд: {gaze_ms:.1f} мс/кадр -> до {1000 / loop:.0f} FPS (+{draw_ms:.1f} мс, если открыта карта 'g')")
    y = res.get("YOLO yolov8n")
    if y:
        print(f"  YOLO: {y['mean']:.0f} мс на кадр")
        print(f"  если бы YOLO шёл в основном цикле: кадр с детекцией {gaze_ms + y['mean']:.0f} мс "
              f"-> {1000 / (gaze_ms + y['mean']):.0f} FPS на этих кадрах")
        print(f"  сейчас (фоновый поток, <= {C.PHONE_MAX_FPS} дет/с): цикл не ждёт нейросеть, ~{1000 / loop:.0f} FPS")


def run_camera(index, frames):
    from camera import open_camera
    cap = open_camera(index)
    if cap is None:
        raise SystemExit(f"Камера {index} не открылась")
    tracker = gaze.GazeTracker()
    read_ms, gaze_ms, total = [], [], []
    calibrated = False
    print(f"\nЖивая камера {index}: {frames} кадров, смотри в камеру...")
    for i in range(frames + 10):
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        if not calibrated:
            f, _ = tracker.raw_features(frame)
            if f is not None:
                tracker.calib = synthetic_calibration(f)
                calibrated = True
        tracker.update(frame, t1)
        t2 = time.perf_counter()
        if i >= 10:
            read_ms.append((t1 - t0) * 1000)
            gaze_ms.append((t2 - t1) * 1000)
            total.append((t2 - t0) * 1000)
    cap.release()
    h, w = frame.shape[:2]
    print(f"  разрешение кадра: {w}x{h}")
    for name, a in (("cap.read", read_ms), ("взгляд", gaze_ms), ("весь цикл", total)):
        a = np.array(a)
        print(f"  {name:<12} среднее {a.mean():6.2f}  медиана {np.median(a):6.2f}  p95 {np.percentile(a, 95):6.2f} мс")
    print(f"  реальный FPS цикла: {1000 / np.mean(total):.0f}")
    return {"read": float(np.mean(read_ms)), "gaze": float(np.mean(gaze_ms)), "fps": float(1000 / np.mean(total))}


def main():
    p = argparse.ArgumentParser(description="Замеры производительности")
    p.add_argument("--image", help="фото лица (по умолчанию тестовое из scikit-image)")
    p.add_argument("--camera", type=int, help="замерить на живой камере")
    p.add_argument("--n", type=int, default=200, help="повторов на этап")
    p.add_argument("--json", help="сохранить результаты в файл")
    args = p.parse_args()

    print(f"Система: {platform.platform()} | CPU: {platform.processor() or platform.machine()} | "
          f"Python {platform.python_version()} | OpenCV {cv2.__version__}")
    results = {}
    if args.camera is not None:
        results["camera"] = run_camera(args.camera, args.n)
    else:
        results["stages"] = run_stages(load_photo(args.image), args.n)
        summarize(results["stages"])
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nРезультаты записаны в {args.json}")


if __name__ == "__main__":
    main()
