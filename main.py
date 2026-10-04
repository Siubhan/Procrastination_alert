"""
Procrastination alert: следит через веб-камеру.
  - взгляд отведён от монитора -> звук нападения + картинка away.png
  - в кадре телефон            -> звук выстрела  + картинка phone.png (приоритет выше)

Запуск:
  python main.py                 - камера из config.json (по умолчанию 0)
  python main.py --camera 1      - выбрать камеру по номеру
  python main.py --list-cameras  - показать доступные камеры и выйти

Клавиши (в окне с камерой):
  c - калибровка: шаг 1 глазами по 15 точкам, шаг 2 голову двигаешь, глаза на точке (~50 с)
  t - тест точности на новых точках (результат по зонам экрана, пишется в accuracy_log.csv)
  z - "я сел иначе": посмотри в центр экрана и нажми, поза примется за нормальную
  [ / ] - уменьшить / увеличить допуск "взгляд за экраном" (ложные тревоги: нажми ])
  n - переключить на следующую камеру
  g - окно "карта взгляда": точка, след и тепловая карта
  o - оверлей взгляда поверх рабочего стола (Windows)
  r - сбросить тепловую карту
  d - показать/скрыть отладочную информацию
  q / Esc - выход
"""
import os
import warnings

# Безвредные предупреждения/логи mediapipe + protobuf, скрываем до импорта mediapipe
warnings.filterwarnings("ignore", message=r".*GetPrototype.*")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import csv
import json
import time
from collections import deque

import cv2
import numpy as np

import config as C
from camera import list_cameras, open_camera
from gaze import GazeTracker, load_calibration
from logic import ActiveAlert, AlertLogic
from perf import PerfStats
from phone import PhoneDetector
from sounds import load_sound
from ui import (AlertPresenter, DesktopOverlay, accuracy_test, calibrate, detect_screen_size,
                draw_debug, draw_gaze_map, load_image)


def load_config():
    try:
        with open(C.CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_config(cfg):
    try:
        with open(C.CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except OSError:
        pass


def format_tolerances(calib, extra=0.0):
    t = calib.tolerances(extra)
    return f"слева {t['left']:.2f}, справа {t['right']:.2f}, сверху {t['top']:.2f}, снизу {t['bottom']:.2f}"


def dismiss_alert(active, alerts, sounds):
    """Убирает текущую тревогу (картинка и звук), например перед калибровкой."""
    kind = active.cancel()
    alerts.hide()
    if kind:
        sounds[kind].fadeout(C.ALERT_FADE_MS)


def log_event(event, state):
    """Печатает причину тревоги и дописывает её в events.csv: по ней видно, что именно сработало ложно."""
    if event == "phone":
        reason = "телефон в кадре"
    elif state is None:
        reason = "лицо не найдено"
    else:
        reason = state.reason or "взгляд"
    print(f"ТРЕВОГА [{event}]: {reason}")
    try:
        new = not os.path.exists(C.EVENTS_LOG)
        with open(C.EVENTS_LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "event", "reason"] + list(C.FEATURE_NAMES) + ["x", "y", "roll"])
            feats = [f"{v:.3f}" for v in state.feats] if state is not None else [""] * len(C.FEATURE_NAMES)
            pt = [f"{v:.3f}" for v in state.pt] if state is not None and state.pt is not None else ["", ""]
            roll = [f"{state.roll:.1f}"] if state is not None else [""]
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), event, reason] + feats + pt + roll)
    except OSError:
        pass


def parse_args():
    p = argparse.ArgumentParser(description="Procrastination alert")
    p.add_argument("--camera", type=int, help="номер камеры (0, 1, 2...)")
    p.add_argument("--list-cameras", action="store_true", help="показать доступные камеры и выйти")
    return p.parse_args()


def open_first_available(preferred):
    cap = open_camera(preferred)
    if cap is not None:
        return cap, preferred
    print(f"Камера {preferred} не открылась, ищу другие...")
    for i, _, _ in list_cameras():
        cap = open_camera(i)
        if cap is not None:
            return cap, i
    raise RuntimeError("Не удалось открыть ни одну камеру")


def main():
    args = parse_args()
    if args.list_cameras:
        cams = list_cameras()
        print("\n".join(f"Камера {i}: {w}x{h}" for i, w, h in cams) or "Камеры не найдены")
        return

    import pygame
    pygame.mixer.init()
    sounds = {"away": load_sound("attack", "attack"), "phone": load_sound("shot", "shot")}
    screen_size = detect_screen_size()
    alerts = AlertPresenter({"away": load_image("away.png"), "phone": load_image("phone.png")},
                            screen_size if C.FULLSCREEN_ALERT else None)

    cfg = load_config()
    cap, cam_index = open_first_available(args.camera if args.camera is not None else cfg.get("camera", 0))
    cfg["camera"] = cam_index
    save_config(cfg)
    print(f"Используется камера {cam_index}")

    tracker = GazeTracker()
    phone = PhoneDetector()
    phone.start()
    logic = AlertLogic(C.AWAY_SECONDS, C.COOLDOWN_SECONDS)
    active = ActiveAlert(C.ALERT_SHOW_SECONDS)
    perf = PerfStats()

    tracker.calib = load_calibration(cam_index)
    if tracker.calib is None:
        print("Для этой камеры нет актуальной калибровки, запускаю...")
        calibrate(cap, tracker, cam_index, screen_size)

    cached_cams = None
    show_map, debug, overlay = False, True, None
    trail = deque(maxlen=40)
    heat = np.zeros((C.MAP_SIZE[1] // 4, C.MAP_SIZE[0] // 4), np.float32)

    print("Клавиши: c калибровка, t тест точности, n камера, g карта, o оверлей, d отладка, q выход")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            now = time.monotonic()

            phone.submit(frame)  # не блокирует: YOLO работает в фоне
            with perf.measure("gaze"):
                state = tracker.update(frame, now)
            if state is not None:
                away_kind, pt = (state.kind or None), state.pt
            else:
                away_kind, pt = ("noface" if C.NO_FACE_AS_AWAY else None), None

            # --- карта взгляда ---
            if pt is not None:
                trail.append((pt[0], pt[1]))
                hx = int(pt[0] * (heat.shape[1] - 1))
                hy = int(pt[1] * (heat.shape[0] - 1))
                if 0 <= hx < heat.shape[1] and 0 <= hy < heat.shape[0]:
                    heat[hy, hx] += 1.0
            heat *= C.HEAT_DECAY
            if show_map:
                cv2.imshow("Gaze map", draw_gaze_map(pt, trail, heat))
            if overlay is not None:
                overlay.update(pt)

            # --- тревоги ---
            event = logic.update(now, away_kind, phone.present)
            if event:
                log_event(event, state)
                replaced = active.start(event, now)
                if replaced:
                    sounds[replaced].fadeout(C.ALERT_FADE_MS)
                sounds[event].play()
                alerts.show(event)
            # отмена сразу, как только причина пропала: взгляд вернулся или телефона нет в кадре
            ended = active.update(now, away_kind, phone.visible_now)
            if ended:
                alerts.hide()
                if ended[1] == "resolved":
                    sounds[ended[0]].fadeout(C.ALERT_FADE_MS)
            phone.fast = active.kind == "phone"   # пока тревога про телефон, детектируем чаще

            if debug:
                draw_debug(frame, state, tracker, phone, perf, cam_index, away_kind)
            cv2.imshow("Watchdog", frame)
            perf.tick()

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                dismiss_alert(active, alerts, sounds)
                calibrate(cap, tracker, cam_index, screen_size)
                logic.reset_away()
            elif key == ord("t"):
                dismiss_alert(active, alerts, sounds)
                accuracy_test(cap, tracker, cam_index, screen_size)
                logic.reset_away()
            elif key == ord("z"):
                print("Поза принята за нормальную" if tracker.recenter() else "Лицо не найдено или нет калибровки")
                logic.reset_away()
            elif key in (ord("["), ord("]")):
                extra = tracker.adjust_tolerance(C.TOL_STEP if key == ord("]") else -C.TOL_STEP)
                print(f"Допуск (поправка {extra:+.2f}): {format_tolerances(tracker.calib, extra)}" if tracker.calib else "Нет калибровки")
            elif key == ord("d"):
                debug = not debug
            elif key == ord("r"):
                heat[:] = 0
                trail.clear()
            elif key == ord("g"):
                show_map = not show_map
                if not show_map:
                    try:
                        cv2.destroyWindow("Gaze map")
                    except cv2.error:
                        pass
            elif key == ord("o"):
                if overlay is not None:
                    overlay.close()
                    overlay = None
                elif screen_size is None:
                    print("Оверлей поддерживается только на Windows (используй окно 'g')")
                else:
                    try:
                        overlay = DesktopOverlay(screen_size)
                    except Exception as e:
                        print("Не удалось создать оверлей:", e)
            elif key == ord("n"):
                if cached_cams is None:
                    cached_cams = [c[0] for c in list_cameras()]  # медленно, поэтому один раз
                if len(cached_cams) < 2:
                    print("Другие камеры не найдены")
                    continue
                nxt = cached_cams[(cached_cams.index(cam_index) + 1) % len(cached_cams)] \
                    if cam_index in cached_cams else cached_cams[0]
                new_cap = open_camera(nxt)
                if new_cap is None:
                    continue
                cap.release()
                cap, cam_index = new_cap, nxt
                cfg["camera"] = cam_index
                save_config(cfg)
                tracker.reset()
                logic.reset_away()
                tracker.calib = load_calibration(cam_index)
                if tracker.calib is None:
                    calibrate(cap, tracker, cam_index, screen_size)
                print(f"Переключено на камеру {cam_index}")
    finally:
        phone.close()
        if overlay is not None:
            overlay.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
