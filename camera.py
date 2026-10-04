"""Работа с камерами: открытие и поиск."""
import sys

import cv2

import config as C


def open_camera(index, width=C.CAMERA_WIDTH, height=C.CAMERA_HEIGHT):
    """Открывает камеру; на Windows через DirectShow (быстрее и стабильнее для USB-камер).
    width/height = 0 - не менять разрешение. Возвращает VideoCapture или None."""
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return None
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # меньше задержка: не копим старые кадры
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


def list_cameras(max_index=C.MAX_CAMERA_PROBE):
    """[(индекс, ширина, высота)] для всех работающих камер (медленно: пробует каждый индекс)."""
    found = []
    for i in range(max_index):
        cap = open_camera(i, 0, 0)
        if cap is not None:
            found.append((i, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            cap.release()
    return found
