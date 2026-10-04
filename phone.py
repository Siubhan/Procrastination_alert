"""Детектор телефона (YOLO) в фоновом потоке: основной цикл с камерой не ждёт нейросеть."""
import threading
import time
from collections import deque

import config as C

COCO_CELL_PHONE = 67


class PhoneDetector:
    def __init__(self, model=None):
        self._model = model
        self.hist = deque(maxlen=C.PHONE_WINDOW)
        self.boxes = []
        self.last_ms = 0.0  # время последнего вывода нейросети
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._frame = None
        self._busy = False
        self._stop = False
        self._last_start = 0.0
        self._thread = None
        self._error_reported = False
        self.fast = False   # True, пока показывается тревога про телефон: детектируем чаще

    # ---- нейросеть ----
    def _load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO("yolov8n.pt")  # скачается при первом запуске
        return self._model

    def _infer(self, frame):
        r = self._load().predict(frame, classes=[COCO_CELL_PHONE], conf=C.PHONE_CONF,
                                 imgsz=C.PHONE_IMGSZ, verbose=False)[0]
        return [tuple(map(int, b.xyxy[0].tolist())) for b in r.boxes]

    def process(self, frame):
        """Синхронная обработка одного кадра (используется потоком, тестами и бенчмарком)."""
        t0 = time.perf_counter()
        boxes = self._infer(frame)
        self.last_ms = (time.perf_counter() - t0) * 1000
        with self._lock:
            self.boxes = boxes
            self.hist.append(bool(boxes))

    # ---- фоновая работа ----
    def start(self):
        self._load()  # грузим модель сразу, чтобы ошибки/скачивание были видны при старте
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, frame):
        """Не блокирует: если нейросеть занята или рано - кадр пропускается."""
        now = time.monotonic()
        fps = C.PHONE_FAST_FPS if self.fast else C.PHONE_MAX_FPS
        if self._busy or now - self._last_start < 1.0 / fps:
            return
        self._busy = True
        self._last_start = now
        self._frame = frame.copy()  # основной цикл рисует поверх кадра, поэтому копия
        self._wake.set()

    def _worker(self):
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._stop:
                return
            try:
                self.process(self._frame)
            except Exception as e:  # поток не должен умирать из-за одного сбоя
                if not self._error_reported:
                    print("Ошибка детектора телефона:", e)
                    self._error_reported = True
            finally:
                self._busy = False

    def close(self):
        self._stop = True
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    # ---- состояние ----
    @property
    def visible_now(self):
        """Телефон есть в кадре прямо сейчас: он найден хотя бы в одной из последних PHONE_CLEAR_MISSES детекций.
        Быстрый признак для отмены тревоги; один пропуск детектора не гасит её."""
        with self._lock:
            return any(list(self.hist)[-C.PHONE_CLEAR_MISSES:])

    @property
    def present(self):
        """Телефон надёжно есть: найден в PHONE_HITS из последних PHONE_WINDOW детекций и виден сейчас."""
        with self._lock:
            recent = list(self.hist)
            return sum(recent) >= C.PHONE_HITS and any(recent[-C.PHONE_CLEAR_MISSES:])
