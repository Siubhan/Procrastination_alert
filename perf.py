"""Лёгкие замеры времени этапов цикла (для отладочного оверлея и бенчмарка)."""
import time
from contextlib import contextmanager


class PerfStats:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.ms = {}   # сглаженное время этапов, мс
        self.fps = 0.0
        self._last = None

    def add(self, name, ms):
        old = self.ms.get(name)
        self.ms[name] = ms if old is None else old + self.alpha * (ms - old)

    @contextmanager
    def measure(self, name):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - t0) * 1000)

    def tick(self):
        """Вызывать раз за кадр: считает FPS."""
        now = time.perf_counter()
        if self._last is not None and now > self._last:
            inst = 1.0 / (now - self._last)
            self.fps = inst if self.fps == 0 else self.fps + self.alpha * (inst - self.fps)
        self._last = now
