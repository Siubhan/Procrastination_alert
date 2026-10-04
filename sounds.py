"""Звуки: загрузка из папки sounds/ или генерация запасных."""
import os
import wave

import numpy as np

import config as C


def synth_wav(path, kind):
    """Запасные звуки, если файлов нет (можно заменить своими .wav)."""
    sr = 44100
    if kind == "attack":  # воющая сирена
        t = np.linspace(0, 1.4, int(sr * 1.4), endpoint=False)
        freq = 500 + 500 * np.abs(np.sin(2 * np.pi * 1.5 * t))
        phase = 2 * np.pi * np.cumsum(freq) / sr
        data = 0.3 * np.sign(np.sin(phase)) + 0.4 * np.sin(phase)
    else:  # выстрел: шумовой всплеск с затуханием + низкий удар
        t = np.linspace(0, 0.8, int(sr * 0.8), endpoint=False)
        noise = np.random.uniform(-1, 1, t.size) * np.exp(-t * 12)
        data = 0.8 * noise + 0.7 * np.sin(2 * np.pi * 70 * t) * np.exp(-t * 9)
    pcm = (np.clip(data, -1, 1) * 32767 * 0.8).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def find_sound_file(name):
    for ext in ("wav", "ogg", "mp3"):
        p = os.path.join(C.SOUNDS, f"{name}.{ext}")
        if os.path.exists(p):
            return p
    return None


def load_sound(name, kind):
    """pygame.mixer.Sound для sounds/<name>.(wav|ogg|mp3); если файла нет - создаёт запасной."""
    import pygame
    path = find_sound_file(name)
    if path is None:
        os.makedirs(C.SOUNDS, exist_ok=True)
        path = os.path.join(C.SOUNDS, f"{name}.wav")
        synth_wav(path, kind)
    return pygame.mixer.Sound(path)
