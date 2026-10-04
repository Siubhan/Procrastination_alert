"""Логика срабатывания тревог (без камеры и окон, поэтому легко тестируется)."""
import math


class AlertLogic:
    """Решает, когда звать тревогу. Телефон важнее взгляда; для каждого типа своя пауза.

    away_kind: куда ушёл взгляд ("down", "up", "side", "head", "noface") или None, если всё в порядке.
    away_seconds: число или словарь {kind: секунды}: взгляд вбок терпимее, чем вниз на телефон.
    """

    def __init__(self, away_seconds, cooldown):
        self.away_seconds = away_seconds
        self.cooldown = cooldown
        self.last = {"away": -math.inf, "phone": -math.inf}
        self.away_since = None

    def reset_away(self):
        self.away_since = None

    def _need(self, kind):
        if isinstance(self.away_seconds, dict):
            return self.away_seconds.get(kind, max(self.away_seconds.values()))
        return self.away_seconds

    def update(self, now, away_kind, phone_present):
        """Возвращает "phone", "away" или None."""
        if away_kind:
            if self.away_since is None:
                self.away_since = now
        else:
            self.away_since = None
        away_long = away_kind is not None and self.away_since is not None \
            and now - self.away_since >= self._need(away_kind)

        if phone_present and now - self.last["phone"] > self.cooldown:
            self.last["phone"] = now
            return "phone"
        if away_long and not phone_present and now - self.last["away"] > self.cooldown:
            self.last["away"] = now
            return "away"
        return None


class ActiveAlert:
    """Какая тревога показывается сейчас и когда её гасить. Гаснет сразу, как только причина пропала:

    - "away": взгляд вернулся (away_kind стал None);
    - "phone": телефона больше нет в кадре;
    - или истёк максимальный срок показа.
    Новая тревога заменяет текущую.
    """

    def __init__(self, max_seconds):
        self.max_seconds = max_seconds
        self.kind = None
        self.until = 0.0

    def start(self, kind, now):
        """Начинает показ; возвращает вытесненную тревогу (или None), чтобы заглушить её звук."""
        replaced = self.kind
        self.kind, self.until = kind, now + self.max_seconds
        return replaced

    def update(self, now, away_kind, phone_visible):
        """Возвращает None, если тревога продолжается, иначе (kind, "resolved" | "expired")."""
        if self.kind is None:
            return None
        resolved = (self.kind == "phone" and not phone_visible) or (self.kind == "away" and away_kind is None)
        if not resolved and now < self.until:
            return None
        kind, self.kind = self.kind, None
        return kind, "resolved" if resolved else "expired"

    def cancel(self):
        kind, self.kind = self.kind, None
        return kind
