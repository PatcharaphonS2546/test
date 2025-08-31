
import time
import numpy as np
from ..utils.screen import safe_screen_size

try:
    import pyautogui  # type: ignore
except Exception:  # pragma: no cover
    pyautogui = None

class MouseController:
    def __init__(self, enable: bool = False, rate_limit_ms: int = 10):
        self.enable = bool(enable)
        self.rate_ms = int(rate_limit_ms)
        self.sw, self.sh = safe_screen_size()
        self._last_move_at = 0.0

    def set_enable(self, v: bool):
        self.enable = bool(v)

    def update(self, x_norm: float, y_norm: float):
        if not self.enable or pyautogui is None:
            return
        now = time.time()
        if (now - self._last_move_at) * 1000.0 < self.rate_ms:
            return
        x_px = int(np.clip(x_norm, 0, 1) * self.sw)
        y_px = int(np.clip(y_norm, 0, 1) * self.sh)
        try:
            pyautogui.moveTo(x_px, y_px)
        except Exception:
            pass
        self._last_move_at = now
