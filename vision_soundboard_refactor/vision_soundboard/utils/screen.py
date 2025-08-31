
from typing import Tuple

try:
    import pyautogui  # type: ignore
except Exception:  # pragma: no cover
    pyautogui = None

def safe_screen_size() -> Tuple[int, int]:
    """Return (width, height), fallback to 1920x1080 if unavailable."""
    if pyautogui:
        try:
            w, h = pyautogui.size()
            return int(w), int(h)
        except Exception:
            pass
    return 1920, 1080
