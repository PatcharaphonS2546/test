import time
import math
import tkinter as tk

# ===== Mouse backends (pynput preferred, fall back to pyautogui) =====
MOUSE_BACKEND = None
mouse = None
Button = None
screen_w = 1920
screen_h = 1080

try:
    from pynput.mouse import Controller as PynputMouse, Button as PynputButton
    mouse = PynputMouse()
    Button = PynputButton
    MOUSE_BACKEND = "pynput"
except Exception as e:
    try:
        import pyautogui
        MOUSE_BACKEND = "pyautogui"
        screen_w, screen_h = pyautogui.size()
    except Exception as e2:
        print("[WARN] No mouse control backend available. Install 'pynput' or 'pyautogui'.")

# screen size fallback via Tk if not set by pyautogui
if MOUSE_BACKEND == "pynput":
    try:
        _t = tk.Tk()
        _t.withdraw()
        screen_w = _t.winfo_screenwidth()
        screen_h = _t.winfo_screenheight()
        _t.destroy()
    except Exception as e:
        # In environments without a display, this can fail.
        pass

def move_mouse_abs(x, y):
    if MOUSE_BACKEND == "pynput":
        mouse.position = (int(x), int(y))
    elif MOUSE_BACKEND == "pyautogui":
        import pyautogui
        pyautogui.moveTo(int(x), int(y))
    else:
        pass

def click_mouse_left():
    if MOUSE_BACKEND == "pynput":
        mouse.click(Button.left, 1)
    elif MOUSE_BACKEND == "pyautogui":
        import pyautogui
        pyautogui.click()
    else:
        pass

class MouseController:
    def __init__(self):
        self.enabled = False
        self.dwell_enabled = False
        self.dwell_time = 0.8  # seconds
        self.dwell_radius = 30 # pixels
        self._dwell_anchor = None
        self._dwell_start = None
        self._dwell_clicked = False

        self.smooth_alpha = 0.35
        self._sx = None
        self._sy = None

    def update(self, norm_x: float, norm_y: float):
        """
        Updates the mouse position based on normalized screen coordinates.
        Args:
            norm_x (float): Gaze X position, normalized to [0, 1].
            norm_y (float): Gaze Y position, normalized to [0, 1].
        """
        if not self.enabled:
            self._dwell_anchor = None
            self._dwell_start = None
            self._dwell_clicked = False
            return

        # Map normalized coords to absolute screen pixels
        target_x = norm_x * (screen_w - 1)
        target_y = norm_y * (screen_h - 1)

        # Apply smoothing
        if self._sx is None:
            self._sx, self._sy = target_x, target_y
        else:
            a = self.smooth_alpha
            self._sx = a * target_x + (1 - a) * self._sx
            self._sy = a * target_y + (1 - a) * self._sy

        move_mouse_abs(self._sx, self._sy)

        if self.dwell_enabled:
            self._handle_dwell(self._sx, self._sy)

    def _handle_dwell(self, x: float, y: float):
        now = time.time()
        pos = (x, y)
        if self._dwell_anchor is None:
            self._dwell_anchor = pos
            self._dwell_start = now
            self._dwell_clicked = False
            return

        dx = pos[0] - self._dwell_anchor[0]
        dy = pos[1] - self._dwell_anchor[1]
        dist = math.hypot(dx, dy)

        if dist <= self.dwell_radius:
            # Inside dwell radius
            if not self._dwell_clicked and (now - self._dwell_start) >= self.dwell_time:
                print("[INFO] Dwell click triggered.")
                click_mouse_left()
                self._dwell_clicked = True
                # Reset anchor to allow for another dwell click without moving away
                self._dwell_anchor = None
        else:
            # Moved too far, reset dwell
            self._dwell_anchor = pos
            self._dwell_start = now
            self._dwell_clicked = False

    def set_config(self, enabled: bool, dwell_enabled: bool, dwell_time: float, dwell_radius: int, smooth_alpha: float):
        self.enabled = enabled
        self.dwell_enabled = dwell_enabled
        self.dwell_time = dwell_time
        self.dwell_radius = dwell_radius
        self.smooth_alpha = smooth_alpha
        if not self.enabled:
            self._sx = None
            self._sy = None
