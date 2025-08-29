import cv2
import random
import math
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import time
import json
import sys
import platform
import threading

STREAM_URL = "http://172.20.10.3:81/stream"  # แก้ให้ตรง ESP32-CAM ของคุณ

# --- สำหรับ stream ภาพแบบต่อเนื่อง ---
latest_frame = None
stream_thread = None
stream_stop = threading.Event()
# --------------------- Transparent Fullscreen Overlay (Tkinter) ---------------------
def esp32_stream_worker(url, stop_evt):
    global latest_frame
    cap = None
    while not stop_evt.is_set():
        if cap is None:
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                time.sleep(1.0)
                cap.release()
                cap = None
                continue
        ret, frame = cap.read()
        if ret and frame is not None:
            latest_frame = frame.copy()
        else:
            time.sleep(0.2)
            cap.release()
            cap = None

# Optional: real mouse control
try:
    import pyautogui
    pyautogui.FAILSAFE = False
    SCREEN_W, SCREEN_H = pyautogui.size()
    HAVE_PYAUTO = True
except Exception:
    SCREEN_W, SCREEN_H = 1920, 1080
    HAVE_PYAUTO = False

# Optional: OpenGL eye sphere (ถ้ามี)
try:
    import gl_sphere
    GL_SPHERE_AVAILABLE = True
except Exception:
    GL_SPHERE_AVAILABLE = False
    print("gl_sphere module not found. OpenGL rendering will be disabled.")

latest_gaze_direction = None
ray_lines = []
model_centers = []
max_rays = 100
prev_model_center_avg = (320, 240)
max_observed_distance = 0

# --------------------- Transparent Fullscreen Overlay (Tkinter) ---------------------
class FullscreenOverlay:
    """
    หน้าต่างโปร่งใสเต็มจอสำหรับวาด 'จุดคาลิเบรท 9 จุด' ทับเดสก์ท็อปจริง
    - Windows: ใช้ -transparentcolor เพื่อโปร่งใสจริง (คลิกทะลุไม่ได้)
    - macOS/Linux: fallback ด้วย alpha โปร่งใสทั้งหน้าต่าง
    ใช้งาน: สร้างครั้งเดียว แล้วเรียก update_point(x, y) ทุกครั้งที่ต้องการเลื่อนจุด
    """
    def __init__(self, screen_w, screen_h):
        self.screen_w = int(screen_w)
        self.screen_h = int(screen_h)
        self.enabled = True   # สำหรับเปิด/ปิดชั่วคราว
        self._system = platform.system().lower()

        # สร้าง root แยก เพื่อไม่ไปรบกวน selection_gui()
        self.root = tk.Tk()
        self.root.withdraw()  # ซ่อนระหว่างตั้งค่า

        # ค่าพื้นหลังที่ใช้เป็น "สีโปร่งใส" บน Windows
        self._transparent_key = "#ff00ff"  # magenta

        # เต็มจอ & ทับบนสุด
        self.root.geometry(f"{self.screen_w}x{self.screen_h}+0+0")
        self.root.overrideredirect(True)
        self.root.lift()
        self.root.attributes("-topmost", True)

        # โหมดโปร่งใส
        if "windows" in self._system:
            # โปร่งใสด้วยสี (ตัวหน้าต่างจะทึบ ยกเว้นพื้นที่ที่ทาสีนี้)
            self.root.config(bg=self._transparent_key)
            try:
                self.root.wm_attributes("-transparentcolor", self._transparent_key)
            except Exception:
                # บางเครื่อง/ไดรเวอร์ไม่รองรับ → fallback alpha
                self.root.attributes("-alpha", 0.25)
                self._transparent_key = None
        else:
            # macOS / Linux: ใช้ alpha โปร่งใสทั้งหน้าต่าง
            self.root.attributes("-alpha", 0.25)
            self.root.config(bg="black")
            self._transparent_key = None

        # แคนวาสวาดจุด
        bg = self._transparent_key if self._transparent_key else "black"
        self.canvas = tk.Canvas(self.root, width=self.screen_w, height=self.screen_h,
                                bg=bg, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)

        # วาดครั้งแรก
        self._last_id_main = None
        self._last_id_ring = None
        self._last_text = None

        # แสดงหน้าต่าง
        self.root.deiconify()
        self._safe_update()

    def _safe_update(self):
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            pass

    def show(self):  # เปิดใช้งาน overlay
        self.enabled = True
        self.root.deiconify()
        self._safe_update()

    def hide(self):  # ซ่อน overlay
        self.enabled = False
        self.root.withdraw()
        self._safe_update()

    def destroy(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    def clear(self):
        if not self.enabled: return
        self.canvas.delete("all")

    def update_point(self, x, y, label=None, r=32):
        """
        วาดจุดคาลิเบรทที่พิกัด 'จอจริง' (SCREEN_W x SCREEN_H)
        """
        if not self.enabled:
            return
        x = int(max(0, min(self.screen_w-1, x)))
        y = int(max(0, min(self.screen_h-1, y)))
        self.canvas.delete("all")
        # วงกลมเขียวทึบ + วงแหวน
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="#00ff00", width=0)
        self.canvas.create_oval(x-(r+18), y-(r+18), x+(r+18), y+(r+18), outline="#33cc33", width=4)
        if label:
            self.canvas.create_text(x, y-(r+30), text=label, fill="white", font=("Arial", 18, "bold"))
        self._safe_update()

    def update_predicted(self, px, py, r=10):
        """
        วาดจุดคาดคะเน (predicted) สีขาวบน overlay (ไว้ดู mapping)
        """
        if not self.enabled:
            return
        px = int(max(0, min(self.screen_w-1, px)))
        py = int(max(0, min(self.screen_h-1, py)))
        # เติมแบบไม่ล้างจุดเป้าหมาย
        self.canvas.create_oval(px-r, py-r, px+r, py+r, fill="#f0f0f0", width=0)
        self._safe_update()

# --------------------- Image / Eye ops ---------------------
def crop_to_aspect_ratio(image, width=640, height=480):
    h, w = image.shape[:2]
    desired_ratio = width / height
    current_ratio = w / h
    if current_ratio > desired_ratio:
        new_w = int(desired_ratio * h)
        off = (w - new_w) // 2
        cropped = image[:, off:off+new_w]
    else:
        new_h = int(w / desired_ratio)
        off = (h - new_h) // 2
        cropped = image[off:off+new_h, :]
    return cv2.resize(cropped, (width, height))

def apply_binary_threshold(image, darkestPixelValue, addedThreshold):
    threshold = darkestPixelValue + addedThreshold
    _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    return thresholded_image

def get_darkest_area(image):
    ignoreBounds = 20
    imageSkipSize = 10
    searchArea = 20
    internalSkipSize = 5
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_sum = float('inf')
    darkest_point = None
    for y in range(ignoreBounds, gray.shape[0] - ignoreBounds, imageSkipSize):
        for x in range(ignoreBounds, gray.shape[1] - ignoreBounds, imageSkipSize):
            current_sum = 0
            num_pixels = 0
            for dy in range(0, searchArea, internalSkipSize):
                if y + dy >= gray.shape[0]: break
                for dx in range(0, searchArea, internalSkipSize):
                    if x + dx >= gray.shape[1]: break
                    current_sum += gray[y+dy][x+dx]
                    num_pixels += 1
            if current_sum < min_sum and num_pixels > 0:
                min_sum = current_sum
                darkest_point = (x + searchArea // 2, y + searchArea // 2)
    return darkest_point

def mask_outside_square(image, center, size):
    x, y = center
    half = size // 2
    mask = np.zeros_like(image)
    tlx = max(0, x - half); tly = max(0, y - half)
    brx = min(image.shape[1], x + half); bry = min(image.shape[0], y + half)
    mask[tly:bry, tlx:brx] = 255
    return cv2.bitwise_and(image, mask)

def filter_contours_by_area_and_return_largest(contours, pixel_thresh, ratio_thresh):
    max_area = 0; largest = None
    for c in contours:
        area = cv2.contourArea(c)
        if area >= pixel_thresh:
            x, y, w, h = cv2.boundingRect(c)
            lw = max(w / h, h / w)
            if lw <= ratio_thresh and area > max_area:
                max_area = area; largest = c
    return [largest] if largest is not None else []

def check_contour_pixels(contour, image_shape, debug_mode_on):
    if len(contour) < 5:
        return [0, 0, None]
    contour_mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, (255), 1)
    ellipse_mask_thick = np.zeros(image_shape, dtype=np.uint8)
    ellipse_mask_thin = np.zeros(image_shape, dtype=np.uint8)
    ellipse = cv2.fitEllipse(contour)
    cv2.ellipse(ellipse_mask_thick, ellipse, (255), 10)
    cv2.ellipse(ellipse_mask_thin, ellipse, (255), 4)
    overlap_thick = cv2.bitwise_and(contour_mask, ellipse_mask_thick)
    overlap_thin = cv2.bitwise_and(contour_mask, ellipse_mask_thin)
    absolute_pixel_total_thick = np.sum(overlap_thick > 0)
    absolute_pixel_total_thin = np.sum(overlap_thin > 0)
    total_border_pixels = np.sum(contour_mask > 0)
    ratio_under_ellipse = absolute_pixel_total_thin / total_border_pixels if total_border_pixels > 0 else 0
    return [absolute_pixel_total_thick, ratio_under_ellipse, overlap_thin]

def check_ellipse_goodness(binary_image, contour, debug_mode_on):
    ellipse_goodness = [0, 0, 0]
    if len(contour) < 5: return 0
    ellipse = cv2.fitEllipse(contour)
    mask = np.zeros_like(binary_image)
    cv2.ellipse(mask, ellipse, (255), -1)
    ellipse_area = np.sum(mask == 255)
    covered = np.sum((binary_image == 255) & (mask == 255))
    if ellipse_area == 0: return ellipse_goodness
    ellipse_goodness[0] = covered / ellipse_area
    axes = ellipse[1]
    ellipse_goodness[2] = min(axes[1] / axes[0], axes[0] / axes[1])
    return ellipse_goodness

# ---------------- 9-point calibration model ----------------
def _design_row(yaw, pitch):
    return np.array([1.0, yaw, pitch, yaw*pitch, yaw*yaw, pitch*pitch], dtype=np.float64)

def _fit_poly(samples):
    if len(samples) < 6: return None, None, None, None
    A = np.vstack([_design_row(y, p) for (y, p, _, _) in samples])
    bx = np.array([sx for (_, _, sx, _) in samples], dtype=np.float64)
    by = np.array([sy for (_, _, _, sy) in samples], dtype=np.float64)
    coef_x, *_ = np.linalg.lstsq(A, bx, rcond=None)
    coef_y, *_ = np.linalg.lstsq(A, by, rcond=None)
    rx = float(np.sqrt(np.mean((A @ coef_x - bx) ** 2)))
    ry = float(np.sqrt(np.mean((A @ coef_y - by) ** 2)))
    return coef_x, coef_y, rx, ry

def _map_with_model(coef_x, coef_y, yaw, pitch):
    row = _design_row(yaw, pitch)
    x = float(row @ coef_x); y = float(row @ coef_y)
    x = max(0.0, min(SCREEN_W - 1, x))
    y = max(0.0, min(SCREEN_H - 1, y))
    return x, y

def _yaw_pitch_from_dir(direction):
    dx, dy, dz = float(direction[0]), float(direction[1]), float(direction[2])
    if abs(dz) < 1e-6: dz = 1e-6
    yaw = math.atan2(dx, dz)      # left(-)/right(+)
    pitch = math.atan2(-dy, dz)   # up(-)/down(+)
    return yaw, pitch

def _save_calib(coef_x, coef_y, path="calib_9pt.json"):
    data = {"screen_w": SCREEN_W, "screen_h": SCREEN_H,
            "coef_x": [float(v) for v in coef_x],
            "coef_y": [float(v) for v in coef_y],
            "ts": time.time()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _load_calib(path="calib_9pt.json"):
    if not os.path.exists(path): return None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return np.array(d["coef_x"], dtype=np.float64), np.array(d["coef_y"], dtype=np.float64)
    except Exception:
        return None, None

# ---------------- NinePointOverlay (จุดจริง + overlay จอจริง) ----------------
class NinePointOverlay:
    def __init__(self, overlay=None):
        margin = 0.10
        xs = [margin, 0.5, 1.0 - margin]
        ys = [margin, 0.5, 1.0 - margin]
        self.targets_norm = [(x, y) for y in ys for x in xs]
        self.targets_screen = [(int(x*(SCREEN_W-1)), int(y*(SCREEN_H-1)))
                               for (x, y) in self.targets_norm]

        self.idx = 0
        self.samples = []          # (yaw, pitch, sx, sy)
        self.test_records = []     # (tx,ty, px,py, err, yaw,pitch)
        self.coef_x = None
        self.coef_y = None
        self.rms_fit = (None, None)
        self.mode = "idle"         # idle | calib | test | live
        self.mouse_on = False
        self._last_yaw_pitch = None
        self.overlay = overlay     # FullscreenOverlay

    def set_overlay(self, overlay):
        self.overlay = overlay

    def load_if_exists(self):
        cx, cy = _load_calib()
        if cx is not None:
            self.coef_x, self.coef_y = cx, cy
            self.mode = "idle"
            print("[CAL-9] found saved calib, ready.")

    def start(self):
        self.idx = 0
        self.samples.clear()
        self.test_records.clear()
        self.coef_x = self.coef_y = None
        self.rms_fit = (None, None)
        self.mode = "calib"
        if self.overlay:  # โชว์ overlay
            self.overlay.show()
        print("[CAL-9] เริ่มคาลิเบรท 9 จุด: จ้องวงกลมสีเขียวแล้วกด Enter")

    def _pick_yaw_pitch(self, gaze_dir):
        if gaze_dir is not None:
            yaw, pitch = _yaw_pitch_from_dir(gaze_dir)
            self._last_yaw_pitch = (yaw, pitch)
            return yaw, pitch, True
        if self._last_yaw_pitch is not None:
            yaw, pitch = self._last_yaw_pitch
            print("[CAL-9] ใช้ค่า gaze ล่าสุด (fallback)")
            return yaw, pitch, False
        return 0.0, 0.0, False

    def record_current_calib(self, gaze_dir):
        yaw, pitch, ok = self._pick_yaw_pitch(gaze_dir)
        sx, sy = self.targets_screen[self.idx]
        if not ok:
            print("ยังอ่านค่า gaze เฟรมนี้ไม่ได้ → ใช้ค่าเดิมล่าสุด")
        self.samples.append((yaw, pitch, sx, sy))
        print(f"[CAL-9] เก็บจุด {self.idx+1}/9")

        if self.idx == len(self.targets_screen) - 1:
            cx, cy, rx, ry = _fit_poly(self.samples)
            if cx is None:
                print("[CAL-9] ตัวอย่างยังไม่พอ (≥6)"); return
            self.coef_x, self.coef_y = cx, cy
            self.rms_fit = (rx, ry)
            self.mode = "test"
            self.idx = 0
            print(f"[CAL-9] ฟิตแล้ว → RMS x={rx:.1f}px y={ry:.1f}px → เข้าทดสอบ")
        else:
            self.idx += 1

    def record_current_test(self, gaze_dir):
        if self.coef_x is None:
            print("ยังไม่มีโมเดล"); return
        yaw, pitch, ok = self._pick_yaw_pitch(gaze_dir)
        tx, ty = self.targets_screen[self.idx]
        px, py = _map_with_model(self.coef_x, self.coef_y, yaw, pitch)
        err = math.hypot(px - tx, py - ty)
        self.test_records.append((tx, ty, px, py, err, yaw, pitch))
        print(f"[TEST] จุด {self.idx+1}/9: error={err:.1f}px")

        if self.idx == len(self.targets_screen) - 1:
            errs = np.array([r[4] for r in self.test_records], dtype=np.float64)
            mae = float(np.mean(errs)); rmse = float(np.sqrt(np.mean(errs**2)))
            print(f"[TEST] สรุป: MAE={mae:.1f}px  RMSE={rmse:.1f}px  → รีฟิตด้วยข้อมูลทดสอบ")
            aug = self.samples + [(r[5], r[6], r[0], r[1]) for r in self.test_records]
            cx, cy, rx, ry = _fit_poly(aug)
            if cx is not None:
                self.coef_x, self.coef_y = cx, cy
                self.rms_fit = (rx, ry)
                print(f"[REFIT] RMS ใหม่: x={rx:.1f}px y={ry:.1f}px")
            self.mode = "live"
            self.idx = 0
            if self.overlay:
                # จบคาลิบแล้วซ่อน overlay (หรือจะโชว์ต่อใน test/live ก็ได้)
                self.overlay.hide()
            print("[LIVE] พร้อมควบคุมเมาส์ด้วยสายตา (กด m เปิด/ปิด)")
        else:
            self.idx += 1

    def toggle_mouse(self):
        if self.mode != "live":
            print("ต้องจบการทดสอบก่อนถึงจะเข้าโหมด Live ได้")
            return
        if not HAVE_PYAUTO:
            print("pyautogui ไม่พร้อม จะไม่ขยับเมาส์จริง")
        self.mouse_on = not self.mouse_on
        print(f"[LIVE] mouse_control={self.mouse_on}")

    def save(self):
        if self.coef_x is None:
            print("ยังไม่มีโมเดลให้บันทึก"); return
        _save_calib(self.coef_x, self.coef_y)
        print("[SAVE] บันทึก calib_9pt.json แล้ว")

    def reset(self):
        self.__init__(overlay=self.overlay)
        print("[CAL-9] reset")

    def draw_overlay(self, frame):
        """
        1) วาด reference บนเฟรมกล้อง (จุด normalized)
        2) วาดจุดเป้าหมายทับ 'จอจริง' ด้วย FullscreenOverlay (ไม่ใช่จอดำ)
        3) (ออปชัน) วาดจุดคาดคะเนบน overlay เพื่อดู mapping
        """
        global latest_gaze_direction
        if latest_gaze_direction is not None:
            self._last_yaw_pitch = _yaw_pitch_from_dir(latest_gaze_direction)

        H, W = frame.shape[:2]

        # ----- (A) วาดบนเฟรมกล้อง (เพื่ออ้างอิง) -----
        if self.mode in ("calib", "test"):
            xn, yn = self.targets_norm[self.idx]
            tx_f, ty_f = int(xn*(W-1)), int(yn*(H-1))
            cv2.circle(frame, (tx_f,ty_f), 18, (0,255,0), -1)
            cv2.circle(frame, (tx_f,ty_f), 32, (60,200,60), 2)

        # จุดคาดคะเนบนเฟรม (แมปจากจอ→เฟรม) เพื่อเช็คการเรียนรู้
        if self.coef_x is not None and latest_gaze_direction is not None:
            yaw, pitch = _yaw_pitch_from_dir(latest_gaze_direction)
            sx, sy = _map_with_model(self.coef_x, self.coef_y, yaw, pitch)  # พิกัดจอจริง
            px_f = int((sx/(SCREEN_W-1))*(W-1))
            py_f = int((sy/(SCREEN_H-1))*(H-1))
            cv2.circle(frame, (px_f,py_f), 6, (240,240,240), -1)

        # ข้อความช่วย
        y0 = 30
        def put(txt):
            nonlocal y0
            cv2.putText(frame, txt, (12,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255,255,255), 2, cv2.LINE_AA)
            y0 += 28
        put(f"Mode: {self.mode.upper()}  |  Screen {SCREEN_W}x{SCREEN_H}")
        if self.mode == "idle":
            put("กด 9 เริ่ม 9 จุด • Enter เก็บค่า • n ข้าม • b ย้อน • s บันทึก • m เมาส์ • 0 รีเซ็ต • q ออก")
        elif self.mode == "calib":
            put(f"คาลิเบรท: จ้องจุดสีเขียวแล้วกด Enter  ({self.idx+1}/9)")
        elif self.mode == "test":
            rx, ry = self.rms_fit
            if rx is not None:
                put(f"ทดสอบ: จ้องแล้ว Enter  ({self.idx+1}/9) | RMS ~ x={rx:.1f} y={ry:.1f}px")
        elif self.mode == "live":
            put("Live: กด m เปิด/ปิดเมาส์จริง  |  s บันทึก  •  0 รีเซ็ต")

        # ----- (B) วาดทับ 'จอจริง' ด้วย overlay -----
        if self.overlay and self.mode in ("calib", "test"):
            sx, sy = self.targets_screen[self.idx]
            label = f"{self.idx+1}/9"
            self.overlay.update_point(sx, sy, label=label, r=32)

            # (ออปชัน) วาดจุดคาดคะเนสีขาวบน overlay
            if self.coef_x is not None and latest_gaze_direction is not None:
                yaw, pitch = _yaw_pitch_from_dir(latest_gaze_direction)
                px_s, py_s = _map_with_model(self.coef_x, self.coef_y, yaw, pitch)
                self.overlay.update_predicted(int(px_s), int(py_s), r=10)

# ---------------- Gaze vector & frame pipeline ----------------
def compute_gaze_vector(x, y, center_x, center_y, screen_width=640, screen_height=480):
    viewport_width = screen_width; viewport_height = screen_height
    fov_y_deg = 45.0; aspect_ratio = viewport_width / viewport_height
    far_clip = 100.0; camera_position = np.array([0.0, 0.0, 3.0])

    fov_y_rad = np.radians(fov_y_deg)
    half_height_far = np.tan(fov_y_rad / 2) * far_clip
    half_width_far = half_height_far * aspect_ratio

    ndc_x = (2.0 * x) / viewport_width - 1.0
    ndc_y = 1.0 - (2.0 * y) / viewport_height
    far_x = ndc_x * half_width_far
    far_y = ndc_y * half_height_far
    far_z = camera_position[2] - far_clip
    far_point = np.array([far_x, far_y, far_z])

    ray_origin = camera_position
    ray_direction = far_point - camera_position
    ray_direction /= np.linalg.norm(ray_direction)
    ray_direction = -ray_direction

    inner_radius = 1.0 / 1.05
    sphere_offset_x = (center_x / screen_width) * 2.0 - 1.0
    sphere_offset_y = 1.0 - (center_y / screen_height) * 2.0
    sphere_center = np.array([sphere_offset_x * 1.5, sphere_offset_y * 1.5, 0.0])

    origin = ray_origin; direction = -ray_direction
    L = origin - sphere_center
    a = np.dot(direction, direction)
    b = 2 * np.dot(direction, L)
    c = np.dot(L, L) - inner_radius ** 2
    disc = b ** 2 - 4 * a * c
    if disc < 0:
        t = -np.dot(direction, L) / np.dot(direction, direction)
    else:
        sqrt_disc = np.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        t = None
        if t1 > 0 and t2 > 0: t = min(t1, t2)
        elif t1 > 0: t = t1
        elif t2 > 0: t = t2
        if t is None:
            return None, None

    intersection_point = origin + t * direction
    intersection_local = intersection_point - sphere_center
    target_direction = intersection_local / np.linalg.norm(intersection_local)

    circle_local_center = np.array([0.0, 0.0, inner_radius])
    circle_local_center /= np.linalg.norm(circle_local_center)
    rotation_axis = np.cross(circle_local_center, target_direction)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    if rotation_axis_norm < 1e-6:
        gaze_rotated = circle_local_center
    else:
        rotation_axis /= rotation_axis_norm
        dot = np.dot(circle_local_center, target_direction)
        dot = np.clip(dot, -1.0, 1.0)
        angle_rad = np.arccos(dot)
        c = np.cos(angle_rad); s = np.sin(angle_rad); t_ = 1 - c
        x_, y_, z_ = rotation_axis
        rotation_matrix = np.array([
            [t_ * x_ * x_ + c,     t_ * x_ * y_ - s * z_, t_ * x_ * z_ + s * y_],
            [t_ * x_ * y_ + s * z_, t_ * y_ * y_ + c,     t_ * y_ * z_ - s * x_],
            [t_ * x_ * z_ - s * y_, t_ * y_ * z_ + s * x_, t_ * z_ * z_ + c]
        ])
        gaze_local = np.array([0.0, 0.0, inner_radius])
        gaze_rotated = rotation_matrix @ gaze_local
        gaze_rotated /= np.linalg.norm(gaze_rotated)

    # export for external usage
    try:
        with open("gaze_vector.txt", "w") as f:
            all_values = np.concatenate((sphere_center, gaze_rotated))
            csv_line = ",".join(f"{v:.6f}" for v in all_values)
            f.write(csv_line + "\n")
    except Exception:
        pass

    global latest_gaze_direction
    latest_gaze_direction = gaze_rotated
    return sphere_center, gaze_rotated

stored_intersections = []

def prune_intersections(intersections, maximum_intersections):
    if len(intersections) <= maximum_intersections:
        return intersections
    return intersections[-maximum_intersections:]

def find_line_intersection(ellipse1, ellipse2):
    (cx1, cy1), (_, minor_axis1), angle1 = ellipse1
    (cx2, cy2), (_, minor_axis2), angle2 = ellipse2
    a1 = np.deg2rad(angle1); a2 = np.deg2rad(angle2)
    dx1, dy1 = (minor_axis1 / 2) * np.cos(a1), (minor_axis1 / 2) * np.sin(a1)
    dx2, dy2 = (minor_axis2 / 2) * np.cos(a2), (minor_axis2 / 2) * np.sin(a2)
    A = np.array([[dx1, -dx2], [dy1, -dy2]])
    B = np.array([cx2 - cx1, cy2 - cy1])
    if np.linalg.det(A) == 0: return None
    t1, _ = np.linalg.solve(A, B)
    ix = cx1 + t1 * dx1
    iy = cy1 + t1 * dy1
    return (int(ix), int(iy))

def compute_average_intersection(frame, ray_lines, N, M, spacing):
    global stored_intersections
    if len(ray_lines) < 2 or N < 2:
        return (0, 0)
    h, w = frame.shape[:2]
    selected = random.sample(ray_lines, min(N, len(ray_lines)))
    intersections = []
    for i in range(len(selected) - 1):
        l1 = selected[i]; l2 = selected[i + 1]
        a1 = l1[2]; a2 = l2[2]
        if abs(a1 - a2) >= 2:
            inter = find_line_intersection(l1, l2)
            if inter and (0 <= inter[0] < w) and (0 <= inter[1] < h):
                intersections.append(inter)
                stored_intersections.append(inter)
    if len(stored_intersections) > M:
        stored_intersections = prune_intersections(stored_intersections, M)
    if not intersections: return None
    avg_x = np.mean([pt[0] for pt in stored_intersections])
    avg_y = np.mean([pt[1] for pt in stored_intersections])
    return (int(avg_x), int(avg_y))

def update_and_average_point(point_list, new_point, N):
    point_list.append(new_point)
    if len(point_list) > N: point_list.pop(0)
    if not point_list: return None
    avg_x = int(np.mean([p[0] for p in point_list]))
    avg_y = int(np.mean([p[1] for p in point_list]))
    return (avg_x, avg_y)

def process_frames(thr_strict, thr_medium, thr_relaxed, frame, gray, darkest_point,
                   debug_mode_on, render_cv_window):
    global ray_lines, max_rays, prev_model_center_avg, max_observed_distance
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thr_medium, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    reduced = filter_contours_by_area_and_return_largest(contours, 1000, 3)

    final_rotated_rect = None
    if len(reduced) > 0 and len(reduced[0]) > 5:
        ellipse = cv2.fitEllipse(reduced[0])
        final_rotated_rect = ellipse
        ray_lines.append(final_rotated_rect)
        if len(ray_lines) > max_rays:
            ray_lines = ray_lines[-max_rays:]

    model_center_average = (320, 240)
    model_center = compute_average_intersection(frame, ray_lines, 5, 1500, 5)
    if model_center is not None:
        model_center_average = update_and_average_point(model_centers, model_center, 200)
    if model_center_average[0] == 320:
        model_center_average = prev_model_center_avg
    if model_center_average[0] != 0:
        prev_model_center_avg = model_center_average

    if final_rotated_rect is None:
        cv2.imshow("Orlosky 3D EyeTracker", frame)
        return None

    (center_x, center_y) = tuple(map(int, final_rotated_rect[0]))
    if len(model_centers) >= 100 and center_x is not None:
        dist = math.hypot(center_x - model_center_average[0],
                          center_y - model_center_average[1])
        if dist > max_observed_distance:
            max_observed_distance = dist
    max_observed_distance = 202  # เพื่อตีกรอบ overlay ให้เห็นชัด

    cv2.circle(frame, model_center_average, int(max_observed_distance), (255, 50, 50), 2)
    cv2.circle(frame, model_center_average, 8, (255, 255, 0), -1)
    cv2.line(frame, model_center_average, (center_x, center_y), (255, 150, 50), 2)
    cv2.ellipse(frame, final_rotated_rect, (20, 255, 255), 2)

    dx = center_x - model_center_average[0]
    dy = center_y - model_center_average[1]
    extended_x = int(model_center_average[0] + 2 * dx)
    extended_y = int(model_center_average[1] + 2 * dy)
    cv2.line(frame, (center_x, center_y), (extended_x, extended_y), (200, 255, 0), 3)

    center, direction = compute_gaze_vector(center_x, center_y,
                                            model_center_average[0], model_center_average[1])
    cv2.imshow("Orlosky 3D EyeTracker", frame)
    return final_rotated_rect

def process_frame(frame):
    frame = crop_to_aspect_ratio(frame)
    darkest_point = get_darkest_area(frame)
    if darkest_point is None:
        h, w = frame.shape[:2]
        darkest_point = (w // 2, h // 2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dpv = int(gray[darkest_point[1], darkest_point[0]])
    thr_strict  = apply_binary_threshold(gray, dpv, 8)
    thr_medium  = apply_binary_threshold(gray, dpv, 12)
    thr_relaxed = apply_binary_threshold(gray, dpv, 18)
    thr_strict  = mask_outside_square(thr_strict,  darkest_point, 250)
    thr_medium  = mask_outside_square(thr_medium,  darkest_point, 250)
    thr_relaxed = mask_outside_square(thr_relaxed, darkest_point, 250)
    final_rotated_rect = process_frames(thr_strict, thr_medium, thr_relaxed,
                                        frame, gray, darkest_point, False, False)
    return final_rotated_rect, frame

# ---------------- Stream / Video Loops ----------------
def process_camera_esp32(cal):
    print(f"[INFO] Connecting to ESP32-CAM: {STREAM_URL}")
    flip_h = False; flip_v = False; paused = False
    last_ok = time.time()

    # กล้อง (อ้างอิง)
    cv2.namedWindow("Orlosky 3D EyeTracker (ESP32) + 9pt", cv2.WINDOW_NORMAL)

    # สร้าง overlay ทับจอจริง
    overlay = FullscreenOverlay(SCREEN_W, SCREEN_H)
    overlay.show()
    cal.set_overlay(overlay)

    while True:
        cap = cv2.VideoCapture(STREAM_URL)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print("[WARN] Open stream failed. Retry in 1.5s …")
            cv2.waitKey(1); time.sleep(1.5); continue
        print("[OK] Stream opened")
        # --- เริ่ม stream thread ---
        global stream_thread, stream_stop, latest_frame
        stream_stop.clear()
        if stream_thread is None or not stream_thread.is_alive():
            stream_thread = threading.Thread(target=esp32_stream_worker, args=(STREAM_URL, stream_stop), daemon=True)
            stream_thread.start()

        last_ok = time.time()
        while True:
            frame = latest_frame
            if not paused and frame is not None:
                last_ok = time.time()
                if flip_h: frame = cv2.flip(frame, 1)
                if flip_v: frame = cv2.flip(frame, 0)

                _, frame = process_frame(frame)
                cal.draw_overlay(frame)

                # live mouse
                if cal.mouse_on and HAVE_PYAUTO and cal.coef_x is not None and latest_gaze_direction is not None:
                    yaw, pitch = _yaw_pitch_from_dir(latest_gaze_direction)
                    px, py = _map_with_model(cal.coef_x, cal.coef_y, yaw, pitch)
                    try: pyautogui.moveTo(px, py, _pause=False)
                    except Exception: pass

                cv2.imshow("Orlosky 3D EyeTracker (ESP32) + 9pt", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                stream_stop.set()
                overlay.destroy()
                cv2.destroyAllWindows()
                return
            elif key == ord(' '):
                paused = not paused
            elif key == ord('r'):
                print("[INFO] Manual reconnect …")
                stream_stop.set()
                break
            elif key == ord('f'):
                flip_h = not flip_h; print(f"[TOGGLE] flip_h={flip_h}")
            elif key == ord('v'):
                flip_v = not flip_v; print(f"[TOGGLE] flip_v={flip_v}")
            elif key == ord('s') and not paused and frame is not None:
                ts = int(time.time()); cv2.imwrite(f"frame_{ts}.png", frame); print(f"[SAVE] frame_{ts}.png")
            elif key in (13, 10):  # Enter
                if   cal.mode == "calib": cal.record_current_calib(latest_gaze_direction)
                elif cal.mode == "test":  cal.record_current_test(latest_gaze_direction)
            elif key == ord('n'):
                if cal.mode in ("calib","test"): cal.idx = min(len(cal.targets_screen)-1, cal.idx+1)
            elif key == ord('b'):
                if cal.mode in ("calib","test"): cal.idx = max(0, cal.idx-1)
            elif key == ord('9'):
                cal.start()
            elif key == ord('m'):
                cal.toggle_mouse()
            elif key == ord('0'):
                cal.reset()

            # ถ้าไม่มี frame ใหม่เกิน 3 วินาที ให้แจ้งเตือน
            if time.time() - last_ok > 3.0:
                print("[TIMEOUT] No fresh frame from stream.")
                time.sleep(0.5)
                break

def process_video(cal):
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
    if not video_path: return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file."); return

    cv2.namedWindow("Orlosky 3D EyeTracker (Video) + 9pt", cv2.WINDOW_NORMAL)

    overlay = FullscreenOverlay(SCREEN_W, SCREEN_H)
    overlay.show()
    cal.set_overlay(overlay)

    while True:
        ret, frame = cap.read()
        if not ret: break
        _, frame = process_frame(frame)
        cal.draw_overlay(frame)
        cv2.imshow("Orlosky 3D EyeTracker (Video) + 9pt", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key in (13, 10):
            if   cal.mode == "calib": cal.record_current_calib(latest_gaze_direction)
            elif cal.mode == "test":  cal.record_current_test(latest_gaze_direction)
        elif key == ord('9'): cal.start()
        elif key == ord('n'):
            if cal.mode in ("calib","test"): cal.idx = min(len(cal.targets_screen)-1, cal.idx+1)
        elif key == ord('b'):
            if cal.mode in ("calib","test"): cal.idx = max(0, cal.idx-1)

    cap.release()
    overlay.destroy()
    cv2.destroyAllWindows()

# ---------------- Simple GUI ----------------
def selection_gui():
    root = tk.Tk()
    root.title("Select Input Source")
    tk.Label(root, text="Orlosky Eye Tracker 3D (ESP32 + 9pt Calib)", font=("Arial", 12, "bold")).pack(pady=10)
    tk.Label(root, text=f"ESP32-CAM URL: {STREAM_URL}").pack(pady=6)

    cal = NinePointOverlay()
    cal.load_if_exists()

    tk.Button(root, text="Start ESP32-CAM",
              command=lambda: [root.destroy(), process_camera_esp32(cal)]).pack(pady=8)
    tk.Button(root, text="Browse Video",
              command=lambda: [root.destroy(), process_video(cal)]).pack(pady=4)
    if GL_SPHERE_AVAILABLE:
        gl_sphere.start_gl_window()
    root.mainloop()

# ---------------- Main ----------------
if __name__ == "__main__":
    selection_gui()
