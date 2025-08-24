
import cv2
import random
import math
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, filedialog
import sys
import time
import json

# ===== ESP32‑CAM stream URL (แก้ IP ให้ตรงของเจ้านาย) =====
STREAM_URL = "http://10.63.100.193:81/stream"

# optional real mouse control
try:
    import pyautogui
    pyautogui.FAILSAFE = False
    SCREEN_W, SCREEN_H = pyautogui.size()
    HAVE_PYAUTO = True
except Exception:
    SCREEN_W, SCREEN_H = 1920, 1080
    HAVE_PYAUTO = False

# keep latest gaze direction for overlay / calibration
latest_gaze_direction = None

# gl_sphere (optional)
try:
    import gl_sphere
    GL_SPHERE_AVAILABLE = True
except ImportError:
    GL_SPHERE_AVAILABLE = False
    print("gl_sphere module not found. OpenGL rendering will be disabled.")

ray_lines = []
model_centers = []
max_rays = 100
prev_model_center_avg = (320, 240)
max_observed_distance = 0  # Initialize adaptive radius

# --------- helpers from the user's script (trimmed) ---------
def crop_to_aspect_ratio(image, width=640, height=480):
    current_height, current_width = image.shape[:2]
    desired_ratio = width / height
    current_ratio = current_width / current_height
    if current_ratio > desired_ratio:
        new_width = int(desired_ratio * current_height)
        offset = (current_width - new_width) // 2
        cropped_img = image[:, offset:offset + new_width]
    else:
        new_height = int(current_width / desired_ratio)
        offset = (current_height - new_height) // 2
        cropped_img = image[offset:offset + new_height, :]
    return cv2.resize(cropped_img, (width, height))

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
                    current_sum += gray[y + dy][x + dx]
                    num_pixels += 1
            if current_sum < min_sum and num_pixels > 0:
                min_sum = current_sum
                darkest_point = (x + searchArea // 2, y + searchArea // 2)
    return darkest_point

def mask_outside_square(image, center, size):
    x, y = center
    half_size = size // 2
    mask = np.zeros_like(image)
    top_left_x = max(0, x - half_size)
    top_left_y = max(0, y - half_size)
    bottom_right_x = min(image.shape[1], x + half_size)
    bottom_right_y = min(image.shape[0], y + half_size)
    mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 255
    return cv2.bitwise_and(image, mask)

def optimize_contours_by_angle(contours, image):
    if len(contours) < 1:
        return contours
    all_contours = np.concatenate(contours[0], axis=0)
    spacing = max(1, int(len(all_contours) / 25))
    filtered_points = []
    centroid = np.mean(all_contours, axis=0)
    for i in range(0, len(all_contours), 1):
        current_point = all_contours[i]
        prev_point = all_contours[i - spacing] if i - spacing >= 0 else all_contours[-spacing]
        next_point = all_contours[i + spacing] if i + spacing < len(all_contours) else all_contours[spacing]
        vec1 = prev_point - current_point
        vec2 = next_point - current_point
        with np.errstate(invalid='ignore'):
            _ = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        vec_to_centroid = centroid - current_point
        cos_threshold = np.cos(np.radians(60))
        if np.dot(vec_to_centroid, (vec1 + vec2) / 2) >= cos_threshold:
            filtered_points.append(current_point)
    return np.array(filtered_points, dtype=np.int32).reshape((-1, 1, 2))

def filter_contours_by_area_and_return_largest(contours, pixel_thresh, ratio_thresh):
    max_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= pixel_thresh:
            x, y, w, h = cv2.boundingRect(contour)
            length_to_width_ratio = max(w / h, h / w)
            if length_to_width_ratio <= ratio_thresh:
                if area > max_area:
                    max_area = area
                    largest_contour = contour
    return [largest_contour] if largest_contour is not None else []

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
    if len(contour) < 5:
        return 0
    ellipse = cv2.fitEllipse(contour)
    mask = np.zeros_like(binary_image)
    cv2.ellipse(mask, ellipse, (255), -1)
    ellipse_area = np.sum(mask == 255)
    covered_pixels = np.sum((binary_image == 255) & (mask == 255))
    if ellipse_area == 0:
        return ellipse_goodness
    ellipse_goodness[0] = covered_pixels / ellipse_area
    axes_lengths = ellipse[1]
    ellipse_goodness[2] = min(axes_lengths[1] / axes_lengths[0], axes_lengths[0] / axes_lengths[1])
    return ellipse_goodness

# --------- 9-point calibration utilities ---------
def _design_row(yaw, pitch):
    return np.array([1.0, yaw, pitch, yaw*pitch, yaw*yaw, pitch*pitch], dtype=np.float64)

def _fit_poly(samples):
    if len(samples) < 6:
        return None, None, None, None
    A = np.vstack([_design_row(y,p) for (y,p,_,_) in samples])
    bx = np.array([sx for (_,_,sx,_) in samples], dtype=np.float64)
    by = np.array([sy for (_,_,_,sy) in samples], dtype=np.float64)
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
    data = {
        "screen_w": SCREEN_W, "screen_h": SCREEN_H,
        "coef_x": [float(v) for v in coef_x],
        "coef_y": [float(v) for v in coef_y],
        "ts": time.time()
    }
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

class NinePointOverlay:
    def __init__(self):
        margin = 0.10
        xs = [margin, 0.5, 1.0 - margin]
        ys = [margin, 0.5, 1.0 - margin]
        self.targets = [(int(x*(SCREEN_W-1)), int(y*(SCREEN_H-1))) for y in ys for x in xs]
        self.idx = 0
        self.samples = []          # (yaw, pitch, sx, sy)
        self.test_records = []     # (tx,ty, px,py, err, yaw,pitch)
        self.coef_x = None
        self.coef_y = None
        self.rms_fit = (None, None)
        self.mode = "idle"         # idle | calib | test | live
        self.mouse_on = False
        self._last_yaw_pitch = None  # Fallback เมื่อเฟรมปัจจุบันไม่มี gaze

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
        # default safe
        return 0.0, 0.0, False

    def record_current_calib(self, gaze_dir):
        yaw, pitch, ok = self._pick_yaw_pitch(gaze_dir)
        sx, sy = self.targets[self.idx]
        if not ok:
            print("ยังอ่านค่า gaze เฟรมนี้ไม่ได้ → ใช้ค่าเดิมล่าสุด")
        self.samples.append((yaw, pitch, sx, sy))
        print(f"[CAL-9] เก็บจุด {self.idx+1}/9")
        if self.idx == len(self.targets)-1:
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
        tx, ty = self.targets[self.idx]
        px, py = _map_with_model(self.coef_x, self.coef_y, yaw, pitch)
        err = math.hypot(px - tx, py - ty)
        self.test_records.append((tx, ty, px, py, err, yaw, pitch))
        print(f"[TEST] จุด {self.idx+1}/9: error={err:.1f}px")
        if self.idx == len(self.targets)-1:
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
        self.__init__()
        print("[CAL-9] reset")

    def draw_overlay(self, frame):
        # อัพเดต last yaw/pitch ทุกเฟรมถ้ามี gaze
        global latest_gaze_direction
        if latest_gaze_direction is not None:
            self._last_yaw_pitch = _yaw_pitch_from_dir(latest_gaze_direction)

        # เป้าหมาย (จุดสีเขียว)
        if self.mode in ("calib", "test"):
            tx, ty = self.targets[self.idx]
            cv2.circle(frame, (tx,ty), 18, (0,255,0), -1)
            cv2.circle(frame, (tx,ty), 32, (60,200,60), 2)

        # จุดคาดคะเน (สีขาว)
        if self.coef_x is not None and latest_gaze_direction is not None:
            yaw, pitch = _yaw_pitch_from_dir(latest_gaze_direction)
            px, py = _map_with_model(self.coef_x, self.coef_y, yaw, pitch)
            cv2.circle(frame, (int(px),int(py)), 6, (240,240,240), -1)

        # ข้อความช่วย
        y0 = 30
        def put(txt):
            nonlocal y0
            cv2.putText(frame, txt, (12,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
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

# --------- gaze vector (store latest_gaze_direction) ---------
def compute_gaze_vector(x, y, center_x, center_y, screen_width=640, screen_height=480):
    viewport_width = screen_width
    viewport_height = screen_height
    fov_y_deg = 45.0
    aspect_ratio = viewport_width / viewport_height
    far_clip = 100.0
    camera_position = np.array([0.0, 0.0, 3.0])
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
    origin = ray_origin
    direction = -ray_direction
    L = origin - sphere_center
    a = np.dot(direction, direction)
    b = 2 * np.dot(direction, L)
    c = np.dot(L, L) - inner_radius ** 2
    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        t = -np.dot(direction, L) / np.dot(direction, direction)
    else:
        sqrt_disc = np.sqrt(discriminant)
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
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        t_ = 1 - c
        x_, y_, z_ = rotation_axis
        rotation_matrix = np.array([
            [t_ * x_ * x_ + c, t_ * x_ * y_ - s * z_, t_ * x_ * z_ + s * y_],
            [t_ * x_ * y_ + s * z_, t_ * y_ * y_ + c, t_ * y_ * z_ - s * x_],
            [t_ * x_ * z_ - s * y_, t_ * y_ * z_ + s * x_, t_ * z_ * z_ + c]
        ])
        gaze_local = np.array([0.0, 0.0, inner_radius])
        gaze_rotated = rotation_matrix @ gaze_local
        gaze_rotated /= np.linalg.norm(gaze_rotated)
    # write file for external usage
    file_path = "gaze_vector.txt"
    try:
        with open(file_path, "w") as f:
            all_values = np.concatenate((sphere_center, gaze_rotated))
            csv_line = ",".join(f"{v:.6f}" for v in all_values)
            f.write(csv_line + "\n")
    except Exception:
        pass
    # keep latest for overlay
    global latest_gaze_direction
    latest_gaze_direction = gaze_rotated
    return sphere_center, gaze_rotated

def check_ellipse_goodness(binary_image, contour, debug_mode_on):
    return [1,1,1]

def process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, frame, gray_frame,
                   darkest_point, debug_mode_on, render_cv_window):
    global ray_lines, max_rays, prev_model_center_avg, max_observed_distance
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_image = cv2.dilate(thresholded_image_medium, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    reduced_contours = filter_contours_by_area_and_return_largest(contours, 1000, 3)
    final_rotated_rect = None
    if len(reduced_contours) > 0 and len(reduced_contours[0]) > 5:
        ellipse = cv2.fitEllipse(reduced_contours[0])
        final_rotated_rect = ellipse
        ray_lines.append(final_rotated_rect)
        if len(ray_lines) > max_rays:
            num_to_remove = len(ray_lines) - max_rays
            ray_lines = ray_lines[num_to_remove:]
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
        distance = math.hypot(center_x - model_center_average[0], center_y - model_center_average[1])
        if distance > max_observed_distance:
            max_observed_distance = distance
    max_observed_distance = 202
    cv2.circle(frame, model_center_average, int(max_observed_distance), (255, 50, 50), 2)
    cv2.circle(frame, model_center_average, 8, (255, 255, 0), -1)
    cv2.line(frame, model_center_average, (center_x, center_y), (255, 150, 50), 2)
    cv2.ellipse(frame, final_rotated_rect, (20, 255, 255), 2)
    dx = center_x - model_center_average[0]
    dy = center_y - model_center_average[1]
    extended_x = int(model_center_average[0] + 2 * dx)
    extended_y = int(model_center_average[1] + 2 * dy)
    cv2.line(frame, (center_x, center_y), (extended_x, extended_y), (200, 255, 0), 3)
    center, direction = compute_gaze_vector(center_x, center_y, model_center_average[0], model_center_average[1])
    cv2.imshow("Orlosky 3D EyeTracker", frame)
    return final_rotated_rect

def update_and_average_point(point_list, new_point, N):
    point_list.append(new_point)
    if len(point_list) > N:
        point_list.pop(0)
    if not point_list:
        return None
    avg_x = int(np.mean([p[0] for p in point_list]))
    avg_y = int(np.mean([p[1] for p in point_list]))
    return (avg_x, avg_y)

stored_intersections = []

def compute_average_intersection(frame, ray_lines, N, M, spacing):
    global stored_intersections
    if len(ray_lines) < 2 or N < 2:
        return (0, 0)
    height, width = frame.shape[:2]
    selected_lines = random.sample(ray_lines, min(N, len(ray_lines)))
    intersections = []
    for i in range(len(selected_lines) - 1):
        line1 = selected_lines[i]; line2 = selected_lines[i + 1]
        angle1 = line1[2]; angle2 = line2[2]
        if abs(angle1 - angle2) >= 2:
            intersection = find_line_intersection(line1, line2)
            if intersection and (0 <= intersection[0] < width) and (0 <= intersection[1] < height):
                intersections.append(intersection)
                stored_intersections.append(intersection)
    if len(stored_intersections) > M:
        stored_intersections = prune_intersections(stored_intersections, M)
    if not intersections:
        return None
    avg_x = np.mean([pt[0] for pt in stored_intersections])
    avg_y = np.mean([pt[1] for pt in stored_intersections])
    return (int(avg_x), int(avg_y))

def prune_intersections(intersections, maximum_intersections):
    if len(intersections) <= maximum_intersections:
        return intersections
    return intersections[-maximum_intersections:]

def find_line_intersection(ellipse1, ellipse2):
    (cx1, cy1), (_, minor_axis1), angle1 = ellipse1
    (cx2, cy2), (_, minor_axis2), angle2 = ellipse2
    angle1_rad = np.deg2rad(angle1)
    angle2_rad = np.deg2rad(angle2)
    dx1, dy1 = (minor_axis1 / 2) * np.cos(angle1_rad), (minor_axis1 / 2) * np.sin(angle1_rad)
    dx2, dy2 = (minor_axis2 / 2) * np.cos(angle2_rad), (minor_axis2 / 2) * np.sin(angle2_rad)
    A = np.array([[dx1, -dx2], [dy1, -dy2]])
    B = np.array([cx2 - cx1, cy2 - cy1])
    if np.linalg.det(A) == 0:
        return None
    t1, t2 = np.linalg.solve(A, B)
    intersection_x = cx1 + t1 * dx1
    intersection_y = cy1 + t1 * dy1
    return (int(intersection_x), int(intersection_y))

# --------- frame wrappers ---------
def process_frame(frame):
    frame = crop_to_aspect_ratio(frame)
    darkest_point = get_darkest_area(frame)
    if darkest_point is None:
        h, w = frame.shape[:2]
        darkest_point = (w // 2, h // 2)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    darkest_pixel_value = int(gray_frame[darkest_point[1], darkest_point[0]])
    thresholded_image_strict  = apply_binary_threshold(gray_frame, darkest_pixel_value, 8)
    thresholded_image_medium  = apply_binary_threshold(gray_frame, darkest_pixel_value, 12)
    thresholded_image_relaxed = apply_binary_threshold(gray_frame, darkest_pixel_value, 18)
    thresholded_image_strict  = mask_outside_square(thresholded_image_strict,  darkest_point, 250)
    thresholded_image_medium  = mask_outside_square(thresholded_image_medium,  darkest_point, 250)
    thresholded_image_relaxed = mask_outside_square(thresholded_image_relaxed, darkest_point, 250)
    final_rotated_rect = process_frames(
        thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed,
        frame, gray_frame, darkest_point, False, False
    )
    return final_rotated_rect, frame

# --------- ESP32‑CAM streaming loop with calibration overlay ---------
def process_camera_esp32(cal):
    print(f"[INFO] Connecting to ESP32‑CAM: {STREAM_URL}")
    flip_h = False
    flip_v = False
    paused = False
    last_ok = time.time()
    while True:
        cap = cv2.VideoCapture(STREAM_URL)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print("[WARN] Open stream failed. Retry in 1.5s …")
            cv2.waitKey(1); time.sleep(1.5); continue
        print("[OK] Stream opened")
        for _ in range(4): cap.grab()
        while True:
            if not paused:
                for _ in range(2): cap.grab()
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[WARN] Read frame failed. Reconnect …")
                    break
                last_ok = time.time()
                if flip_h: frame = cv2.flip(frame, 1)
                if flip_v: frame = cv2.flip(frame, 0)
                _, frame = process_frame(frame)
                # overlay here:
                cal.draw_overlay(frame)

                # move mouse continuously in live mode
                if cal.mouse_on and HAVE_PYAUTO and cal.coef_x is not None and latest_gaze_direction is not None:
                    yaw, pitch = _yaw_pitch_from_dir(latest_gaze_direction)
                    px, py = _map_with_model(cal.coef_x, cal.coef_y, yaw, pitch)
                    try: pyautogui.moveTo(px, py, _pause=False)
                    except Exception: pass

                cv2.imshow("Orlosky 3D EyeTracker (ESP32) + 9pt", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release(); cv2.destroyAllWindows(); return
            elif key == ord(' '):
                paused = not paused
            elif key == ord('r'):
                print("[INFO] Manual reconnect …"); break
            elif key == ord('f'):
                flip_h = not flip_h; print(f"[TOGGLE] flip_h={flip_h}")
            elif key == ord('v'):
                flip_v = not flip_v; print(f"[TOGGLE] flip_v={flip_v}")
            elif key == ord('s') and not paused:
                ts = int(time.time()); cv2.imwrite(f"frame_{ts}.png", frame); print(f"[SAVE] frame_{ts}.png")
            elif key in (13, 10):  # Enter
                if cal.mode == "calib":
                    cal.record_current_calib(latest_gaze_direction)
                elif cal.mode == "test":
                    cal.record_current_test(latest_gaze_direction)
            elif key == ord('n'):
                # explicit skip to next target
                if cal.mode in ("calib","test"):
                    cal.idx = min(len(cal.targets)-1, cal.idx+1)
            elif key == ord('9'):
                cal.start()
            elif key == ord('b'):
                if cal.mode in ("calib","test"): cal.idx = max(0, cal.idx-1)
            elif key == ord('m'):
                cal.toggle_mouse()
            elif key == ord('0'):
                cal.reset()
            if time.time() - last_ok > 3.0:
                print("[TIMEOUT] No fresh frame. Reconnect …")
                break
        cap.release()
        cv2.destroyWindow("Orlosky 3D EyeTracker (ESP32) + 9pt")

# --------- Optional video file flow ---------
def process_video(cal):
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
    if not video_path: return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file."); return
    while True:
        ret, frame = cap.read()
        if not ret: break
        _, frame = process_frame(frame)
        cal.draw_overlay(frame)
        cv2.imshow("Orlosky 3D EyeTracker (Video) + 9pt", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key in (13, 10):
            if     cal.mode == "calib": cal.record_current_calib(latest_gaze_direction)
            elif   cal.mode == "test":  cal.record_current_test(latest_gaze_direction)
        elif key == ord('9'): cal.start()
        elif key == ord('n'):
            if cal.mode in ("calib","test"):
                cal.idx = min(len(cal.targets)-1, cal.idx+1)
    cap.release(); cv2.destroyAllWindows()

# --------- Simple GUI ---------
def selection_gui():
    root = tk.Tk()
    root.title("Select Input Source")
    tk.Label(root, text="Orlosky Eye Tracker 3D (ESP32 + 9pt Calib)", font=("Arial", 12, "bold")).pack(pady=10)
    tk.Label(root, text=f"ESP32‑CAM URL: {STREAM_URL}").pack(pady=6)
    cal = NinePointOverlay()
    cal.load_if_exists()
    tk.Button(root, text="Start ESP32‑CAM", command=lambda: [root.destroy(), process_camera_esp32(cal)]).pack(pady=8)
    tk.Button(root, text="Browse Video",   command=lambda: [root.destroy(), process_video(cal)]).pack(pady=4)
    if GL_SPHERE_AVAILABLE:
        gl_sphere.start_gl_window()
    root.mainloop()

# ===== Run =====
if __name__ == "__main__":
    selection_gui()
