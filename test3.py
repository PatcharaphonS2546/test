# test3.py
# 👀 Gaze All-in-One — Webcam / ESP32 / Calibration Overlay (+ Compensation)
# + Realtime metrics: FPS & Latency (UI + overlay)
# + Evaluation logging: RMSE / MAE (normalized & pixels) -> CSV + UI table
# + Screen override, Auto-extend calibration, Adjustable Gate thresholds

import math, time, threading, queue, json, pathlib, os
from dataclasses import dataclass
from typing import Optional, Tuple, List
from collections import deque
import numpy as np
import streamlit as st

# Optional deps guard
try:
    import cv2
except Exception:
    cv2 = None
try:
    import mediapipe as mp
except Exception:
    mp = None
try:
    import pyautogui
except Exception:
    pyautogui = None
try:
    import pandas as pd  # for pretty dataframe download
except Exception:
    pd = None

from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from av import VideoFrame

# ----------------------------- Utils -----------------------------
def _safe_size():
    if pyautogui:
        try:
            return pyautogui.size()
        except Exception:
            pass
    return 1920, 1080

def _now_iso():
    try:
        import datetime as _dt
        return _dt.datetime.now().isoformat(timespec="seconds")
    except Exception:
        return str(time.time())

# ----------------------------- Mouse Controller -----------------------------
class MouseController:
    def __init__(self, enable: bool = False, dwell_ms: int = 700, dwell_radius_px: int = 40):
        self.sw, self.sh = _safe_size()
        self.enabled = enable
        self.dwell_ms = dwell_ms
        self.dwell_radius_px = dwell_radius_px
        self._last_in_time = None
        self._last_target = None
        self.move_rate_limit_ms = 33
        self._last_move_at = 0.0

    def set_enable(self, v: bool):
        self.enabled = v
        if not v:
            self._last_in_time = None
            self._last_target = None

    def _click_here(self):
        if pyautogui is not None:
            try:
                pyautogui.click()
            except Exception:
                pass

    def update(self, x_norm: float, y_norm: float, do_click: bool = True):
        if not self.enabled:
            self._last_in_time = None
            return
        x_px = int(x_norm * self.sw)
        y_px = int(y_norm * self.sh)
        now = time.time()
        if (now - self._last_move_at) * 1000.0 >= self.move_rate_limit_ms:
            if pyautogui is not None:
                try:
                    pyautogui.moveTo(x_px, y_px)
                except Exception:
                    pass
            self._last_move_at = now
        tgt = (x_px, y_px)
        if (
            self._last_target is None
            or math.hypot(tgt[0] - self._last_target[0], tgt[1] - self._last_target[1])
            > self.dwell_radius_px
        ):
            self._last_target = tgt
            self._last_in_time = time.time()
            return
        if do_click and self._last_in_time is not None:
            if (time.time() - self._last_in_time) * 1000.0 >= self.dwell_ms:
                self._click_here()
                self._last_in_time = None

# ----------------------------- One Euro Filter -----------------------------
class OneEuroFilter:
    def __init__(self, freq=120.0, mincutoff=1.0, beta=0.01, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def alpha(self, cutoff):
        te = 1.0 / max(1e-6, self.freq)
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x, t=None):
        if self.t_prev is None:
            self.t_prev = time.time() if t is None else t
            self.x_prev = x
            return x
        tnow = time.time() if t is None else t
        dt = max(1e-6, tnow - self.t_prev)
        self.freq = 1.0 / dt
        self.t_prev = tnow
        dx = (x - self.x_prev) * self.freq
        a_d = self.alpha(self.dcutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self.alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

# ----------------------------- Config & Session -----------------------------
@dataclass
class GazeConfig:
    screen_w: int = _safe_size()[0]
    screen_h: int = _safe_size()[1]
    gain: float = 1.0
    gain_x: float = 1.0
    gain_y: float = 1.0
    gamma: float = 1.0
    deadzone: float = 0.02
    fallback_kx: float = 1.6
    fallback_ky: float = 1.4

@dataclass
class CalibrationReport:
    n_points: int
    rmse_px: Optional[float]
    rmse_cv_px: Optional[float]
    uniformity: float
    width: int
    height: int

    def passed(self) -> bool:
        # default strict rule (click_gate ใช้เกณฑ์จาก engine.sess.gate_* แทน)
        if self.n_points < 9:
            return False
        if self.uniformity < 0.55:
            return False
        if self.rmse_px is None or self.rmse_cv_px is None:
            return False
        return (self.rmse_px <= 30.0 and self.rmse_cv_px <= 35.0)

class SessionState:
    def __init__(self):
        self.smooth_x = OneEuroFilter(120, 1.0, 0.01, 1.0)
        self.smooth_y = OneEuroFilter(120, 1.0, 0.01, 1.0)
        self.calib_X = []
        self.calib_yx = []
        self.calib_yy = []
        self.calib_w = []
        self.model_ready = False
        self.model_pipeline = None
        self.last_feat: Optional[np.ndarray] = None
        self.last_quality: float = 0.5
        self.t_prev = None
        self.fps_hist = []
        self.drift_enabled = False
        self.drift_dx = 0.0
        self.drift_dy = 0.0
        self.drift_alpha = 0.15
        self.calib_report: Optional[CalibrationReport] = None
        # compensation
        self.comp_ax = 1.0
        self.comp_bx = 0.0
        self.comp_ay = 1.0
        self.comp_by = 0.0
        self.comp_valid = False
        self.comp_alpha = 1.0
        self.comp_enabled = True
        # realtime & eval logs
        self.last_metrics = {"fps": 0.0, "latency_ms": 0.0, "mode": "fallback"}
        self.eval_log: List[dict] = []
        self.eval_csv_path = "gaze_eval_log.csv"
        # gate thresholds (configurable)
        self.gate_rmse_train_max = 30.0
        self.gate_rmse_cv_max = 35.0
        self.gate_uniformity_min = 0.55
        self.gate_min_points = 9

try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge
except Exception:
    Pipeline = PolynomialFeatures = Ridge = None

# ----------------------------- Feature Vector -----------------------------
def _feat_vec(feat: dict) -> np.ndarray:
    ex = float(feat.get("eye_cx_norm", 0.5))
    ey = float(feat.get("eye_cy_norm", 0.5))
    fx = float(feat.get("face_cx_norm", 0.5))
    fy = float(feat.get("face_cy_norm", 0.5))
    # attenuate face influence (head sway)
    fx = 0.5 + (fx - 0.5) * 0.3
    fy = 0.5 + (fy - 0.5) * 0.3
    return np.array([ex, ey, fx, fy, 1.0], dtype=np.float32)

def _poly_expand(X: np.ndarray) -> np.ndarray:
    if X.ndim == 1:
        X = X[None, :]
    cols = [X]
    n = X.shape[1]
    for i in range(n):
        for j in range(i, n):
            cols.append((X[:, i] * X[:, j])[:, None])
    return np.concatenate(cols, axis=1)

def _build_penalty_vector(n_base: int = 5, eye_pen: float = 0.1, face_pen: float = 6.0, bias_pen: float = 0.5) -> np.ndarray:
    base_pen = [eye_pen, eye_pen, face_pen, face_pen, bias_pen]
    pens = list(base_pen)
    n = n_base
    for i in range(n):
        for j in range(i, n):
            pens.append(max(base_pen[i], base_pen[j]))
    return np.array(pens, dtype=np.float32)

def _fit_weighted_ridge(Phi, y, sample_w, pen_vec, lam):
    sw = np.sqrt(np.clip(sample_w.reshape(-1, 1), 0.0, 1e6))
    Phi_w = Phi * sw
    y_w = y.reshape(-1, 1) * sw
    A = Phi_w.T @ Phi_w + lam * np.diag(pen_vec.astype(np.float64))
    b = Phi_w.T @ y_w
    w = np.linalg.solve(A, b)
    return w.reshape(-1)

def _rmse_px(pred_xy, true_xy, W, H) -> float:
    dx = (pred_xy[:, 0] - true_xy[:, 0]) * W
    dy = (pred_xy[:, 1] - true_xy[:, 1]) * H
    return float(np.sqrt(np.mean(dx * dx + dy * dy)))

def _mae_px(pred_xy, true_xy, W, H) -> float:
    dx = np.abs((pred_xy[:, 0] - true_xy[:, 0]) * W)
    dy = np.abs((pred_xy[:, 1] - true_xy[:, 1]) * H)
    return float(np.mean(np.sqrt(dx * dx + dy * dy)))

def _uniformity_score(points_xy_norm: np.ndarray) -> float:
    pts = np.asarray(points_xy_norm, float)
    if len(pts) < 3:
        return 0.0
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    upper = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.vstack([lower[:-1], upper[:-1]])
    area = 0.0
    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % len(hull)]
        area += x1 * y2 - x2 * y1
    return float(max(0.0, min(1.0, abs(area) / 2.0)))

# ----------------------------- MediaPipe Extractor -----------------------------
class FeatureExtractor:
    LEFT_EYE_IDS = [33, 133, 159, 145]
    RIGHT_EYE_IDS = [362, 263, 386, 374]
    LEFT_IRIS_IDS = [468, 469, 470, 471, 472]
    RIGHT_IRIS_IDS = [473, 474, 475, 476, 477] if mp is not None else [468, 469, 470, 471, 472]

    def __init__(self, use_mediapipe: bool = True):
        self.use_mediapipe = use_mediapipe and (mp is not None) and (cv2 is not None)
        if self.use_mediapipe:
            self.mp_face = mp.solutions.face_mesh
            self.mesh = self.mp_face.FaceMesh(
                static_image_mode=False,
                refine_landmarks=True,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self.mesh = None
        self.earL_base = None
        self.earR_base = None
        self._ema_alpha = 0.05

    def close(self):
        if self.mesh is not None:
            self.mesh.close()

    @staticmethod
    def _avg_xy(ids, pts):
        xs = [pts[i][0] for i in ids if i < len(pts)]
        ys = [pts[i][1] for i in ids if i < len(pts)]
        if not xs or not ys:
            return None
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    @staticmethod
    def _eye_box_metrics(ids, pts):
        xs = [pts[i][0] for i in ids if i < len(pts)]
        ys = [pts[i][1] for i in ids if i < len(pts)]
        if not xs or not ys:
            return None
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        w = max(6.0, x1 - x0)
        h = max(2.0, y1 - y0)
        ear = h / max(6.0, w)
        return (x0, x1, y0, y1, w, h, ear)

    @staticmethod
    def _norm_in_box(p, box):
        if p is None or box is None:
            return (0.5, 0.5)
        x0, x1, y0, y1, w, h, _ = box
        nx = (p[0] - x0) / w
        ny = (p[1] - y0) / h
        return (min(1.0, max(0.0, nx)), min(1.0, max(0.0, ny)))

    def _extract_mediapipe(self, frame_bgr) -> Optional[dict]:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(frame_rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0]
        pts = [(lm.landmark[i].x * w, lm.landmark[i].y * h, lm.landmark[i].z) for i in range(len(lm.landmark))]
        l_iris = self._avg_xy(self.LEFT_IRIS_IDS, pts)
        r_iris = self._avg_xy(self.RIGHT_IRIS_IDS, pts)
        if l_iris is None and r_iris is None:
            return None
        l_box = self._eye_box_metrics(self.LEFT_EYE_IDS, pts)
        r_box = self._eye_box_metrics(self.RIGHT_EYE_IDS, pts)
        l_n = self._norm_in_box(l_iris, l_box) if l_iris is not None else (0.5, 0.5)
        r_n = self._norm_in_box(r_iris, r_box) if r_iris is not None else (0.5, 0.5)

        def _upd_base(cur, base):
            if cur is None:
                return base
            return cur if base is None else (base * (1.0 - self._ema_alpha) + cur * self._ema_alpha)

        earL = l_box[6] if l_box is not None else None
        earR = r_box[6] if r_box is not None else None
        if earL is not None and earL > 0.18:
            self.earL_base = _upd_base(earL, self.earL_base)
        if earR is not None and earR > 0.18:
            self.earR_base = _upd_base(earR, self.earR_base)

        def _eye_conf(ear, base, box, iris_ok):
            if (ear is None) or (box is None) or (not iris_ok):
                return 0.0
            ratio = (ear / 0.26) if base is None else (ear / max(1e-6, 0.8 * base))
            ratio = max(0.0, min(1.4, ratio))
            _, _, _, _, bw, bh, _ = box
            area_norm = (bw * bh) / (w * h)
            area_boost = min(1.0, area_norm / 0.02)
            return max(0.0, min(1.0, 0.7 * ratio + 0.3 * area_boost))

        cL = _eye_conf(earL, self.earL_base, l_box, l_iris is not None)
        cR = _eye_conf(earR, self.earR_base, r_box, r_iris is not None)
        eye_open = bool((earL or 0.0) > 0.20 or (earR or 0.0) > 0.20)
        wL, wR = cL, cR
        if (wL + wR) < 1e-3:
            eye_cx_norm = 0.5
            eye_cy_norm = 0.5
        else:
            eye_cx_norm = float((l_n[0] * wL + r_n[0] * wR) / (wL + wR))
            eye_cy_norm = float((l_n[1] * wL + r_n[1] * wR) / (wL + wR))
        face_ids = [1, 9, 152, 33, 263]
        fxs = [pts[i][0] for i in face_ids if i < len(pts)]
        fys = [pts[i][1] for i in face_ids if i < len(pts)]
        face_cx_norm = float(sum(fxs) / len(fxs) / max(1, w)) if fxs else 0.5
        face_cy_norm = float(sum(fys) / len(fys) / max(1, h)) if fys else 0.5
        box_margin = min(eye_cx_norm, 1 - eye_cx_norm, eye_cy_norm, 1 - eye_cy_norm)
        quality = max(cL, cR) * (0.6 + 0.4 * max(0.0, min(1.0, box_margin * 2)))
        return {
            "eye_cx_norm": eye_cx_norm,
            "eye_cy_norm": eye_cy_norm,
            "face_cx_norm": face_cx_norm,
            "face_cy_norm": face_cy_norm,
            "eye_open": eye_open,
            "quality": float(quality),
            "eye_conf_L": float(cL),
            "eye_conf_R": float(cR),
            "earL": float(earL or 0.0),
            "earR": float(earR or 0.0),
            "earL_base": float(self.earL_base or 0.0),
            "earR_base": float(self.earR_base or 0.0),
        }

    def extract(self, frame_bgr) -> Optional[dict]:
        if frame_bgr is None:
            return None
        if self.use_mediapipe and self.mesh is not None:
            try:
                return self._extract_mediapipe(frame_bgr)
            except Exception:
                return None
        return {
            "eye_cx_norm": 0.5,
            "eye_cy_norm": 0.5,
            "face_cx_norm": 0.5,
            "face_cy_norm": 0.5,
            "eye_open": True,
            "quality": 0.2,
            "eye_conf_L": 0.1,
            "eye_conf_R": 0.1,
            "earL": 0.2,
            "earR": 0.2,
            "earL_base": 0.2,
            "earR_base": 0.2,
        }

# ----------------------------- ESP32 Reader & Iris Detection -----------------------------
def _crop_to_aspect_ratio(image, width=640, height=480):
    h, w = image.shape[:2]
    desired = width / height
    cur = w / h
    if cur > desired:
        new_w = int(desired * h)
        off = (w - new_w) // 2
        image = image[:, off : off + new_w]
    else:
        new_h = int(w / desired)
        off = (h - new_h) // 2
        image = image[off : off + new_h, :]
    return cv2.resize(image, (width, height)) if cv2 is not None else image

# --- Kalman 2D (constant-velocity) for iris center -------------------------
class Kalman2D:
    def __init__(self, x0, y0, vx0=0.0, vy0=0.0):
        self.x = np.array([[x0], [y0], [vx0], [vy0]], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 100.0
        self.Q = np.diag([1e-2, 1e-2, 5e-1, 5e-1]).astype(np.float32)
        self.R_base = np.diag([3.0, 3.0]).astype(np.float32)
        self.last_t = time.time()

    def predict(self, dt=None):
        t = time.time()
        if dt is None:
            dt = max(1e-3, t - self.last_t)
        self.last_t = t
        F = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, zx, zy, meas_scale=1.0):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        z = np.array([[zx], [zy]], dtype=np.float32)
        R = self.R_base * (meas_scale**2)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4, dtype=np.float32) - K @ H) @ self.P

    def state(self):
        return float(self.x[0, 0]), float(self.x[1, 0])

# --- Debug holder for ESP32 iris detection ---
try:
    _IRIS_DBG
except NameError:
    _IRIS_DBG = {"t": 0, "det_hist": deque(maxlen=240)}

def _iris_center_norm_from_frame(frame_bgr):
    if cv2 is None or frame_bgr is None:
        return None
    frame = _crop_to_aspect_ratio(frame_bgr, 640, 480)
    h, w = frame.shape[:2]
    st = getattr(_iris_center_norm_from_frame, "_st", None)
    if st is None:
        st = {"c": (w // 2, h // 2), "roi": 180, "kf": None, "last_ts": time.time()}
        _iris_center_norm_from_frame._st = st

    cx_prev, cy_prev = st["c"]
    roi = int(max(80, min(320, st["roi"])))
    x0 = max(0, cx_prev - roi // 2)
    y0 = max(0, cy_prev - roi // 2)
    x1 = min(w, x0 + roi)
    y1 = min(h, y0 + roi)
    x0 = max(0, min(x0, w - (x1 - x0)))
    y0 = max(0, min(y0, h - (y1 - y0)))
    x1 = min(w, x0 + roi)
    y1 = min(h, y0 + roi)

    roi_img = frame[y0:y1, x0:x1]
    if roi_img.size == 0:
        roi_img = frame
        x0 = y0 = 0
        x1 = w
        y1 = h

    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    except Exception:
        pass
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = cv2.magnitude(gx, gy)

    hi = float(np.percentile(gray, 96))
    med = float(np.median(gray))
    glint_mask = gray > hi
    if np.any(glint_mask):
        gray = gray.copy()
        gray[glint_mask] = int(med)

    gX = float(np.mean(np.abs(gx)))
    gY = float(np.mean(np.abs(gy)))
    eyelid_heavy = (gY > 1.3 * max(1e-3, gX))
    p_base = 20
    off_base = 8
    if eyelid_heavy:
        p = max(15, p_base - 2)
        off = off_base + 2
    else:
        p = p_base
        off = off_base
    med_val = float(np.median(gray))
    p20 = float(np.percentile(gray, 20))
    if (med_val - p20) < 10:
        off = max(5, off - 2)
    thr_val = int(max(0, min(255, p + off)))
    _, th = cv2.threshold(gray, thr_val, 255, cv2.THRESH_BINARY_INV)
    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k3, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k5, iterations=1)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1.0
    best_axes = None
    max_area = (roi * roi * 0.55)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 300 or area > max_area:
            continue
        if len(c) >= 5:
            (cx_r, cy_r), (MA, ma), _ = cv2.fitEllipse(c)
            if ma < 1:
                continue
            ratio = max(MA, ma) / max(1e-6, min(MA, ma))
            if ratio > 3.5:
                continue
            per = cv2.arcLength(c, True)
            circ = 4 * math.pi * area / (per * per + 1e-6)
            gc_vals = []
            cx_i, cy_i = float(cx_r), float(cy_r)
            for i in range(0, len(c), 5):
                px, py = int(c[i, 0, 0]), int(c[i, 0, 1])
                if 1 <= px < gray.shape[1] - 1 and 1 <= py < gray.shape[0] - 1:
                    gxv = float(gx[py, px])
                    gyv = float(gy[py, px])
                    mag = float(gmag[py, px]) + 1e-6
                    rx = cx_i - px
                    ry = cy_i - py
                    rmag = math.hypot(rx, ry) + 1e-6
                    cos_sim = ((-gxv) * rx + (-gyv) * ry) / (mag * rmag)
                    gc_vals.append(max(0.0, min(1.0, (cos_sim + 1.0) / 2.0)))
            gcons = float(np.mean(gc_vals)) if gc_vals else 0.5
            if gcons < 0.35:
                continue
            score = area * circ * (0.5 + 0.5 * gcons)
            if score > best_score:
                best_score = score
                best = (int(cx_r), int(cy_r))
                best_axes = (MA, ma)
    method = "ellipse"
    used_darkest = False
    if best is None:
        try:
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=30, param1=60, param2=18, minRadius=8, maxRadius=int(roi * 0.45)
            )
            if circles is not None and len(circles[0]) > 0:
                c0 = circles[0][0]
                best = (int(c0[0]), int(c0[1]))
                best_axes = (2 * c0[2], 2 * c0[2])
                method = "hough"
        except Exception:
            pass
    if best is None:
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        step = 10
        win = 20
        best_sum = 1e18
        bx = by = None
        for yy in range(y0, y1 - win, step):
            for xx in range(x0, x1 - win, step):
                s = float(gray_full[yy : yy + win, xx : xx + win].sum())
                if s < best_sum:
                    best_sum = s
                    bx = xx + win // 2
                    by = yy + win // 2
        if bx is None:
            return None
        cx, cy = bx, by
        method = "darkest"
        used_darkest = True
    else:
        cx, cy = (x0 + best[0], y0 + best[1])

    now = time.time()
    dt = max(1e-3, now - st["last_ts"])
    st["last_ts"] = now
    if st["kf"] is None:
        st["kf"] = Kalman2D(cx, cy, 0.0, 0.0)

    if best_score > 0 and not used_darkest:
        conf = float(min(1.0, max(0.0, best_score / (roi * roi * 5.0))))
        if method == "hough":
            conf *= 0.9
    else:
        conf = 0.25

    st["kf"].predict(dt=dt)
    meas_scale = 2.5 if used_darkest else float(np.interp(conf, [0.25, 0.6, 0.9], [1.8, 1.0, 0.7]))
    st["kf"].update(cx, cy, meas_scale=meas_scale)
    cx_kf, cy_kf = st["kf"].state()
    cx_i, cy_i = int(cx_kf), int(cy_kf)
    st["c"] = (cx_i, cy_i)

    if conf < 0.45 or used_darkest:
        st["roi"] = min(320, int(st["roi"] * 1.12 + 6))
    elif conf > 0.65 and method in ("ellipse", "hough"):
        st["roi"] = max(120, int(st["roi"] * 0.92))

    _IRIS_DBG.setdefault("det_hist", deque(maxlen=240))
    _IRIS_DBG["det_hist"].append((now, 1 if ((not used_darkest) and (conf >= 0.5)) else 0))
    while _IRIS_DBG["det_hist"] and (now - _IRIS_DBG["det_hist"][0][0] > 2.0):
        _IRIS_DBG["det_hist"].popleft()
    if _IRIS_DBG["det_hist"]:
        total = len(_IRIS_DBG["det_hist"])
        good = sum(v for _, v in _IRIS_DBG["det_hist"])
        det_rate = float(good / max(1, total))
    else:
        det_rate = 0.0
    _IRIS_DBG.update(
        {
            "t": now,
            "frame_wh": (w, h),
            "roi_xyxy": (int(x0), int(y0), int(x1), int(y1)),
            "cxcy": (int(cx_i), int(cy_i)),
            "conf": float(conf),
            "method": method,
            "thr": int(thr_val),
            "det_rate": det_rate,
        }
    )
    nx = float(cx_i) / float(w)
    ny = float(cy_i) / float(h)
    nx = min(1.0, max(0.0, nx))
    ny = min(1.0, max(0.0, ny))
    return (nx, ny)

class ESP32Reader(threading.Thread):
    def __init__(self, url: str, out_q: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.url = url
        self.out_q = out_q
        self.stop_event = stop_event

    def run(self):
        if cv2 is None:
            return
        while not self.stop_event.is_set():
            cap = None
            try:
                cap = cv2.VideoCapture(self.url)
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                if not cap.isOpened():
                    time.sleep(1.0)
                    continue
                for _ in range(3):
                    cap.grab()
                last_ok = time.time()
                while not self.stop_event.is_set():
                    for _ in range(2):
                        cap.grab()
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    last_ok = time.time()
                    res = _iris_center_norm_from_frame(frame)
                    if res is not None:
                        nx, ny = res
                        try:
                            self.out_q.put((nx, ny), timeout=0.005)
                        except Exception:
                            pass
                    if (time.time() - last_ok) > 3.0:
                        break
            except Exception:
                time.sleep(1.0)
            finally:
                if cap is not None:
                    cap.release()

# ----------------------------- Model helpers -----------------------------
def _make_model():
    if Pipeline and PolynomialFeatures and Ridge:
        return (
            Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=True)), ("reg", Ridge(alpha=1.0))]),
            Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=True)), ("reg", Ridge(alpha=1.0))]),
        )
    else:
        return ("manual_poly_ridge", "manual_poly_ridge")

def _fit_model_dispatch(model_tuple, X, Y):
    if isinstance(model_tuple[0], str):
        Phi = _poly_expand(X)
        lam = 1e-3
        A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
        bx = Phi.T @ Y[:, 0]
        by = Phi.T @ Y[:, 1]
        wx = np.linalg.solve(A, bx)
        wy = np.linalg.solve(A, by)
        return (wx, wy)
    else:
        pipe_x, pipe_y = model_tuple
        pipe_x = pipe_x.fit(X, Y[:, 0])
        pipe_y = pipe_y.fit(X, Y[:, 1])
        return (pipe_x, pipe_y)

def _predict_dispatch(fitted_tuple, X):
    if isinstance(fitted_tuple[0], (np.ndarray, list)):
        wx, wy = fitted_tuple
        Phi = _poly_expand(X)
        x = Phi @ wx
        y = Phi @ wy
        return np.stack([x, y], axis=1).astype(np.float32)
    pipe_x, pipe_y = fitted_tuple
    x = pipe_x.predict(X)
    y = pipe_y.predict(X)
    return np.stack([x, y], axis=1).astype(np.float32)

# ----------------------------- Gaze Engine -----------------------------
class GazeEngine:
    def __init__(self, cfg: GazeConfig, extractor: FeatureExtractor):
        self.cfg = cfg
        self.ext = extractor
        self.sess = SessionState()

    def update_config(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)

    # ----- Gate thresholds -----
    def set_gate_thresholds(self, rmse_train_px_max: float, rmse_cv_px_max: float, uniformity_min: float, min_points: int = 9):
        self.sess.gate_rmse_train_max = float(rmse_train_px_max)
        self.sess.gate_rmse_cv_max = float(rmse_cv_px_max)
        self.sess.gate_uniformity_min = float(uniformity_min)
        self.sess.gate_min_points = int(min_points)

    # ----- Compensation -----
    def set_comp_control(self, enabled: bool, alpha: float):
        self.sess.comp_enabled = bool(enabled)
        self.sess.comp_alpha = float(max(0.0, min(1.0, alpha)))

    def _apply_comp(self, x: float, y: float) -> Tuple[float, float]:
        if not (self.sess.comp_enabled and self.sess.comp_valid):
            return x, y
        a = self.sess.comp_alpha
        cx = self.sess.comp_ax * x + self.sess.comp_bx
        cy = self.sess.comp_ay * y + self.sess.comp_by
        x = a * cx + (1.0 - a) * x
        y = a * cy + (1.0 - a) * y
        return min(1.0, max(0.0, x)), min(1.0, max(0.0, y))

    # ----- Stabilizer -----
    def _stabilize(self, x: float, y: float, quality: float, eye_open: bool) -> Tuple[float, float]:
        if not hasattr(self, "_stab_last"):
            self._stab_last = (0.5, 0.5)
            self._stab_hist = []
            self._stab_last_t = time.time()
        last_x, last_y = self._stab_last
        now = time.time()
        dt = max(1e-3, now - self._stab_last_t)
        if (not eye_open) or (quality < 0.35):  # ↑ เพิ่มคุณภาพขั้นต่ำให้รับค่าคาลิเบรตดีขึ้น
            return last_x, last_y
        self._stab_hist.append((x, y))
        if len(self._stab_hist) > 5:
            self._stab_hist = self._stab_hist[-5:]
        medx = float(np.median([p[0] for p in self._stab_hist]))
        medy = float(np.median([p[1] for p in self._stab_hist]))
        thr = 0.10 - 0.05 * quality
        if math.hypot(x - medx, y - medy) > thr:
            x, y = medx, medy
        max_step = (0.08 + 0.06 * (1.0 - quality)) * dt * 60.0
        dx, dy = x - last_x, y - last_y
        dist = math.hypot(dx, dy)
        if dist > max_step:
            sc = max_step / max(1e-6, dist)
            x = last_x + dx * sc
            y = last_y + dy * sc
        self._stab_last = (x, y)
        self._stab_last_t = now
        return x, y

    # ----- Calibration -----
    def calibration_add_point(self, sx, sy):
        if self.sess.last_feat is None:
            return False
        x = self.sess.last_feat.astype(np.float32)
        self.sess.calib_X.append(x.tolist())
        self.sess.calib_yx.append(float(sx))
        self.sess.calib_yy.append(float(sy))
        q = float(self.sess.last_quality or 0.5)
        w_q = max(0.0, min(1.0, (q - 0.20) / 0.80)) ** 2
        w_y = 1.0 + 0.6 * abs(sy - 0.5) * 2.0
        w_x = 1.0 + 0.6 * abs(sx - 0.5) * 2.0
        self.sess.calib_w.append(float((0.5 + 1.5 * w_q) * 0.5 * (w_x + w_y)))
        return True

    def _compute_report(self, X, Y) -> CalibrationReport:
        W, H = int(self.cfg.screen_w), int(self.cfg.screen_h)
        pred_train = self._map_from_feat_batch(X, apply_comp=False)
        rmse_train = _rmse_px(pred_train, Y, W, H)

        def _cv_rmse():
            n = len(X)
            if n < 5:
                return float("inf")
            if n >= 9:
                k = max(2, min(5, n // 3))
                idx = np.arange(n)
                np.random.shuffle(idx)
                folds = np.array_split(idx, k)
                acc = []
                for i in range(k):
                    te = folds[i]
                    tr = np.concatenate([folds[j] for j in range(k) if j != i])
                    model = _make_model()
                    model = _fit_model_dispatch(model, X[tr], Y[tr])
                    pred = _predict_dispatch(model, X[te])
                    acc.append(_rmse_px(pred, Y[te], W, H))
                return float(np.mean(acc))
            else:
                reps = 3
                acc = []
                for _ in range(reps):
                    idx = np.arange(n)
                    np.random.shuffle(idx)
                    cut = max(2, int(0.8 * n))
                    tr, te = idx[:cut], idx[cut:]
                    model = _make_model()
                    model = _fit_model_dispatch(model, X[tr], Y[tr])
                    pred = _predict_dispatch(model, X[te])
                    acc.append(_rmse_px(pred, Y[te], W, H))
                return float(np.mean(acc))

        rmse_cv = _cv_rmse()
        unif = _uniformity_score(Y)
        return CalibrationReport(n_points=len(X), rmse_px=rmse_train, rmse_cv_px=rmse_cv, uniformity=unif, width=W, height=H)

    def _fit_compensation(self, preds: np.ndarray, targets: np.ndarray):
        def fit_axis(p, t):
            p = np.asarray(p, float)
            t = np.asarray(t, float)
            var = np.var(p)
            if var < 1e-6:
                return 1.0, 0.0
            cov = np.cov(p, t, bias=True)[0, 1]
            a = cov / var
            a = float(np.clip(a, 0.80, 1.30))
            b = float(np.mean(t) - a * np.mean(p))
            return a, float(np.clip(b, -0.15, 0.15))

        ax, bx = fit_axis(preds[:, 0], targets[:, 0])
        ay, by = fit_axis(preds[:, 1], targets[:, 1])
        self.sess.comp_ax, self.sess.comp_bx = ax, bx
        self.sess.comp_ay, self.sess.comp_by = ay, by
        self.sess.comp_valid = True

    def _append_eval_log(self, row: dict):
        self.sess.eval_log.append(row)
        p = pathlib.Path(self.sess.eval_csv_path)
        header = list(row.keys())
        line = ",".join([str(row[k]) for k in header])
        try:
            if not p.exists():
                p.write_text(",".join(header) + "\n" + line + "\n")
            else:
                with p.open("a") as f:
                    f.write(line + "\n")
        except Exception:
            pass

    def calibration_finish(self):
        n = len(self.sess.calib_X)
        if n < 6:
            self.sess.model_ready = False
            self.sess.model_pipeline = None
            self.sess.calib_report = None
            self.sess.comp_valid = False
            return {"ok": False, "msg": "ต้องเก็บคาลิเบรทอย่างน้อย 6 จุด"}

        X = np.array(self.sess.calib_X, dtype=np.float32)
        Y = np.stack([np.array(self.sess.calib_yx, dtype=np.float32), np.array(self.sess.calib_yy, dtype=np.float32)], axis=1)
        Wsamples = np.array(self.sess.calib_w if self.sess.calib_w else [1.0] * n, dtype=np.float32)

        Phi = _poly_expand(X)
        pen_vec = _build_penalty_vector(n_base=X.shape[1], eye_pen=0.1, face_pen=8.0, bias_pen=0.5)
        lam = 1e-3
        wx = _fit_weighted_ridge(Phi, Y[:, 0], Wsamples, pen_vec, lam)
        wy = _fit_weighted_ridge(Phi, Y[:, 1], Wsamples, pen_vec, lam)
        self.sess.model_pipeline = (wx, wy)
        self.sess.model_ready = True

        preds = _predict_dispatch(self.sess.model_pipeline, X)
        self._fit_compensation(preds, Y)
        rep = self._compute_report(X, Y)
        self.sess.calib_report = rep

        # RMSE/MAE both normalized & px
        rmse_norm = float(np.sqrt(np.mean((preds - Y) ** 2)))
        mae_norm = float(np.mean(np.sqrt(np.sum((preds - Y) ** 2, axis=1))))
        W, H = int(self.cfg.screen_w), int(self.cfg.screen_h)
        rmse_px = _rmse_px(preds, Y, W, H)
        mae_px = _mae_px(preds, Y, W, H)

        # Add realtime metrics to log (BUGFIX: use sess.last_metrics; latency_ms key)
        lm = self.sess.last_metrics or {}
        row = {
            "ts": _now_iso(),
            "n_points": n,
            "rmse_px": round(rmse_px, 3),
            "rmse_cv_px": round(rep.rmse_cv_px, 3) if rep.rmse_cv_px is not None else "",
            "mae_px": round(mae_px, 3),
            "rmse_norm": round(rmse_norm, 6),
            "mae_norm": round(mae_norm, 6),
            "uniformity": round(rep.uniformity, 4),
            "passed": bool(rep.passed()),
            "mode": lm.get("model", ""),
            "fps": lm.get("fps", 0.0),
            "latency_ms": lm.get("latency_ms", 0.0),
            "screen_w": self.cfg.screen_w,
            "screen_h": self.cfg.screen_h,
        }
        self._append_eval_log(row)

        return {
            "ok": True,
            "rmse_norm": rmse_norm,
            "n": n,
            "rmse_px": rmse_px,
            "rmse_cv_px": rep.rmse_cv_px,
            "mae_norm": mae_norm,
            "mae_px": mae_px,
            "uniformity": rep.uniformity,
            "passed": rep.passed(),
            "comp": {"ax": self.sess.comp_ax, "bx": self.sess.comp_bx, "ay": self.sess.comp_ay, "by": self.sess.comp_by},
        }

    # ----- Save/Load -----
    def save_profile(self, path, meta=None):
        p = pathlib.Path(path)
        data = {
            "calib_X": self.sess.calib_X,
            "calib_yx": self.sess.calib_yx,
            "calib_yy": self.sess.calib_yy,
            "screen": {"w": self.cfg.screen_w, "h": self.cfg.screen_h},
            "comp": {
                "ax": self.sess.comp_ax,
                "bx": self.sess.comp_bx,
                "ay": self.sess.comp_ay,
                "by": self.sess.comp_by,
                "valid": self.sess.comp_valid,
            },
            "meta": meta or {},
        }
        try:
            p.write_text(json.dumps(data))
            return True
        except Exception:
            return False

    def load_profile(self, path):
        p = pathlib.Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text())
            self.sess.calib_X = data.get("calib_X", [])
            self.sess.calib_yx = data.get("calib_yx", [])
            self.sess.calib_yy = data.get("calib_yy", [])
            sc = data.get("screen", {})
            if sc:
                self.cfg.screen_w = sc.get("w", self.cfg.screen_w)
                self.cfg.screen_h = sc.get("h", self.cfg.screen_h)
            comp = data.get("comp", {})
            if comp:
                self.sess.comp_ax = float(comp.get("ax", 1.0))
                self.sess.comp_bx = float(comp.get("bx", 0.0))
                self.sess.comp_ay = float(comp.get("ay", 1.0))
                self.sess.comp_by = float(comp.get("by", 0.0))
                self.sess.comp_valid = bool(comp.get("valid", True))
            rep = self.calibration_finish()
            return bool(rep.get("ok", False))
        except Exception:
            return False

    # ----- Drift -----
    def drift_start(self):
        self.sess.drift_enabled = True

    def drift_stop(self):
        self.sess.drift_enabled = False

    def drift_add(self, sx, sy):
        if self.sess.last_feat is None:
            return False
        pred = self._map_from_feat(self.sess.last_feat, apply_comp=False)
        ex = sx - pred[0]
        ey = sy - pred[1]
        a = self.sess.drift_alpha
        self.sess.drift_dx = (1 - a) * self.sess.drift_dx + a * ex
        self.sess.drift_dy = (1 - a) * self.sess.drift_dy + a * ey
        return True

    def drift_status(self):
        return {"enabled": self.sess.drift_enabled, "dx": self.sess.drift_dx, "dy": self.sess.drift_dy}

    # ----- Mapping -----
    def _map_from_feat(self, feat_vec: np.ndarray, apply_comp: bool = True) -> np.ndarray:
        ex, ey = float(feat_vec[0]), float(feat_vec[1])
        eye_only = np.array(
            [0.5 + (ex - 0.5) * self.cfg.fallback_kx, 0.5 + (ey - 0.5) * self.cfg.fallback_ky], dtype=np.float32
        )
        if self.sess.model_ready and self.sess.model_pipeline is not None:
            pm = _predict_dispatch(self.sess.model_pipeline, feat_vec[None, :])[0]
            alpha_x = 0.10
            alpha_y = 0.20
            px = (1.0 - alpha_x) * pm[0] + alpha_x * eye_only[0]
            py = (1.0 - alpha_y) * pm[1] + alpha_y * eye_only[1]
        else:
            px, py = eye_only[0], eye_only[1]
        if apply_comp:
            px, py = self._apply_comp(px, py)
        return np.array([px, py], dtype=np.float32)

    def _map_from_feat_batch(self, X, apply_comp: bool = True):
        if X.ndim == 1:
            X = X[None, :]
        if self.sess.model_ready and self.sess.model_pipeline is not None:
            pred = _predict_dispatch(self.sess.model_pipeline, X)
        else:
            ex, ey = X[:, 0], X[:, 1]
            pred = np.stack(
                [0.5 + (ex - 0.5) * self.cfg.fallback_kx, 0.5 + (ey - 0.5) * self.cfg.fallback_ky], axis=1
            ).astype(np.float32)
        if apply_comp:
            a = self.sess.comp_alpha
            if self.sess.comp_enabled and self.sess.comp_valid:
                pred[:, 0] = a * (self.sess.comp_ax * pred[:, 0] + self.sess.comp_bx) + (1.0 - a) * pred[:, 0]
                pred[:, 1] = a * (self.sess.comp_ay * pred[:, 1] + self.sess.comp_by) + (1.0 - a) * pred[:, 1]
        return np.clip(pred, 0.0, 1.0)

    def _shape_and_smooth(self, x: float, y: float) -> Tuple[float, float]:
        if self.sess.drift_enabled:
            x += self.sess.drift_dx
            y += self.sess.drift_dy
        gx = self.cfg.gain * self.cfg.gain_x
        gy = self.cfg.gain * self.cfg.gain_y
        x = 0.5 + (x - 0.5) * gx
        y = 0.5 + (y - 0.5) * gy
        if self.cfg.gamma != 1.0:
            def _gamma(v):
                s = (v - 0.5)
                sign = 1.0 if s >= 0 else -1.0
                return 0.5 + sign * (abs(s) ** self.cfg.gamma)
            x = _gamma(x)
            y = _gamma(y)
        dz = self.cfg.deadzone
        def _dead(v):
            d = v - 0.5
            return 0.5 if abs(d) < dz else v
        x = _dead(x)
        y = _dead(y)
        x = min(1.0, max(0.0, x))
        y = min(1.0, max(0.0, y))
        t = time.time()
        x = float(self.sess.smooth_x.filter(x, t))
        y = float(self.sess.smooth_y.filter(y, t))
        return x, y

    # ----- Frames -----
    def process_frame(self, frame_bgr):
        t0 = time.time()
        feat = self.ext.extract(frame_bgr)
        if feat is None:
            pred = np.array([0.5, 0.5], dtype=np.float32)
            q = 0.2
            eye_open = True
        else:
            fv = _feat_vec(feat) if self.sess.last_feat is None or feat.get("eye_open", True) else self.sess.last_feat
            self.sess.last_feat = fv
            pred = self._map_from_feat(fv, apply_comp=True)
            q = float(feat.get("quality", 0.5))
            eye_open = bool(feat.get("eye_open", True))
            self.sess.last_quality = q

        for f in (self.sess.smooth_x, self.sess.smooth_y):
            f.mincutoff = max(0.3, 1.6 - 1.2 * q)
            f.beta = 0.01 + 0.12 * (1.0 - q)

        px, py = float(pred[0]), float(pred[1])
        px, py = self._stabilize(px, py, q, eye_open)
        x, y = self._shape_and_smooth(px, py)

        t1 = time.time()
        if self.sess.t_prev is not None:
            dt = t1 - self.sess.t_prev
            if dt > 0:
                self.sess.fps_hist.append(1.0 / dt)
                if len(self.sess.fps_hist) > 90:
                    self.sess.fps_hist = self.sess.fps_hist[-90:]
        self.sess.t_prev = t1
        fps = float(np.mean(self.sess.fps_hist)) if self.sess.fps_hist else 0.0
        latency_ms = float((t1 - t0) * 1000.0)
        metrics = {"fps": fps, "latency_ms": latency_ms, "model": "calibrated" if self.sess.model_ready else "fallback"}
        self.sess.last_metrics = metrics
        return x, y, metrics, frame_bgr

    def process_external(self, nx, ny):
        # ESP32 path mirrors timing to keep FPS/latency updated
        t0 = time.time()
        fv = np.array([nx, ny, 0.5, 0.5, 1.0], dtype=np.float32)
        self.sess.last_feat = fv
        pred = self._map_from_feat(fv, apply_comp=True)
        px, py = float(pred[0]), float(pred[1])
        px, py = self._stabilize(px, py, 0.5, True)
        x, y = self._shape_and_smooth(px, py)
        self.sess.last_quality = 0.5
        t1 = time.time()
        if self.sess.t_prev is not None:
            dt = t1 - self.sess.t_prev
            if dt > 0:
                self.sess.fps_hist.append(1.0 / dt)
                if len(self.sess.fps_hist) > 90:
                    self.sess.fps_hist = self.sess.fps_hist[-90:]
        self.sess.t_prev = t1
        fps = float(np.mean(self.sess.fps_hist)) if self.sess.fps_hist else 0.0
        latency_ms = float((t1 - t0) * 1000.0)
        self.sess.last_metrics = {
            "fps": fps,
            "latency_ms": latency_ms,
            "model": "calibrated" if self.sess.model_ready else "fallback",
        }
        return x, y

    def get_report(self):
        rmse = None
        if self.sess.model_ready and len(self.sess.calib_X) >= 6:
            X = np.array(self.sess.calib_X, dtype=np.float32)
            Y = np.stack(
                [np.array(self.sess.calib_yx, dtype=np.float32), np.array(self.sess.calib_yy, dtype=np.float32)], axis=1
            )
            preds = self._map_from_feat_batch(X, apply_comp=False)
            rmse = float(np.sqrt(np.mean((preds - Y) ** 2)))
        return {"model_ready": self.sess.model_ready, "rmse_norm": rmse, "n_samples": len(self.sess.calib_X)}

    def get_calibration_report(self) -> Optional[CalibrationReport]:
        return self.sess.calib_report

    def get_comp_params(self):
        return {
            "enabled": self.sess.comp_enabled,
            "alpha": self.sess.comp_alpha,
            "ax": self.sess.comp_ax,
            "bx": self.sess.comp_bx,
            "ay": self.sess.comp_ay,
            "by": self.sess.comp_by,
            "valid": self.sess.comp_valid,
        }

    def get_last_metrics(self):
        return dict(self.sess.last_metrics)

    def get_eval_log(self) -> List[dict]:
        return list(self.sess.eval_log)

    def click_gate(self) -> Tuple[bool, str]:
        rep = self.sess.calib_report
        if rep is None:
            return False, "ยังไม่คาลิเบรท"
        reasons = []
        if rep.n_points < self.sess.gate_min_points:
            reasons.append(f"จุดน้อย ({rep.n_points}/{self.sess.gate_min_points})")
        if rep.uniformity < self.sess.gate_uniformity_min:
            reasons.append(f"กระจายตัวต่ำ ({rep.uniformity:.2f} < {self.sess.gate_uniformity_min:.2f})")
        if rep.rmse_px is None or rep.rmse_px > self.sess.gate_rmse_train_max:
            reasons.append(
                f"RMSE เทรน {rep.rmse_px:.0f}px > {self.sess.gate_rmse_train_max:.0f}px"
                if rep.rmse_px is not None
                else "RMSE เทรนไม่พร้อม"
            )
        if rep.rmse_cv_px is None or rep.rmse_cv_px > self.sess.gate_rmse_cv_max:
            reasons.append(
                f"RMSE CV {rep.rmse_cv_px:.0f}px > {self.sess.gate_rmse_cv_max:.0f}px"
                if rep.rmse_cv_px is not None
                else "RMSE CV ไม่พร้อม"
            )
        passed = len(reasons) == 0
        return passed, ("ผ่าน" if passed else " , ".join(reasons))

# ----------------------------- App State -----------------------------
class AppState:
    def __init__(self):
        self.extractor_mode = "Webcam/MediaPipe"
        self.mouse_enabled = False
        self.dwell_ms = 1200           # ↑ default ยาวขึ้นเพื่อเก็บคุณภาพดี
        self.dwell_radius = 40
        self.esp32_url = "http://10.63.100.193:81/stream"
        self.esp32_q = None
        self.esp32_stop = None
        self.synthetic_hud = False
        self.esp32_debug = False
        self.mirror_input = False
        self.invert_x = False
        self.invert_y = False
        self.gain = 1.2
        self.gamma = 1.0
        self.deadzone = 0.02
        self.filter_mincutoff = 1.0
        self.filter_beta = 0.01
        self.shared_gaze_x = 0.5
        self.shared_gaze_y = 0.5
        self.calib_overlay_active = False
        self.calib_targets = []
        self.calib_idx = 0
        self.calib_dwell_ms = 1200     # ↑ default ยาวขึ้น
        self.calib_radius_norm = 0.02
        self.mouse_gate_passed = False
        self.mouse_gate_reason = "ยังไม่คาลิเบรท"
        self.comp_enabled = True
        self.comp_alpha = 1.0
        # realtime UI metrics
        self.ui_fps = 0.0
        self.ui_latency_ms = 0.0
        # gate mode
        self.gate_mode = "Balanced"
        self.gate_rmse_train_max = 45.0
        self.gate_rmse_cv_max = 55.0
        self.gate_uniformity_min = 0.50
        self.gate_min_points = 9
        # screen override
        sw, sh = _safe_size()
        self.use_screen_override = False
        self.screen_w_override = sw
        self.screen_h_override = sh

if "APP_STATE" not in st.session_state:
    st.session_state["APP_STATE"] = AppState()
APP_STATE: AppState = st.session_state["APP_STATE"]

if "calib_ph" not in st.session_state:
    st.session_state["calib_ph"] = st.empty()

# ----------------------------- WebRTC Processor -----------------------------
class GazeProcessor(VideoProcessorBase):
    def __init__(self):
        self.engine = None
        self.mouse = None
        self.last_gaze = (0.5, 0.5)
        self.calib_active = False
        self.calib_targets = []
        self.calib_idx = 0
        self.calib_hold_start = None
        self.calib_radius_norm = 0.02
        self._calib_banner_until = 0.0
        self._calib_banner_text = ""
        self._feat_pool = []
        self._last_fix = (0.5, 0.5, time.time())
        self._auto_extended_once = False

    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if APP_STATE.extractor_mode == "Webcam/MediaPipe" and APP_STATE.mirror_input and cv2 is not None:
            img = cv2.flip(img, 1)
        if APP_STATE.extractor_mode == "ESP32 (Orlosky)" and getattr(APP_STATE, "synthetic_hud", False):
            img = np.full((480, 640, 3), 40, dtype=np.uint8)

        if self.engine is None:
            extractor = FeatureExtractor(use_mediapipe=(APP_STATE.extractor_mode == "Webcam/MediaPipe"))
            cfg = GazeConfig()
            self.engine = GazeEngine(cfg, extractor)
            self.mouse = MouseController(enable=False, dwell_ms=APP_STATE.dwell_ms, dwell_radius_px=APP_STATE.dwell_radius)

        # Apply shaping/filter + compensation controls
        try:
            # screen override
            if APP_STATE.use_screen_override:
                self.engine.update_config(screen_w=int(APP_STATE.screen_w_override), screen_h=int(APP_STATE.screen_h_override))
            # general
            self.engine.update_config(gain=float(APP_STATE.gain), gamma=float(APP_STATE.gamma), deadzone=float(APP_STATE.deadzone))
            for f in (self.engine.sess.smooth_x, self.engine.sess.smooth_y):
                f.mincutoff = float(APP_STATE.filter_mincutoff)
                f.beta = float(APP_STATE.filter_beta)
            self.engine.set_comp_control(APP_STATE.comp_enabled, APP_STATE.comp_alpha)
            # gate thresholds
            self.engine.set_gate_thresholds(
                APP_STATE.gate_rmse_train_max,
                APP_STATE.gate_rmse_cv_max,
                APP_STATE.gate_uniformity_min,
                APP_STATE.gate_min_points,
            )
        except Exception:
            pass

        # Assist during calibration
        if APP_STATE.calib_overlay_active and self.engine is not None:
            if not hasattr(self, "_saved_calib_assist"):
                self._saved_calib_assist = (self.engine.cfg.fallback_kx, self.engine.cfg.fallback_ky, self.engine.cfg.deadzone)
            self.engine.cfg.fallback_kx = max(self.engine.cfg.fallback_kx, 2.0)
            self.engine.cfg.fallback_ky = max(self.engine.cfg.fallback_ky, 1.8)
            self.engine.cfg.deadzone = 0.0
        else:
            if hasattr(self, "_saved_calib_assist") and self.engine is not None:
                fbk, fby, dz = self._saved_calib_assist
                self.engine.cfg.fallback_kx, self.engine.cfg.fallback_ky = fbk, fby
                self.engine.cfg.deadzone = dz
                delattr(self, "_saved_calib_assist")

        # per-frame processing
        if APP_STATE.extractor_mode == "Webcam/MediaPipe":
            x, y, metrics, img = self.engine.process_frame(img)
        else:
            x, y = self.last_gaze
            if APP_STATE.esp32_q is not None:
                try:
                    nx, ny = APP_STATE.esp32_q.get_nowait()
                    x, y = self.engine.process_external(nx, ny)
                except Exception:
                    pass
            metrics = self.engine.get_last_metrics()

        if APP_STATE.invert_x:
            x = 1.0 - x
        if APP_STATE.invert_y:
            y = 1.0 - y
        self.last_gaze = (x, y)
        APP_STATE.shared_gaze_x = float(x)
        APP_STATE.shared_gaze_y = float(y)
        # update UI metrics state
        APP_STATE.ui_fps = float(metrics.get("fps", 0.0))
        APP_STATE.ui_latency_ms = float(metrics.get("latency_ms", 0.0))

        # ------------------ Calibration flow ------------------
        if APP_STATE.calib_overlay_active:
            if not self.calib_active:
                self.calib_active = True
                self.calib_targets = list(APP_STATE.calib_targets)
                self.calib_idx = APP_STATE.calib_idx
                self.calib_hold_start = None
                self.calib_radius_norm = float(APP_STATE.calib_radius_norm)

            tx, ty = (
                self.calib_targets[self.calib_idx] if (0 <= self.calib_idx < len(self.calib_targets)) else (0.5, 0.5)
            )
            q = float(self.engine.sess.last_quality or 0.0)
            lx, ly, lt = self._last_fix
            dtv = max(1e-3, time.time() - lt)
            speed = math.hypot(x - lx, y - ly) / dtv
            self._last_fix = (x, y, time.time())
            stable = (q >= 0.35) and (speed < 1.2)  # ↑ เข้มขึ้นเล็กน้อย

            now = time.time()
            if stable and (self.engine.sess.last_feat is not None):
                if self.calib_hold_start is None:
                    self.calib_hold_start = now
                    self._feat_pool = []
                self._feat_pool.append(self.engine.sess.last_feat.copy())
                elapsed_ms = (now - self.calib_hold_start) * 1000.0
            else:
                self.calib_hold_start = None
                self._feat_pool = []
                elapsed_ms = 0.0

            frac = max(0.0, min(1.0, elapsed_ms / float(max(1, APP_STATE.calib_dwell_ms))))
            if cv2 is not None:
                h, w = img.shape[:2]
                gx_i, gy_i = int(tx * w), int(ty * h)
                cv2.circle(img, (gx_i, gy_i), 18, (0, 170, 255), 2)
                end_angle = int(360 * frac)
                cv2.ellipse(img, (gx_i, gy_i), (22, 22), 0, 0, end_angle, (0, 170, 255), 3)
                cv2.putText(
                    img,
                    f"Calibration {min(self.calib_idx + 1, len(self.calib_targets))}/{len(self.calib_targets)}",
                    (24, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    img,
                    f"Q={q:.2f} speed={speed:.2f}",
                    (24, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 255) if stable else (80, 80, 80),
                    2,
                )

            if frac >= 1.0 and 0 <= self.calib_idx < len(self.calib_targets):
                ok_added = False
                if len(self._feat_pool) >= 5:
                    feat_avg = np.mean(np.stack(self._feat_pool, axis=0), axis=0).astype(np.float32)
                    self.engine.sess.last_feat = feat_avg
                    ok_added = self.engine.calibration_add_point(tx, ty)
                elif self.engine.sess.last_feat is not None:
                    ok_added = self.engine.calibration_add_point(tx, ty)

                if ok_added:
                    self.calib_idx += 1
                    APP_STATE.calib_idx = self.calib_idx
                    self.calib_hold_start = None
                    self._feat_pool = []

                if self.calib_idx >= len(self.calib_targets):
                    rep = self.engine.calibration_finish()
                    # Auto-extend step: ถ้ายัง uniformity ต่ำ ให้เพิ่ม 4 มุมแล้วเก็บต่อ (ครั้งเดียว)
                    need_extend = (
                        rep.get("ok")
                        and (rep.get("uniformity", 0.0) < self.engine.sess.gate_uniformity_min)
                        and (not self._auto_extended_once)
                    )
                    if need_extend:
                        extra = [(0.10, 0.10), (0.90, 0.10), (0.10, 0.90), (0.90, 0.90)]
                        APP_STATE.calib_targets = extra
                        APP_STATE.calib_idx = 0
                        APP_STATE.calib_overlay_active = True
                        self._auto_extended_once = True
                        self.calib_active = False
                        self._feat_pool = []
                        self._calib_banner_text = "Uniformity ต่ำ — เพิ่มอีก 4 มุมอัตโนมัติ"
                        self._calib_banner_until = time.time() + 2.0
                    else:
                        APP_STATE.calib_overlay_active = False
                        self.calib_active = False
                        self._feat_pool = []
                        if rep.get("ok"):
                            tag = "PASS" if rep.get("passed") else "LOCK"
                            self._calib_banner_text = (
                                f"Calibration OK · RMSE={rep.get('rmse_px',0):.0f}px / "
                                f"CV={rep.get('rmse_cv_px',0):.0f}px · MAE={rep.get('mae_px',0):.0f}px · "
                                f"U={rep.get('uniformity',0):.2f} · {tag}"
                            )
                        else:
                            self._calib_banner_text = f"Calibration NG: {rep.get('msg','')}"
                        self._calib_banner_until = time.time() + 2.0
        else:
            self.calib_active = False
            self.calib_hold_start = None

        if time.time() < self._calib_banner_until and cv2 is not None and self._calib_banner_text:
            cv2.rectangle(img, (10, 10), (10 + 1100, 10 + 40), (0, 0, 0), -1)
            cv2.putText(img, self._calib_banner_text, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 255), 2)

        # Click Gate
        gate_ok, reason = (False, "ยังไม่คาลิเบรท")
        if self.engine is not None:
            gate_ok, reason = self.engine.click_gate()
        APP_STATE.mouse_gate_passed = bool(gate_ok)
        APP_STATE.mouse_gate_reason = reason

        # Mouse
        if self.mouse:
            self.mouse.set_enable(APP_STATE.mouse_enabled)
            self.mouse.dwell_ms = APP_STATE.dwell_ms
            self.mouse.dwell_radius_px = APP_STATE.dwell_radius
            self.mouse.update(x, y, do_click=(APP_STATE.mouse_enabled and gate_ok))

        # Crosshair + Realtime overlay FPS/Latency
        if cv2 is not None:
            h, w = img.shape[:2]
            gx, gy = int(x * w), int(y * h)
            cv2.circle(img, (gx, gy), 8, (0, 255, 0), 2)
            cv2.line(img, (gx - 15, gy), (gx + 15, gy), (0, 255, 0), 1)
            cv2.line(img, (gx, gy - 15), (gx, gy + 15), (0, 255, 0), 1)
            info = f"FPS {APP_STATE.ui_fps:4.1f} | Latency {APP_STATE.ui_latency_ms:4.1f} ms"
            cv2.putText(img, info, (w - 460, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return VideoFrame.from_ndarray(img, format="bgr24")

# ----------------------------- UI: Sidebar -----------------------------
def _sidebar_controls():
    st.sidebar.title("⚙️ Controls")
    mode = st.sidebar.selectbox("Extractor Mode", ["Webcam/MediaPipe", "ESP32 (Orlosky)"], index=0)
    APP_STATE.extractor_mode = mode

    st.sidebar.subheader("🖥️ Display / Screen")
    APP_STATE.use_screen_override = st.sidebar.checkbox("Override screen resolution (px)", value=APP_STATE.use_screen_override)
    sw, sh = _safe_size()
    colsw, colsh = st.sidebar.columns(2)
    APP_STATE.screen_w_override = int(colsw.number_input("Width", min_value=320, max_value=10000, value=int(APP_STATE.screen_w_override)))
    APP_STATE.screen_h_override = int(colsh.number_input("Height", min_value=240, max_value=10000, value=int(APP_STATE.screen_h_override)))
    st.sidebar.caption(f"OS reports: {sw}×{sh}px")

    st.sidebar.subheader("🎛 Shaping")
    APP_STATE.gain = float(st.sidebar.slider("Gain", 0.5, 2.5, APP_STATE.gain, 0.05))
    APP_STATE.gamma = float(st.sidebar.slider("Gamma", 0.5, 2.0, 1.0, 0.05))
    APP_STATE.deadzone = float(st.sidebar.slider("Deadzone", 0.0, 0.1, 0.02, 0.005))

    st.sidebar.subheader("🖱️ Mouse Control")
    APP_STATE.mouse_enabled = st.sidebar.toggle(
        "Enable Mouse Control (pointer follows; click unlocks after calibration PASS)", value=False
    )
    APP_STATE.dwell_ms = st.sidebar.slider("Dwell Click (ms)", 200, 2000, APP_STATE.dwell_ms, 50)
    APP_STATE.dwell_radius = st.sidebar.slider("Dwell Radius (px)", 10, 120, APP_STATE.dwell_radius, 2)

    st.sidebar.subheader("🪞 Axes / Mirror")
    APP_STATE.mirror_input = st.sidebar.checkbox("Mirror webcam image", value=APP_STATE.mirror_input)
    APP_STATE.invert_x = st.sidebar.checkbox("Invert predicted X (x → 1−x)", value=APP_STATE.invert_x)
    APP_STATE.invert_y = st.sidebar.checkbox("Invert predicted Y (y → 1−y)", value=APP_STATE.invert_y)

    if mode == "ESP32 (Orlosky)":
        st.sidebar.subheader("ESP32")
        APP_STATE.esp32_url = st.sidebar.text_input("Stream URL", value=APP_STATE.esp32_url)
        col_e1, col_e2, col_e3 = st.sidebar.columns(3)
        start_esp = col_e1.button("Start")
        stop_esp = col_e2.button("Stop")
        APP_STATE.synthetic_hud = col_e3.toggle("HUD w/o Webcam", value=APP_STATE.synthetic_hud)
        APP_STATE.esp32_debug = st.sidebar.toggle("ESP32 Debug HUD (ROI & method)", value=APP_STATE.esp32_debug)
        if start_esp:
            if APP_STATE.esp32_stop is None:
                APP_STATE.esp32_q = queue.Queue(maxsize=4)
                APP_STATE.esp32_stop = threading.Event()
                t = ESP32Reader(APP_STATE.esp32_url, APP_STATE.esp32_q, APP_STATE.esp32_stop)
                t.start()
                st.toast("ESP32 reader started")
        if stop_esp and APP_STATE.esp32_stop is not None:
            APP_STATE.esp32_stop.set()
            APP_STATE.esp32_stop = None
            st.toast("ESP32 reader stopped")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🪄 Smoothing (One Euro)")
    APP_STATE.filter_mincutoff = float(st.sidebar.slider("mincutoff", 0.05, 3.0, APP_STATE.filter_mincutoff, 0.05))
    APP_STATE.filter_beta = float(st.sidebar.slider("beta", 0.0, 1.0, APP_STATE.filter_beta, 0.01))

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Compensation (bias + gain)")
    APP_STATE.comp_enabled = st.sidebar.toggle("Enable Compensation", value=APP_STATE.comp_enabled)
    APP_STATE.comp_alpha = st.sidebar.slider("Strength (alpha)", 0.0, 1.0, APP_STATE.comp_alpha, 0.05)

    st.sidebar.markdown("---")
    st.sidebar.subheader("✅ Click Gate Thresholds")
    mode = st.sidebar.selectbox("Mode", ["Strict", "Balanced", "Lenient"], index=["Strict","Balanced","Lenient"].index(APP_STATE.gate_mode))
    APP_STATE.gate_mode = mode
    if mode == "Strict":
        APP_STATE.gate_rmse_train_max = 30.0
        APP_STATE.gate_rmse_cv_max = 35.0
        APP_STATE.gate_uniformity_min = 0.55
        APP_STATE.gate_min_points = 9
    elif mode == "Balanced":
        APP_STATE.gate_rmse_train_max = 45.0
        APP_STATE.gate_rmse_cv_max = 55.0
        APP_STATE.gate_uniformity_min = 0.50
        APP_STATE.gate_min_points = 9
    else:
        APP_STATE.gate_rmse_train_max = 60.0
        APP_STATE.gate_rmse_cv_max = 75.0
        APP_STATE.gate_uniformity_min = 0.45
        APP_STATE.gate_min_points = 9
    st.sidebar.caption(f"RMSE≤{APP_STATE.gate_rmse_train_max}px / CV≤{APP_STATE.gate_rmse_cv_max}px · U≥{APP_STATE.gate_uniformity_min}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧭 Calibration")
    npoints = st.sidebar.selectbox("Points", [9, 12], index=1)  # default 12
    APP_STATE.calib_dwell_ms = st.sidebar.slider("Dwell per target (ms)", 400, 2000, APP_STATE.calib_dwell_ms, 50)
    if st.sidebar.button("Start Calibration"):
        if npoints == 9:
            grid = [
                (0.50, 0.50),
                (0.30, 0.50), (0.70, 0.50),
                (0.50, 0.30), (0.50, 0.70),
                (0.20, 0.20), (0.80, 0.20), (0.20, 0.80), (0.80, 0.80),
            ]
        else:
            grid = [
                (0.50, 0.50),
                (0.30, 0.50), (0.70, 0.50),
                (0.50, 0.30), (0.50, 0.70),
                (0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85),
                (0.30, 0.30), (0.70, 0.30), (0.50, 0.70),
            ]
        APP_STATE.calib_targets = grid
        APP_STATE.calib_idx = 0
        APP_STATE.calib_overlay_active = True
        sw, sh = _safe_size()
        APP_STATE.calib_radius_norm = max(0.012, 40 / max(sw, sh))
        st.toast("เริ่มคาลิเบรท (Fullscreen video)")

# ----------------------------- Overlay CSS -----------------------------
def _render_calibration_overlay(_ctx):
    if APP_STATE.calib_overlay_active:
        st.session_state["calib_ph"].empty()
        st.markdown(
            """
        <style>
          section[data-testid="stSidebar"] { display: none !important; }
          header, footer { display: none !important; }
          video { position: fixed !important; inset: 0 !important;
                  width: 100vw !important; height: 100vh !important;
                  object-fit: cover !important; z-index: 9999 !important; }
        </style>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.session_state["calib_ph"].empty()

# ----------------------------- Main -----------------------------
def main():
    st.set_page_config(page_title="Gaze All-in-One", page_icon="👀", layout="wide")
    st.title("👀 Gaze All-in-One — Webcam / ESP32 / Calibration Overlay (+ Compensation)")
    st.caption("ถ้าพรีวิวไม่ขึ้น ตรวจสิทธิ์กล้อง และติดตั้ง opencv-python / mediapipe")

    _sidebar_controls()
    ctx = webrtc_streamer(
        key="gaze-stream",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 60, "min": 30}},
            "audio": False,
        },
        async_processing=False,
        video_processor_factory=GazeProcessor,
    )
    _render_calibration_overlay(ctx)

    # Row 1: system/mode + realtime metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    mouse_state = "ON (click ✅)" if (APP_STATE.mouse_enabled and APP_STATE.mouse_gate_passed) else ("ON (click ⛔)" if APP_STATE.mouse_enabled else "OFF")
    with col1:
        st.metric("Mouse", mouse_state)
    with col2:
        st.metric("Mode", APP_STATE.extractor_mode)
    with col3:
        st.metric("ESP32", "RUNNING" if APP_STATE.esp32_stop is not None else "STOPPED")
    with col4:
        st.metric("FPS", f"{APP_STATE.ui_fps:.1f}")
    with col5:
        st.metric("Latency (ms)", f"{APP_STATE.ui_latency_ms:.1f}")
    with col6:
        sw = int(APP_STATE.screen_w_override) if APP_STATE.use_screen_override else _safe_size()[0]
        sh = int(APP_STATE.screen_h_override) if APP_STATE.use_screen_override else _safe_size()[1]
        st.metric("Screen (px)", f"{sw}×{sh}")

    st.markdown("### Diagnostics")
    st.write(
        {
            "calib_active": APP_STATE.calib_overlay_active,
            "calib_idx": APP_STATE.calib_idx,
            "n_targets": len(APP_STATE.calib_targets),
            "mirror_input": APP_STATE.mirror_input,
            "invert_x": APP_STATE.invert_x,
            "invert_y": APP_STATE.invert_y,
            "gaze": (round(APP_STATE.shared_gaze_x, 3), round(APP_STATE.shared_gaze_y, 3)),
            "click_gate": {"passed": APP_STATE.mouse_gate_passed, "reason": APP_STATE.mouse_gate_reason},
            "comp_enabled": APP_STATE.comp_enabled,
            "comp_alpha": APP_STATE.comp_alpha,
            "gate_mode": APP_STATE.gate_mode,
        }
    )

    # Evaluation block (RMSE/MAE log + download)
    if ctx and ctx.video_processor and ctx.video_processor.engine:
        eng = ctx.video_processor.engine
        rep = eng.get_calibration_report()
        st.markdown("---")
        colA, colB, colC, colD = st.columns([1, 1, 1, 2])
        with colA:
            if st.button("Start Drift"):
                eng.drift_start()
                st.toast("Drift enabled")
        with colB:
            if st.button("Stop Drift"):
                eng.drift_stop()
                st.toast("Drift disabled")
        with colC:
            if st.button("Add Drift Sample (current gaze ~ center)"):
                eng.drift_add(0.5, 0.5)
        with colD:
            st.write(eng.get_report())

        st.subheader("Calibration Report")
        if rep is None:
            st.info("Model: **Uncalibrated** – คลิกถูกล็อกจนกว่าจะคาลิเบรทผ่านเกณฑ์")
        else:
            ok, reason = eng.click_gate()
            chip = "✅ PASS" if ok else "⛔ LOCK"
            st.markdown(
                f"- {chip} · RMSE(train): **{rep.rmse_px:.0f}px**, RMSE(CV): **{rep.rmse_cv_px:.0f}px**, uniformity: **{rep.uniformity:.2f}**, points: **{rep.n_points}**"
            )
            if not ok:
                st.caption(f"เหตุผล: {reason}")

        st.subheader("Evaluation Log (RMSE/MAE)")
        log_rows = eng.get_eval_log()
        if pd is not None and log_rows:
            df = pd.DataFrame(log_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="gaze_eval_log.csv", mime="text/csv"
            )
        else:
            st.write(log_rows if log_rows else "ยังไม่มีบันทึก (ทำคาลิเบรทให้จบก่อน)")

        st.subheader("Profile")
        cS1, cS2 = st.columns([1, 1])
        with cS1:
            if st.button("Save profile"):
                ok = eng.save_profile("gaze_profile.json", meta={"ts": time.time()})
                st.toast("Saved" if ok else "Save failed")
        with cS2:
            if st.button("Load profile"):
                ok = eng.load_profile("gaze_profile.json")
                st.toast("Loaded" if ok else "Load failed")

    st.markdown(
        """
**ขั้นตอนใช้งานย่อย**
- ดูค่า **FPS / Latency** ได้ที่มุมบนขวาวิดีโอ และที่แถว Metrics ด้านบน
- ใช้ **Screen Override** หากใช้หลายจอ/สเกล Windows เพื่อให้ RMSE/MAE หน่วย px ถูกต้อง
- ทุกครั้งที่ **จบคาลิเบรท** ระบบจะบันทึก **RMSE / MAE** (normalized & pixel) ลง **Evaluation Log** และไฟล์ CSV
- ถ้า **uniformity ต่ำ** ระบบจะเพิ่ม 4 มุมให้อัตโนมัติแล้วคาลิเบรตต่ออีกรอบ (1 ครั้ง)
    """
    )

if __name__ == "__main__":
    main()