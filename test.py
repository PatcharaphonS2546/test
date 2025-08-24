# test.py
# 👀 Gaze All-in-One — Single-file build
# - Calibration วาด/นับบนวิดีโอเฟรม (ไม่รีเฟรชเพจ) + Assist
# - Progressive points, Stabilizer, Monolid/Unequal-eyes, Eye-first Weighted Ridge
# - Pointer follow always; click gate unlocks when PASS (RMSE/CV/Uniformity)
# Requires: streamlit>=1.26, streamlit-webrtc, opencv-python(-headless), mediapipe, av, numpy
# Optional (for real mouse control): pyautogui

import math, time, threading, queue, json, pathlib, os, random
from dataclasses import dataclass
from typing import Optional, Tuple, List
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

# ----------------------------- Mouse Controller -----------------------------
class MouseController:
    """
    - เคลื่อนเมาส์ต่อเนื่อง (rate-limited) เมื่อ enable=True
    - คลิกแบบ dwell เฉพาะเมื่อ do_click=True (ปล่อยให้ gate ควบคุม)
    """
    def __init__(self, enable: bool = False, dwell_ms: int = 700, dwell_radius_px: int = 40):
        self.sw, self.sh = _safe_size()
        self.enabled = enable
        self.dwell_ms = dwell_ms
        self.dwell_radius_px = dwell_radius_px
        self._last_in_time = None
        self._last_target = None
        self.move_rate_limit_ms = 33  # ~30 FPS
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

        # move pointer (rate-limited)
        now = time.time()
        if (now - self._last_move_at) * 1000.0 >= self.move_rate_limit_ms:
            if pyautogui is not None:
                try:
                    pyautogui.moveTo(x_px, y_px)
                except Exception:
                    pass
            self._last_move_at = now

        # dwell detection
        tgt = (x_px, y_px)
        if self._last_target is None or math.hypot(tgt[0] - self._last_target[0], tgt[1] - self._last_target[1]) > self.dwell_radius_px:
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
        self.freq = freq; self.mincutoff = float(mincutoff); self.beta = float(beta); self.dcutoff = float(dcutoff)
        self.x_prev = None; self.dx_prev = 0.0; self.t_prev = None

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

# ---- Report + Gate thresholds ----
@dataclass
class CalibrationReport:
    n_points: int
    rmse_px: Optional[float]
    rmse_cv_px: Optional[float]
    uniformity: float   # 0..1 convex hull area on normalized screen
    width: int
    height: int
    def passed(self) -> bool:
        if self.n_points < 9: return False
        if self.uniformity < 0.55: return False
        if self.rmse_px is None or self.rmse_cv_px is None: return False
        return (self.rmse_px <= 30.0 and self.rmse_cv_px <= 35.0)

class SessionState:
    def __init__(self):
        self.smooth_x = OneEuroFilter(120, 1.0, 0.01, 1.0)
        self.smooth_y = OneEuroFilter(120, 1.0, 0.01, 1.0)
        self.calib_X: List[List[float]] = []
        self.calib_yx: List[float] = []
        self.calib_yy: List[float] = []
        self.calib_w: List[float] = []      # NEW: sample weight ต่อจุดคาลิเบรท
        self.model_ready = False
        self.model_pipeline = None          # (pipe_x, pipe_y) หรือ (wx, wy)
        self.last_feat: Optional[np.ndarray] = None
        self.last_quality: float = 0.5      # NEW: quality ล่าสุดจาก extractor
        self.t_prev = None
        self.fps_hist: List[float] = []
        # Drift
        self.drift_enabled = False
        self.drift_dx = 0.0
        self.drift_dy = 0.0
        self.drift_alpha = 0.15
        # Cached report
        self.calib_report: Optional[CalibrationReport] = None

try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge
except Exception:
    Pipeline = PolynomialFeatures = Ridge = None

# ----------------------------- Feature Vector -----------------------------
def _feat_vec(feat: dict) -> np.ndarray:
    # ลดน้ำหนักฟีเจอร์ใบหน้า (fx, fy) ตั้งแต่ต้นทาง เพื่อลดการยึดหัว
    ex = float(feat.get('eye_cx_norm', 0.5))
    ey = float(feat.get('eye_cy_norm', 0.5))
    fx = float(feat.get('face_cx_norm', 0.5))
    fy = float(feat.get('face_cy_norm', 0.5))
    fx = 0.5 + (fx - 0.5) * 0.3
    fy = 0.5 + (fy - 0.5) * 0.3
    return np.array([ex, ey, fx, fy, 1.0], dtype=np.float32)

def _poly_expand(X: np.ndarray) -> np.ndarray:
    if X.ndim == 1:
        X = X[None, :]
    cols = [X]; n = X.shape[1]
    for i in range(n):
        for j in range(i, n):
            cols.append((X[:, i] * X[:, j])[:, None])
    return np.concatenate(cols, axis=1)

# ---- Penalty & Weighted Ridge helpers ----
def _build_penalty_vector(n_base: int = 5,
                          eye_pen: float = 0.1,
                          face_pen: float = 8.0,
                          bias_pen: float = 0.5) -> np.ndarray:
    """
    คืน penalty ต่อคอลัมน์ของ Phi (poly degree=2)
    base features index: 0=ex, 1=ey, 2=fx, 3=fy, 4=const(1)
    - เทอมที่มี fx/fy → ลงโทษหนัก (ลดการพึ่งหัว)
    - เทอมที่มี ex/ey อย่างเดียว → เบา (ให้พึ่งตา)
    """
    n = n_base
    base_pen = [eye_pen, eye_pen, face_pen, face_pen, bias_pen]
    pens = list(base_pen)
    for i in range(n):
        for j in range(i, n):
            pens.append(max(base_pen[i], base_pen[j]))
    return np.array(pens, dtype=np.float32)

def _fit_weighted_ridge(Phi: np.ndarray, y: np.ndarray,
                        sample_w: np.ndarray, pen_vec: np.ndarray, lam: float) -> np.ndarray:
    """
    แก้ (Phi^T W Phi + lam * diag(pen_vec)) w = Phi^T W y
    โดย W = diag(sample_w)
    """
    sw = np.sqrt(np.clip(sample_w.reshape(-1, 1), 0.0, 1e6))
    Phi_w = Phi * sw
    y_w = y.reshape(-1, 1) * sw
    A = Phi_w.T @ Phi_w + lam * np.diag(pen_vec.astype(np.float64))
    b = Phi_w.T @ y_w
    w = np.linalg.solve(A, b)  # (D x 1)
    return w.reshape(-1)

# ---- Accuracy helpers ----
def _rmse_px(pred_xy: np.ndarray, true_xy: np.ndarray, W: int, H: int) -> float:
    dx = (pred_xy[:, 0] - true_xy[:, 0]) * W
    dy = (pred_xy[:, 1] - true_xy[:, 1]) * H
    return float(np.sqrt(np.mean(dx*dx + dy*dy)))

def _uniformity_score(points_xy_norm: np.ndarray) -> float:
    pts = np.asarray(points_xy_norm, float)
    if len(pts) < 3: return 0.0
    pts = pts[np.lexsort((pts[:,1], pts[:,0]))]
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower=[]
    for p in pts:
        while len(lower)>=2 and cross(lower[-2], lower[-1], p) <= 0: lower.pop()
        lower.append(p)
    upper=[]
    for p in pts[::-1]:
        while len(upper)>=2 and cross(upper[-2], upper[-1], p) <= 0: upper.pop()
        upper.append(p)
    hull = np.vstack([lower[:-1], upper[:-1]])
    area = 0.0
    for i in range(len(hull)):
        x1,y1 = hull[i]; x2,y2 = hull[(i+1)%len(hull)]
        area += x1*y2 - x2*y1
    area = abs(area)/2.0
    return float(max(0.0, min(1.0, area)))

# ----------------------------- MediaPipe FeatureExtractor -----------------------------
class FeatureExtractor:
    # ใช้ Mediapipe FaceMesh + Iris (รองรับ monolid/unequal-eyes ด้วย per-eye weighting)
    LEFT_EYE_IDS = [33, 133, 159, 145]
    RIGHT_EYE_IDS = [362, 263, 386, 374]
    LEFT_IRIS_IDS = [468, 469, 470, 471, 472]
    RIGHT_IRIS_IDS = [473, 474, 475, 476, 477] if mp is not None else [468, 469, 470, 471, 472]

    def __init__(self, use_mediapipe: bool = True):
        self.use_mediapipe = use_mediapipe and (mp is not None) and (cv2 is not None)
        if self.use_mediapipe:
            self.mp_face = mp.solutions.face_mesh
            self.mesh = self.mp_face.FaceMesh(
                static_image_mode=False, refine_landmarks=True, max_num_faces=1,
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            )
        else:
            self.mesh = None

        # Adaptive baseline EAR ต่อข้าง (EMA)
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
        return (sum(xs)/len(xs), sum(ys)/len(ys))

    @staticmethod
    def _eye_box_metrics(ids, pts):
        # คืน (x0,x1,y0,y1,w,h,EAR_simple)
        xs = [pts[i][0] for i in ids if i < len(pts)]
        ys = [pts[i][1] for i in ids if i < len(pts)]
        if not xs or not ys:
            return None
        x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
        w = max(6.0, x1 - x0); h = max(2.0, y1 - y0)
        ear = h / max(6.0, w)  # EAR อย่างง่าย
        return (x0, x1, y0, y1, w, h, ear)

    @staticmethod
    def _norm_in_box(p, box):
        if p is None or box is None:
            return (0.5, 0.5)
        x0, x1, y0, y1, w, h, _ = box
        nx = (p[0] - x0) / w; ny = (p[1] - y0) / h
        return (min(1.0, max(0.0, nx)), min(1.0, max(0.0, ny)))

    def _extract_mediapipe(self, frame_bgr) -> Optional[dict]:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(frame_rgb)
        if not res.multi_face_landmarks:
            return None

        lm = res.multi_face_landmarks[0]
        pts = [(lm.landmark[i].x * w, lm.landmark[i].y * h, lm.landmark[i].z) for i in range(len(lm.landmark))]

        # iris center ต่อข้าง
        l_iris = self._avg_xy(self.LEFT_IRIS_IDS, pts)
        r_iris = self._avg_xy(self.RIGHT_IRIS_IDS, pts)
        if l_iris is None and r_iris is None:
            return None

        # กรอบตา + EAR
        L_EYE = self.LEFT_EYE_IDS; R_EYE = self.RIGHT_EYE_IDS
        l_box = self._eye_box_metrics(L_EYE, pts)
        r_box = self._eye_box_metrics(R_EYE, pts)

        l_n = self._norm_in_box(l_iris, l_box) if l_iris is not None else (0.5, 0.5)
        r_n = self._norm_in_box(r_iris, r_box) if r_iris is not None else (0.5, 0.5)

        # EAR baseline ด้วย EMA
        def _upd_base(cur, base):
            if cur is None:
                return base
            return cur if base is None else (base*(1.0-self._ema_alpha) + cur*self._ema_alpha)

        earL = l_box[6] if l_box is not None else None
        earR = r_box[6] if r_box is not None else None
        if earL is not None and earL > 0.18:
            self.earL_base = _upd_base(earL, self.earL_base)
        if earR is not None and earR > 0.18:
            self.earR_base = _upd_base(earR, self.earR_base)

        # per-eye confidence (0..1)
        def _eye_conf(ear, base, box, iris_ok):
            if (ear is None) or (box is None) or (not iris_ok):
                return 0.0
            ratio = (ear / 0.26) if base is None else (ear / max(1e-6, 0.8*base))
            ratio = max(0.0, min(1.4, ratio))
            _, _, _, _, bw, bh, _ = box
            area_norm = (bw*bh) / (w*h)
            area_boost = min(1.0, area_norm / 0.02)
            conf = (0.7*ratio) + (0.3*area_boost)
            return max(0.0, min(1.0, conf))

        cL = _eye_conf(earL, self.earL_base, l_box, l_iris is not None)
        cR = _eye_conf(earR, self.earR_base, r_box, r_iris is not None)

        eye_open = bool((earL or 0.0) > 0.20 or (earR or 0.0) > 0.20)

        # รวมศูนย์กลางม่านตาแบบถ่วงน้ำหนัก
        wL, wR = cL, cR
        if (wL + wR) < 1e-3:
            eye_cx_norm = 0.5; eye_cy_norm = 0.5
        else:
            eye_cx_norm = float((l_n[0]*wL + r_n[0]*wR) / (wL + wR))
            eye_cy_norm = float((l_n[1]*wL + r_n[1]*wR) / (wL + wR))

        # face center (ชดเชย head translation)
        face_ids = [1, 9, 152, 33, 263]
        fxs = [pts[i][0] for i in face_ids if i < len(pts)]
        fys = [pts[i][1] for i in face_ids if i < len(pts)]
        face_cx_norm = float(sum(fxs)/len(fxs) / max(1, w)) if fxs else 0.5
        face_cy_norm = float(sum(fys)/len(fys) / max(1, h)) if fys else 0.5

        # overall quality
        box_margin = min(eye_cx_norm, 1-eye_cx_norm, eye_cy_norm, 1-eye_cy_norm)
        quality = max(cL, cR) * (0.6 + 0.4*max(0.0, min(1.0, box_margin*2)))

        return {
            "eye_cx_norm": eye_cx_norm, "eye_cy_norm": eye_cy_norm,
            "face_cx_norm": face_cx_norm, "face_cy_norm": face_cy_norm,
            "eye_open": eye_open, "quality": float(quality),
            "eye_conf_L": float(cL), "eye_conf_R": float(cR),
            "earL": float(earL or 0.0), "earR": float(earR or 0.0),
            "earL_base": float(self.earL_base or 0.0), "earR_base": float(self.earR_base or 0.0),
        }

    def extract(self, frame_bgr) -> Optional[dict]:
        if frame_bgr is None:
            return None
        if self.use_mediapipe and self.mesh is not None:
            try:
                return self._extract_mediapipe(frame_bgr)
            except Exception:
                return None
        # fallback (ไม่มี mediapipe)
        return {"eye_cx_norm":0.5,"eye_cy_norm":0.5,"face_cx_norm":0.5,"face_cy_norm":0.5,
                "eye_open":True,"quality":0.2,"eye_conf_L":0.1,"eye_conf_R":0.1,
                "earL":0.2,"earR":0.2,"earL_base":0.2,"earR_base":0.2}

# ----------------------------- ESP32 Reader -----------------------------
def _crop_to_aspect_ratio(image, width=640, height=480):
    h, w = image.shape[:2]; desired = width/height; cur = w/h
    if cur > desired:
        new_w = int(desired * h); off = (w - new_w) // 2; image = image[:, off:off+new_w]
    else:
        new_h = int(w/desired); off = (h - new_h)//2; image = image[off:off+new_h, :]
    return cv2.resize(image, (width, height)) if cv2 is not None else image

def _get_darkest_area(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ignore = 20; step = 10; win = 20
    best_sum, best_pt = 1e18, None
    for y in range(ignore, gray.shape[0] - ignore, step):
        for x in range(ignore, gray.shape[1] - ignore, step):
            tile = gray[y:y+win, x:x+win]; s = float(tile.sum())
            if s < best_sum: best_sum, best_pt = s, (x + win//2, y + win//2)
    return best_pt

def _iris_center_norm_from_frame(frame_bgr):
    if cv2 is None: return None
    frame = _crop_to_aspect_ratio(frame_bgr, 640, 480)
    h, w = frame.shape[:2]
    pt = _get_darkest_area(frame) or (w//2, h//2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    darkest = int(gray[pt[1], pt[0]])
    _, th = cv2.threshold(gray, darkest + 12, 255, cv2.THRESH_BINARY_INV)
    mask = np.zeros_like(th); size = 250; x, y = pt; hs = size // 2
    mask[max(0,y-hs):min(h,y+hs), max(0,x-hs):min(w,x+hs)] = 255
    th = cv2.bitwise_and(th, mask)
    th = cv2.dilate(th, np.ones((5,5), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= 1000]
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5: return None
    (cx, cy), (a, b), _ = cv2.fitEllipse(c)
    if max(a,b)/max(1e-6, min(a,b)) > 3.0: return None
    nx = float(cx) / float(w); ny = float(cy) / float(h)
    return (min(1.0, max(0.0, nx)), min(1.0, max(0.0, ny)))

class ESP32Reader(threading.Thread):
    def __init__(self, url: str, out_q: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True); self.url = url; self.out_q = out_q; self.stop_event = stop_event
    def run(self):
        if cv2 is None: return
        while not self.stop_event.is_set():
            cap = None
            try:
                cap = cv2.VideoCapture(self.url)
                try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception: pass
                if not cap.isOpened(): time.sleep(1.0); continue
                for _ in range(3): cap.grab()
                last_ok = time.time()
                while not self.stop_event.is_set():
                    for _ in range(2): cap.grab()
                    ok, frame = cap.read()
                    if not ok or frame is None: break
                    last_ok = time.time()
                    res = _iris_center_norm_from_frame(frame)
                    if res is not None:
                        nx, ny = res
                        try: self.out_q.put((nx, ny), timeout=0.005)
                        except Exception: pass
                    if (time.time() - last_ok) > 3.0: break
            except Exception:
                time.sleep(1.0)
            finally:
                if cap is not None: cap.release()

# ----------------------------- Gaze Engine -----------------------------
def _make_model():
    if Pipeline and PolynomialFeatures and Ridge:
        return (
            Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=True)), ('reg', Ridge(alpha=1.0))]),
            Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=True)), ('reg', Ridge(alpha=1.0))])
        )
    else:
        return ("manual_poly_ridge", "manual_poly_ridge")

def _fit_model_dispatch(model_tuple, X: np.ndarray, Y: np.ndarray):
    # Y: Nx2 normalized
    if isinstance(model_tuple[0], str):
        Phi = _poly_expand(X); lam = 1e-3
        A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
        bx = Phi.T @ Y[:,0]; by = Phi.T @ Y[:,1]
        wx = np.linalg.solve(A, bx); wy = np.linalg.solve(A, by)
        return (wx, wy)
    else:
        pipe_x, pipe_y = model_tuple
        pipe_x = pipe_x.fit(X, Y[:,0])
        pipe_y = pipe_y.fit(X, Y[:,1])
        return (pipe_x, pipe_y)

def _predict_dispatch(fitted_tuple, X: np.ndarray) -> np.ndarray:
    if isinstance(fitted_tuple[0], np.ndarray) or isinstance(fitted_tuple[0], list):
        wx, wy = fitted_tuple
        Phi = _poly_expand(X)
        x = Phi @ wx; y = Phi @ wy
        return np.stack([x, y], axis=1).astype(np.float32)
    if isinstance(fitted_tuple[0], str):  # shouldn't happen
        raise RuntimeError("Model not fitted")
    pipe_x, pipe_y = fitted_tuple
    x = pipe_x.predict(X); y = pipe_y.predict(X)
    return np.stack([x, y], axis=1).astype(np.float32)

class GazeEngine:
    def __init__(self, cfg: GazeConfig, extractor: FeatureExtractor):
        self.cfg = cfg; self.ext = extractor; self.sess = SessionState()

    def update_config(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.cfg, k): setattr(self.cfg, k, v)

    # ---- Stability layer (reject outliers / clamp velocity / freeze on low quality) ----
    def _stabilize(self, x: float, y: float, quality: float, eye_open: bool) -> Tuple[float, float]:
        if not hasattr(self, "_stab_last"):
            self._stab_last = (0.5, 0.5)
            self._stab_hist = []
            self._stab_last_t = time.time()

        last_x, last_y = self._stab_last
        now = time.time()
        dt = max(1e-3, now - self._stab_last_t)

        # 1) freeze เมื่อคุณภาพต่ำ/หลับตา
        if (not eye_open) or (quality < 0.25):
            return last_x, last_y

        # 2) median guard
        self._stab_hist.append((x, y))
        if len(self._stab_hist) > 5: self._stab_hist = self._stab_hist[-5:]
        medx = float(np.median([p[0] for p in self._stab_hist]))
        medy = float(np.median([p[1] for p in self._stab_hist]))
        thr = 0.10 - 0.05 * quality   # 0.10..0.05
        if math.hypot(x - medx, y - medy) > thr:
            x, y = medx, medy

        # 3) velocity clamp
        max_step = (0.08 + 0.06 * (1.0 - quality)) * dt * 60.0  # ต่อเฟรม อิง 60fps
        dx, dy = x - last_x, y - last_y
        dist = math.hypot(dx, dy)
        if dist > max_step:
            scale = max_step / max(1e-6, dist)
            x = last_x + dx * scale
            y = last_y + dy * scale

        self._stab_last = (x, y); self._stab_last_t = now
        return x, y

    # Calibration (เก็บจุด + sample weight)
    def calibration_add_point(self, screen_x_norm: float, screen_y_norm: float):
        if self.sess.last_feat is None: 
            return False
        x = self.sess.last_feat.astype(np.float32)
        self.sess.calib_X.append(x.tolist())
        self.sess.calib_yx.append(float(screen_x_norm))
        self.sess.calib_yy.append(float(screen_y_norm))
        # sample weight จาก quality + บูสต์จุดบน/ล่าง (ช่วยแกน Y)
        q = float(self.sess.last_quality or 0.5)
        w_q = max(0.0, min(1.0, (q - 0.20) / 0.80)) ** 2
        w_y = 1.0 + 0.6 * abs(screen_y_norm - 0.5) * 2.0
        w = (0.5 + 1.5 * w_q) * w_y
        self.sess.calib_w.append(float(w))
        return True

    def _compute_report(self, X: np.ndarray, Y: np.ndarray) -> CalibrationReport:
        W, H = int(self.cfg.screen_w), int(self.cfg.screen_h)
        pred_train = self._map_from_feat_batch(X)
        rmse_train = _rmse_px(pred_train, Y, W, H)

        # CV
        def _cv_rmse():
            n = len(X)
            if n < 5: return float('inf')
            if n >= 9:
                k = max(2, min(5, n // 3))
                idx = np.arange(n); np.random.shuffle(idx)
                folds = np.array_split(idx, k)
                acc=[]
                for i in range(k):
                    te = folds[i]
                    tr = np.concatenate([folds[j] for j in range(k) if j != i])
                    model = _make_model()
                    model = _fit_model_dispatch(model, X[tr], Y[tr])
                    pred = _predict_dispatch(model, X[te])
                    acc.append(_rmse_px(pred, Y[te], W, H))
                return float(np.mean(acc))
            else:
                reps=3; acc=[]
                for _ in range(reps):
                    idx = np.arange(n); np.random.shuffle(idx)
                    cut = max(2, int(0.8*n))
                    tr, te = idx[:cut], idx[cut:]
                    model = _make_model()
                    model = _fit_model_dispatch(model, X[tr], Y[tr])
                    pred = _predict_dispatch(model, X[te])
                    acc.append(_rmse_px(pred, Y[te], W, H))
                return float(np.mean(acc))

        rmse_cv = _cv_rmse()
        unif = _uniformity_score(Y)
        return CalibrationReport(
            n_points=len(X), rmse_px=rmse_train, rmse_cv_px=rmse_cv,
            uniformity=unif, width=W, height=H
        )

    def calibration_finish(self):
        n = len(self.sess.calib_X)
        if n < 6:
            self.sess.model_ready = False; self.sess.model_pipeline = None
            self.sess.calib_report = None
            return {'ok': False, 'msg': 'ต้องเก็บคาลิเบรทอย่างน้อย 6 จุด'}

        X = np.array(self.sess.calib_X, dtype=np.float32)           # (N x 5)
        Y = np.stack([np.array(self.sess.calib_yx, dtype=np.float32),
                      np.array(self.sess.calib_yy, dtype=np.float32)], axis=1)  # (N x 2)
        Wsamples = np.array(self.sess.calib_w if self.sess.calib_w else [1.0]*n, dtype=np.float32)  # (N,)

        # poly degree=2 + Eye-first penalties
        Phi = _poly_expand(X)  # (N x D)
        pen_vec = _build_penalty_vector(n_base=X.shape[1], eye_pen=0.1, face_pen=6.0, bias_pen=0.5)
        lam = 1e-3

        # ฟิต weighted ridge แยกแกน x,y
        wx = _fit_weighted_ridge(Phi, Y[:, 0], Wsamples, pen_vec, lam)
        wy = _fit_weighted_ridge(Phi, Y[:, 1], Wsamples, pen_vec, lam)
        self.sess.model_pipeline = (wx, wy)
        self.sess.model_ready = True

        # รายงาน
        rep = self._compute_report(X, Y)
        self.sess.calib_report = rep
        preds = self._map_from_feat_batch(X)
        rmse_norm = float(np.sqrt(np.mean((preds - Y) ** 2)))

        return {
            'ok': True, 'rmse_norm': rmse_norm, 'n': n,
            'rmse_px': rep.rmse_px, 'rmse_cv_px': rep.rmse_cv_px, 'uniformity': rep.uniformity,
            'passed': rep.passed()
        }

    # Save/Load
    def save_profile(self, path: str, meta: dict=None):
        p = pathlib.Path(path)
        data = {
            "calib_X": self.sess.calib_X, "calib_yx": self.sess.calib_yx, "calib_yy": self.sess.calib_yy,
            "screen": {"w": self.cfg.screen_w, "h": self.cfg.screen_h}, "meta": meta or {}
        }
        try: p.write_text(json.dumps(data)); return True
        except Exception: return False

    def load_profile(self, path: str):
        p = pathlib.Path(path)
        if not p.exists(): return False
        try:
            data = json.loads(p.read_text())
            self.sess.calib_X = data.get("calib_X", [])
            self.sess.calib_yx = data.get("calib_yx", [])
            self.sess.calib_yy = data.get("calib_yy", [])
            sc = data.get("screen", {})
            if sc:
                self.cfg.screen_w = sc.get("w", self.cfg.screen_w)
                self.cfg.screen_h = sc.get("h", self.cfg.screen_h)
            rep = self.calibration_finish()
            return bool(rep.get("ok", False))
        except Exception:
            return False

    # Drift
    def drift_start(self): self.sess.drift_enabled = True
    def drift_stop(self): self.sess.drift_enabled = False
    def drift_add(self, screen_x_norm: float, screen_y_norm: float):
        if self.sess.last_feat is None: return False
        pred = self._map_from_feat(self.sess.last_feat)
        ex = screen_x_norm - pred[0]; ey = screen_y_norm - pred[1]
        a = self.sess.drift_alpha
        self.sess.drift_dx = (1 - a) * self.sess.drift_dx + a * ex
        self.sess.drift_dy = (1 - a) * self.sess.drift_dy + a * ey
        return True

    def drift_status(self): return {'enabled': self.sess.drift_enabled, 'dx': self.sess.drift_dx, 'dy': self.sess.drift_dy}

    # Mapping
    def _map_from_feat(self, feat_vec: np.ndarray) -> np.ndarray:
        # แผนที่แบบตาล้วน (ไม่ใช้ใบหน้า) — ช่วยยกสัญญาณ Y และลด head-coupling
        ex, ey = float(feat_vec[0]), float(feat_vec[1])
        eye_only = np.array([
            0.5 + (ex - 0.5) * self.cfg.fallback_kx,
            0.5 + (ey - 0.5) * self.cfg.fallback_ky
        ], dtype=np.float32)

        if self.sess.model_ready and self.sess.model_pipeline is not None:
            pm = _predict_dispatch(self.sess.model_pipeline, feat_vec[None, :])[0]
            alpha_x = 0.30  # ผสานกลับมาตาล้วนเล็กน้อยสำหรับ X
            alpha_y = 0.55  # ผสานกลับมาตาล้วนมากกว่าใน Y
            px = (1.0 - alpha_x) * pm[0] + alpha_x * eye_only[0]
            py = (1.0 - alpha_y) * pm[1] + alpha_y * eye_only[1]
            return np.array([px, py], dtype=np.float32)
        else:
            return eye_only

    def _map_from_feat_batch(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1: X = X[None, :]
        if self.sess.model_ready and self.sess.model_pipeline is not None:
            return _predict_dispatch(self.sess.model_pipeline, X)
        ex, ey = X[:,0], X[:,1]
        x = 0.5 + (ex - 0.5) * self.cfg.fallback_kx
        y = 0.5 + (ey - 0.5) * self.cfg.fallback_ky
        return np.stack([x, y], axis=1).astype(np.float32)

    def _shape_and_smooth(self, x: float, y: float) -> Tuple[float, float]:
        if self.sess.drift_enabled: x += self.sess.drift_dx; y += self.sess.drift_dy
        gx = self.cfg.gain * self.cfg.gain_x; gy = self.cfg.gain * self.cfg.gain_y
        x = 0.5 + (x - 0.5) * gx; y = 0.5 + (y - 0.5) * gy
        if self.cfg.gamma != 1.0:
            def _gamma(v):
                s = (v - 0.5); sign = 1.0 if s >= 0 else -1.0
                return 0.5 + sign * (abs(s) ** self.cfg.gamma)
            x = _gamma(x); y = _gamma(y)
        dz = self.cfg.deadzone
        def _dead(v): d = v - 0.5; return 0.5 if abs(d) < dz else v
        x = _dead(x); y = _dead(y)
        x = min(1.0, max(0.0, x)); y = min(1.0, max(0.0, y))
        t = time.time(); x = float(self.sess.smooth_x.filter(x, t)); y = float(self.sess.smooth_y.filter(y, t))
        return x, y

    # Processing
    def process_frame(self, frame_bgr) -> Tuple[float, float, dict, np.ndarray]:
        t0 = time.time()
        feat = self.ext.extract(frame_bgr)
        if feat is None:
            pred = np.array([0.5, 0.5], dtype=np.float32)
            q = 0.2; eye_open = True
        else:
            if (not feat.get("eye_open", True)) and self.sess.last_feat is not None:
                fv = self.sess.last_feat
            else:
                fv = _feat_vec(feat); self.sess.last_feat = fv
            pred = self._map_from_feat(fv)
            q = float(feat.get("quality", 0.5))
            eye_open = bool(feat.get("eye_open", True))
            self.sess.last_quality = q  # NEW: เก็บคุณภาพล่าสุด

        # stabilize -> shape/smooth
        px, py = float(pred[0]), float(pred[1])
        px, py = self._stabilize(px, py, q, eye_open)
        x, y = self._shape_and_smooth(px, py)

        t1 = time.time()
        if self.sess.t_prev is not None:
            dt = t1 - self.sess.t_prev
            if dt > 0:
                self.sess.fps_hist.append(1.0 / dt)
                if len(self.sess.fps_hist) > 90: self.sess.fps_hist = self.sess.fps_hist[-90:]
        self.sess.t_prev = t1
        metrics = {
            'fps': float(np.mean(self.sess.fps_hist)) if len(self.sess.fps_hist) else 0.0,
            'latency_ms': float((t1 - t0) * 1000.0),
            'model': 'calibrated' if self.sess.model_ready else 'fallback'
        }
        return x, y, metrics, frame_bgr

    def process_external(self, nx: float, ny: float) -> Tuple[float, float]:
        fv = np.array([nx, ny, 0.5, 0.5, 1.0], dtype=np.float32)
        self.sess.last_feat = fv
        pred = self._map_from_feat(fv)
        px, py = float(pred[0]), float(pred[1])
        px, py = self._stabilize(px, py, 0.5, True)
        x, y = self._shape_and_smooth(px, py)
        self.sess.last_quality = 0.5  # NEW: ค่ากลางสำหรับ external feed
        return x, y

    # Reporting & gate
    def get_report(self):
        rmse = None
        if self.sess.model_ready and len(self.sess.calib_X) >= 6:
            X = np.array(self.sess.calib_X, dtype=np.float32)
            Y = np.stack([np.array(self.sess.calib_yx, dtype=np.float32),
                          np.array(self.sess.calib_yy, dtype=np.float32)], axis=1)
            preds = self._map_from_feat_batch(X)
            rmse = float(np.sqrt(np.mean((preds - Y) ** 2)))
        return {'model_ready': self.sess.model_ready, 'rmse_norm': rmse, 'n_samples': len(self.sess.calib_X)}

    def get_calibration_report(self) -> Optional[CalibrationReport]:
        return self.sess.calib_report

    def click_gate(self) -> Tuple[bool, str]:
        rep = self.sess.calib_report
        if rep is None:
            return False, "ยังไม่คาลิเบรท"
        reasons = []
        if rep.n_points < 9: reasons.append(f"จุดน้อย ({rep.n_points}/9)")
        if rep.uniformity < 0.55: reasons.append(f"กระจายตัวต่ำ ({rep.uniformity:.2f})")
        if rep.rmse_px is None or rep.rmse_px > 30.0: reasons.append(f"RMSE เทรน {rep.rmse_px:.0f}px > 30px" if rep.rmse_px is not None else "RMSE เทรนไม่พร้อม")
        if rep.rmse_cv_px is None or rep.rmse_cv_px > 35.0: reasons.append(f"RMSE CV {rep.rmse_cv_px:.0f}px > 35px" if rep.rmse_cv_px is not None else "RMSE CV ไม่พร้อม")
        passed = (len(reasons) == 0)
        return passed, ("ผ่าน" if passed else " , ".join(reasons))

# ----------------------------- App State -----------------------------
class AppState:
    def __init__(self):
        # Modes / IO
        self.extractor_mode = 'Webcam/MediaPipe'
        self.mouse_enabled = False      # user toggle
        self.dwell_ms = 700
        self.dwell_radius = 40
        self.esp32_url = 'http://esp32.local/stream'
        self.esp32_q: Optional[queue.Queue] = None
        self.esp32_stop: Optional[threading.Event] = None
        self.synthetic_hud = False
        # Diagnostics
        self.log_enabled = False
        self.log_path = 'gaze_metrics.jsonl'
        # Axes & mirror
        self.mirror_input = False
        self.invert_x = False
        self.invert_y = False
        # Shaping & filtering
        self.gain = 1.2       # เริ่มสูงเล็กน้อย ช่วยไปถึงมุมช่วง fallback
        self.gamma = 1.0
        self.deadzone = 0.02
        self.filter_mincutoff = 1.0
        self.filter_beta = 0.01
        # Shared gaze
        self.shared_gaze_x = 0.5
        self.shared_gaze_y = 0.5
        # Calibration (global)
        self.calib_overlay_active = False
        self.calib_targets: List[Tuple[float, float]] = []
        self.calib_idx = 0
        self.calib_dwell_ms = 1000  # ให้เวลาถือมากขึ้นตอนเริ่ม
        self.calib_radius_norm = 0.02  # ~2% ของจอ
        # Gate status
        self.mouse_gate_passed = False
        self.mouse_gate_reason = "ยังไม่คาลิเบรท"

# persist
if 'APP_STATE' not in st.session_state:
    st.session_state['APP_STATE'] = AppState()
APP_STATE: AppState = st.session_state['APP_STATE']

# placeholder (CSS overlay only)
if "calib_ph" not in st.session_state:
    st.session_state["calib_ph"] = st.empty()

# ----------------------------- Streamlit WebRTC Processor -----------------------------
class GazeProcessor(VideoProcessorBase):
    def __init__(self):
        self.engine: Optional[GazeEngine] = None
        self.mouse: Optional[MouseController] = None
        self.last_gaze = (0.5, 0.5)
        # Calibration-in-video state
        self.calib_active = False
        self.calib_targets = []
        self.calib_idx = 0
        self.calib_hold_start = None
        self.calib_radius_norm = 0.02
        self._calib_banner_until = 0.0
        self._calib_banner_text = ""

    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # mirror webcam image before extract
        if APP_STATE.extractor_mode == 'Webcam/MediaPipe' and APP_STATE.mirror_input and cv2 is not None:
            img = cv2.flip(img, 1)

        if APP_STATE.extractor_mode == 'ESP32 (Orlosky)' and getattr(APP_STATE, 'synthetic_hud', False):
            img = np.full((480, 640, 3), 40, dtype=np.uint8)

        if self.engine is None:
            extractor = FeatureExtractor(use_mediapipe=(APP_STATE.extractor_mode == 'Webcam/MediaPipe'))
            cfg = GazeConfig()
            self.engine = GazeEngine(cfg, extractor)
            self.mouse = MouseController(enable=False,
                                         dwell_ms=APP_STATE.dwell_ms,
                                         dwell_radius_px=APP_STATE.dwell_radius)

        # Apply shaping/filter params
        try:
            self.engine.update_config(gain=float(APP_STATE.gain), gamma=float(APP_STATE.gamma), deadzone=float(APP_STATE.deadzone))
            for f in (self.engine.sess.smooth_x, self.engine.sess.smooth_y):
                f.mincutoff = float(APP_STATE.filter_mincutoff)
                f.beta = float(APP_STATE.filter_beta)
        except Exception:
            pass

        # === Calibration Assist: ขยายระยะวิ่ง + ปิด deadzone เฉพาะตอน calibrating ===
        if APP_STATE.calib_overlay_active and self.engine is not None:
            if not hasattr(self, "_saved_calib_assist"):
                self._saved_calib_assist = (
                    self.engine.cfg.fallback_kx, self.engine.cfg.fallback_ky, self.engine.cfg.deadzone
                )
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
        if APP_STATE.extractor_mode == 'Webcam/MediaPipe':
            x, y, metrics, img = self.engine.process_frame(img)
        else:
            x, y = self.last_gaze
            if APP_STATE.esp32_q is not None:
                try:
                    nx, ny = APP_STATE.esp32_q.get_nowait()
                    x, y = self.engine.process_external(nx, ny)
                except Exception:
                    pass
            metrics = self.engine.get_report()

        # invert axes (if chosen)
        if APP_STATE.invert_x: x = 1.0 - x
        if APP_STATE.invert_y: y = 1.0 - y
        self.last_gaze = (x, y)

        # share gaze for other UI
        APP_STATE.shared_gaze_x = float(x)
        APP_STATE.shared_gaze_y = float(y)

        # ==== Calibration flow INSIDE video (no page refresh) ====
        if APP_STATE.calib_overlay_active:
            if not self.calib_active:
                self.calib_active = True
                self.calib_targets = list(APP_STATE.calib_targets)
                self.calib_idx = APP_STATE.calib_idx
                self.calib_hold_start = None
                self.calib_radius_norm = float(APP_STATE.calib_radius_norm)

            if 0 <= self.calib_idx < len(self.calib_targets):
                tx, ty = self.calib_targets[self.calib_idx]
            else:
                tx, ty = (0.5, 0.5)

            # ---- hold timing with early-wide radius for first 3 points ----
            early_wide = 0.06  # 6% ของจอ
            radius_now = self.calib_radius_norm if self.calib_idx >= 3 else max(self.calib_radius_norm, early_wide)

            dist_norm = math.hypot(x - tx, y - ty)
            now = time.time()
            if dist_norm <= radius_now:
                if self.calib_hold_start is None:
                    self.calib_hold_start = now
                elapsed_ms = (now - self.calib_hold_start) * 1000.0
            else:
                self.calib_hold_start = None
                elapsed_ms = 0.0
            frac = max(0.0, min(1.0, elapsed_ms / float(max(1, APP_STATE.calib_dwell_ms))))

            # draw target/progress + hint
            if cv2 is not None:
                h, w = img.shape[:2]
                gx, gy = int(tx * w), int(ty * h)
                cv2.circle(img, (gx, gy), 18, (0, 170, 255), 2)
                end_angle = int(360 * frac)
                cv2.ellipse(img, (gx, gy), (22, 22), 0, 0, end_angle, (0, 170, 255), 3)
                cv2.putText(img, f"Calibration {min(self.calib_idx+1,len(self.calib_targets))}/{len(self.calib_targets)}",
                            (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                if self.engine.sess.last_feat is None:
                    cv2.putText(img, "No eye feature detected - adjust lighting/pose",
                                (24, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 255), 2)

            # ---- commit point (ต้องมีฟีเจอร์ก่อน) ----
            if frac >= 1.0 and 0 <= self.calib_idx < len(self.calib_targets):
                ok_added = False
                if self.engine.sess.last_feat is not None:
                    ok_added = self.engine.calibration_add_point(tx, ty)
                if ok_added:
                    self.calib_idx += 1
                    APP_STATE.calib_idx = self.calib_idx
                    self.calib_hold_start = None

                if self.calib_idx >= len(self.calib_targets):
                    rep = self.engine.calibration_finish()
                    APP_STATE.calib_overlay_active = False
                    self.calib_active = False
                    if rep.get('ok'):
                        tag = "PASS" if rep.get('passed') else "LOCK"
                        self._calib_banner_text = f"Calibration OK · RMSE={rep.get('rmse_px',0):.0f}px / CV={rep.get('rmse_cv_px',0):.0f}px · U={rep.get('uniformity',0):.2f} · {tag}"
                    else:
                        self._calib_banner_text = f"Calibration NG: {rep.get('msg','')}"
                    self._calib_banner_until = time.time() + 2.0
        else:
            self.calib_active = False
            self.calib_hold_start = None

        # banner after calibration done
        if time.time() < self._calib_banner_until and cv2 is not None and self._calib_banner_text:
            cv2.rectangle(img, (10, 10), (10+900, 10+40), (0,0,0), -1)
            cv2.putText(img, self._calib_banner_text, (18, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 255), 2)

        # ---- Click Gate ----
        gate_ok, reason = (False, "ยังไม่คาลิเบรท")
        if self.engine is not None:
            gate_ok, reason = self.engine.click_gate()
        APP_STATE.mouse_gate_passed = bool(gate_ok)
        APP_STATE.mouse_gate_reason = reason

        # Mouse: pointer follows; click only if gate_ok
        if self.mouse:
            self.mouse.set_enable(APP_STATE.mouse_enabled)
            self.mouse.dwell_ms = APP_STATE.dwell_ms
            self.mouse.dwell_radius_px = APP_STATE.dwell_radius
            self.mouse.update(x, y, do_click=(APP_STATE.mouse_enabled and gate_ok))

        # crosshair
        if cv2 is not None:
            h, w = img.shape[:2]
            gx, gy = int(x * w), int(y * h)
            cv2.circle(img, (gx, gy), 8, (0, 255, 0), 2)
            cv2.line(img, (gx-15, gy), (gx+15, gy), (0, 255, 0), 1)
            cv2.line(img, (gx, gy-15), (gx, gy+15), (0, 255, 0), 1)

        return VideoFrame.from_ndarray(img, format="bgr24")

# ----------------------------- UI: Sidebar -----------------------------
def _sidebar_controls():
    st.sidebar.title("⚙️ Controls")

    mode = st.sidebar.selectbox("Extractor Mode", ["Webcam/MediaPipe", "ESP32 (Orlosky)"], index=0)
    APP_STATE.extractor_mode = mode

    st.sidebar.subheader("🎛 Shaping")
    APP_STATE.gain  = float(st.sidebar.slider("Gain", 0.5, 2.5, APP_STATE.gain, 0.05))
    APP_STATE.gamma = float(st.sidebar.slider("Gamma", 0.5, 2.0, 1.0, 0.05))
    APP_STATE.deadzone = float(st.sidebar.slider("Deadzone", 0.0, 0.1, 0.02, 0.005))

    st.sidebar.subheader("🖱️ Mouse Control")
    APP_STATE.mouse_enabled = st.sidebar.toggle(
        "Enable Mouse Control (pointer follows; click unlocks after calibration PASS)",
        value=False,
        help="คลิกจะถูกปลดล็อกอัตโนมัติเมื่อ RMSE/CV/Uniformity ผ่านเกณฑ์"
    )
    APP_STATE.dwell_ms = st.sidebar.slider("Dwell Click (ms)", 200, 1500, 700, 50)
    APP_STATE.dwell_radius = st.sidebar.slider("Dwell Radius (px)", 10, 120, 40, 2)

    st.sidebar.subheader("🪞 Axes / Mirror")
    APP_STATE.mirror_input = st.sidebar.checkbox("Mirror webcam image", value=False)
    APP_STATE.invert_x     = st.sidebar.checkbox("Invert predicted X (x → 1−x)", value=False)
    APP_STATE.invert_y     = st.sidebar.checkbox("Invert predicted Y (y → 1−y)", value=False)

    if mode == "ESP32 (Orlosky)":
        st.sidebar.subheader("ESP32")
        APP_STATE.esp32_url = st.sidebar.text_input("Stream URL", value=APP_STATE.esp32_url)
        col_e1, col_e2, col_e3 = st.sidebar.columns(3)
        start_esp = col_e1.button("Start")
        stop_esp  = col_e2.button("Stop")
        APP_STATE.synthetic_hud = col_e3.toggle("HUD w/o Webcam", value=False)
        if start_esp:
            if APP_STATE.esp32_stop is None:
                APP_STATE.esp32_q = queue.Queue(maxsize=4)
                APP_STATE.esp32_stop = threading.Event()
                t = ESP32Reader(APP_STATE.esp32_url, APP_STATE.esp32_q, APP_STATE.esp32_stop)
                t.start(); st.toast("ESP32 reader started")
        if stop_esp and APP_STATE.esp32_stop is not None:
            APP_STATE.esp32_stop.set(); APP_STATE.esp32_stop = None
            st.toast("ESP32 reader stopped")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🪄 Smoothing (One Euro)")
    APP_STATE.filter_mincutoff = float(st.sidebar.slider("mincutoff", 0.05, 3.0, 1.0, 0.05))
    APP_STATE.filter_beta      = float(st.sidebar.slider("beta", 0.0, 1.0, 0.01, 0.01))

    st.sidebar.subheader("🧭 Calibration")
    npoints = st.sidebar.selectbox("Points", [9, 12], index=0)
    APP_STATE.calib_dwell_ms = st.sidebar.slider("Dwell per target (ms)", 400, 1500, APP_STATE.calib_dwell_ms, 50)

    if st.sidebar.button("Start Calibration"):
        if npoints == 9:
            # progressive: เริ่มกลาง → กลางแกน → มุม
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
        # ไม่ shuffle เพื่อให้ model ดีขึ้นเรื่อย ๆ แบบ progressive
        APP_STATE.calib_targets = grid
        APP_STATE.calib_idx = 0
        APP_STATE.calib_overlay_active = True
        sw, sh = _safe_size()
        APP_STATE.calib_radius_norm = max(0.012, 40 / max(sw, sh))  # ~40px
        st.toast("เริ่มคาลิเบรท (Fullscreen video)")

# ----------------------------- Overlay (CSS only for fullscreen video) -----------------------------
def _render_calibration_overlay(_ctx):
    """ทำให้วิดีโอเต็มหน้าจอขณะคาลิเบรท (ไม่รีเฟรชเพจ; การวาด/นับทำใน VideoProcessor)"""
    if APP_STATE.calib_overlay_active:
        st.session_state["calib_ph"].empty()
        st.markdown("""
        <style>
          section[data-testid="stSidebar"] { display: none !important; }
          header, footer { display: none !important; }
          video { position: fixed !important; inset: 0 !important;
                  width: 100vw !important; height: 100vh !important;
                  object-fit: cover !important; z-index: 9999 !important; }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.session_state["calib_ph"].empty()

# ----------------------------- Main -----------------------------
def main():
    st.set_page_config(page_title="Gaze All-in-One", page_icon="👀", layout="wide")
    st.title("👀 Gaze All-in-One — Webcam / ESP32 / Calibration Overlay (in-video)")
    st.caption("หากพรีวิวกล้องไม่ขึ้น ให้อนุญาตสิทธิ์กล้อง และติดตั้ง opencv-python / mediapipe")

    # sidebar & controls
    _sidebar_controls()

    # WebRTC (ขอ 60fps + 640x480)
    ctx = webrtc_streamer(
        key="gaze-stream",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 60, "min": 30},
            },
            "audio": False,
        },
        async_processing=False,
        video_processor_factory=GazeProcessor,
    )

    # fullscreen CSS while calibrating (no refresh)
    _render_calibration_overlay(ctx)

    # Diagnostics
    col1, col2, col3 = st.columns(3)
    mouse_state = "ON (click ✅)" if (APP_STATE.mouse_enabled and APP_STATE.mouse_gate_passed) \
                  else ("ON (click ⛔)" if APP_STATE.mouse_enabled else "OFF")
    with col1: st.metric("Mouse", mouse_state)
    with col2: st.metric("Mode", APP_STATE.extractor_mode)
    with col3: st.metric("ESP32", "RUNNING" if APP_STATE.esp32_stop is not None else "STOPPED")

    st.markdown("### Diagnostics")
    st.write({
        "calib_active": APP_STATE.calib_overlay_active,
        "calib_idx": APP_STATE.calib_idx,
        "n_targets": len(APP_STATE.calib_targets),
        "mirror_input": APP_STATE.mirror_input,
        "invert_x": APP_STATE.invert_x,
        "invert_y": APP_STATE.invert_y,
        "gaze": (round(APP_STATE.shared_gaze_x,3), round(APP_STATE.shared_gaze_y,3)),
        "click_gate": {"passed": APP_STATE.mouse_gate_passed, "reason": APP_STATE.mouse_gate_reason}
    })

    if ctx and ctx.video_processor and ctx.video_processor.engine:
        eng = ctx.video_processor.engine
        rep = eng.get_calibration_report()

        st.markdown("---")
        colA, colB, colC, colD = st.columns([1,1,1,2])
        with colA:
            if st.button("Start Drift"): eng.drift_start(); st.toast("Drift enabled")
        with colB:
            if st.button("Stop Drift"): eng.drift_stop(); st.toast("Drift disabled")
        with colC:
            if st.button("Add Drift Sample (current gaze ~ center)"): eng.drift_add(0.5, 0.5)
        with colD:
            st.write(eng.get_report())

        st.subheader("Calibration Report")
        if rep is None:
            st.info("Model: **Uncalibrated** – คลิกถูกล็อกจนกว่าจะคาลิเบรทผ่านเกณฑ์")
        else:
            ok, reason = eng.click_gate()
            chip = "✅ PASS" if ok else "⛔ LOCK"
            st.markdown(f"- {chip} · RMSE(train): **{rep.rmse_px:.0f}px**, RMSE(CV): **{rep.rmse_cv_px:.0f}px**, uniformity: **{rep.uniformity:.2f}**, points: **{rep.n_points}**")
            if not ok: st.caption(f"เหตุผล: {reason}")

        st.subheader("Profile")
        cS1, cS2 = st.columns([1,1])
        with cS1:
            if st.button("Save profile"):
                ok = eng.save_profile("gaze_profile.json", meta={"ts": time.time()})
                st.toast("Saved" if ok else "Save failed")
        with cS2:
            if st.button("Load profile"):
                ok = eng.load_profile("gaze_profile.json")
                st.toast("Loaded" if ok else "Load failed")

    st.markdown("""
**ขั้นตอนใช้งาน**
1) Sidebar → เลือก 9/12 จุด → **Start Calibration** (วิดีโอเต็มจอ)
2) จ้องจุดจนวงแหวนเต็ม → จะเลื่อนไปจุดถัดไปอัตโนมัติ (3 จุดแรกยอมรับกว้างขึ้น)
3) ครบทั้งหมด → แสดงผล PASS/LOCK บนวิดีโอ ~2 วินาที
4) เปิด "Enable Mouse Control" → เมาส์วิ่งตามตาทันที; คลิกได้เมื่อ PASS
    """)

if __name__ == "__main__":
    main()