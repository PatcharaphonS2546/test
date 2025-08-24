# test.py
# 👀 Gaze All-in-One — Single-file build
# - Webcam (MediaPipe) / ESP32-CAM
# - Full-screen Calibration Overlay (stable; no flicker)
# - Session-persistent state, mirror/invert axes, drift, save/load profile
# ต้องมี: streamlit>=1.26, streamlit-webrtc, streamlit-autorefresh, opencv-python(-headless),
#         mediapipe, av, numpy, (pyautogui ถ้าจะคลิก)

import math, time, threading, queue, random, json, pathlib, os
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

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None  # overlay ยังทำงานได้ แต่อาจไม่ smooth เท่าไร

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
    def __init__(self, enable: bool = False, dwell_ms: int = 700, dwell_radius_px: int = 40):
        self.sw, self.sh = _safe_size()
        self.enabled = enable
        self.dwell_ms = dwell_ms
        self.dwell_radius_px = dwell_radius_px
        self._last_in_time = None
        self._last_target = None

    def set_enable(self, v: bool): self.enabled = v

    def _do_click(self, x_px: int, y_px: int):
        if pyautogui is not None:
            try:
                pyautogui.moveTo(x_px, y_px)
                pyautogui.click()
            except Exception:
                pass

    def update(self, x_norm: float, y_norm: float, do_click: bool = True):
        if not self.enabled:
            self._last_in_time = None
            return
        x_px = int(x_norm * self.sw)
        y_px = int(y_norm * self.sh)
        tgt = (x_px, y_px)
        if self._last_target is None or math.hypot(tgt[0] - self._last_target[0], tgt[1] - self._last_target[1]) > self.dwell_radius_px:
            self._last_target = tgt
            self._last_in_time = time.time()
            return
        if do_click and self._last_in_time is not None:
            if (time.time() - self._last_in_time) * 1000.0 >= self.dwell_ms:
                self._do_click(x_px, y_px)
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

class SessionState:
    def __init__(self):
        self.smooth_x = OneEuroFilter(120, 1.0, 0.01, 1.0)
        self.smooth_y = OneEuroFilter(120, 1.0, 0.01, 1.0)
        self.calib_X: List[List[float]] = []
        self.calib_yx: List[float] = []
        self.calib_yy: List[float] = []
        self.model_ready = False
        self.model_pipeline = None  # (pipe_x, pipe_y) or (wx, wy)
        self.last_feat: Optional[np.ndarray] = None
        self.t_prev = None
        self.fps_hist: List[float] = []
        # Drift
        self.drift_enabled = False
        self.drift_dx = 0.0
        self.drift_dy = 0.0
        self.drift_alpha = 0.15

try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge
except Exception:
    Pipeline = PolynomialFeatures = Ridge = None

# ----------------------------- Feature Vector -----------------------------
def _feat_vec(feat: dict) -> np.ndarray:
    return np.array([
        feat.get('eye_cx_norm', 0.5),
        feat.get('eye_cy_norm', 0.5),
        feat.get('face_cx_norm', 0.5),
        feat.get('face_cy_norm', 0.5),
        1.0
    ], dtype=np.float32)

def _poly_expand(X: np.ndarray) -> np.ndarray:
    if X.ndim == 1:
        X = X[None, :]
    cols = [X]; n = X.shape[1]
    for i in range(n):
        for j in range(i, n):
            cols.append((X[:, i] * X[:, j])[:, None])
    return np.concatenate(cols, axis=1)

# ----------------------------- MediaPipe FeatureExtractor -----------------------------
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
                static_image_mode=False, refine_landmarks=True, max_num_faces=1,
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            )
        else:
            self.mesh = None

    def close(self):
        if self.mesh is not None:
            self.mesh.close()

    def _extract_mediapipe(self, frame_bgr) -> Optional[dict]:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(frame_rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0]
        pts = [(lm.landmark[i].x * w, lm.landmark[i].y * h) for i in range(len(lm.landmark))]

        def _avg(ids):
            xs = [pts[i][0] for i in ids if i < len(pts)]
            ys = [pts[i][1] for i in ids if i < len(pts)]
            if not xs or not ys: return None
            return (sum(xs)/len(xs), sum(ys)/len(ys))

        l_iris = _avg(self.LEFT_IRIS_IDS)
        r_iris = _avg(self.RIGHT_IRIS_IDS)
        if l_iris is None or r_iris is None:
            return None

        l_box_pts = [pts[i] for i in self.LEFT_EYE_IDS if i < len(pts)]
        r_box_pts = [pts[i] for i in self.RIGHT_EYE_IDS if i < len(pts)]

        def _norm_in_box(p, box_pts):
            if p is None or not box_pts:
                return (0.5, 0.5)
            xs = [q[0] for q in box_pts]; ys = [q[1] for q in box_pts]
            x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
            bw = max(8.0, x1 - x0); bh = max(6.0, y1 - y0)
            nx = (p[0] - x0) / bw; ny = (p[1] - y0) / bh
            return (min(1.0, max(0.0, nx)), min(1.0, max(0.0, ny)))

        l_nx, l_ny = _norm_in_box(l_iris, l_box_pts)
        r_nx, r_ny = _norm_in_box(r_iris, r_box_pts)
        eye_cx_norm = float((l_nx + r_nx) * 0.5)
        eye_cy_norm = float((l_ny + r_ny) * 0.5)

        face_ids = [1, 9, 152, 33, 263]
        fxs = [pts[i][0] for i in face_ids if i < len(pts)]
        fys = [pts[i][1] for i in face_ids if i < len(pts)]
        if not fxs or not fys:
            face_cx_norm, face_cy_norm = 0.5, 0.5
        else:
            face_cx_norm = float(sum(fxs)/len(fxs) / max(1, w))
            face_cy_norm = float(sum(fys)/len(fys) / max(1, h))

        # simple blink proxy
        def _eye_h(idx): ys = [pts[i][1] for i in idx if i < len(pts)]; return (max(ys)-min(ys)) if ys else 0.0
        def _eye_w(idx): xs = [pts[i][0] for i in idx if i < len(pts)]; return (max(xs)-min(xs)) if xs else 1.0
        l_open = (_eye_h(self.LEFT_EYE_IDS) / max(6.0, _eye_w(self.LEFT_EYE_IDS))) > 0.28
        r_open = (_eye_h(self.RIGHT_EYE_IDS) / max(6.0, _eye_w(self.RIGHT_EYE_IDS))) > 0.28
        eye_open = bool(l_open and r_open)

        return {
            "eye_cx_norm": eye_cx_norm, "eye_cy_norm": eye_cy_norm,
            "face_cx_norm": face_cx_norm, "face_cy_norm": face_cy_norm,
            "eye_open": eye_open
        }

    def extract(self, frame_bgr) -> Optional[dict]:
        if frame_bgr is None: return None
        if self.use_mediapipe and self.mesh is not None:
            try: return self._extract_mediapipe(frame_bgr)
            except Exception: return None
        return {"eye_cx_norm":0.5,"eye_cy_norm":0.5,"face_cx_norm":0.5,"face_cy_norm":0.5,"eye_open":True}

# ----------------------------- ESP32 Reader -----------------------------
def _crop_to_aspect_ratio(image, width=640, height=480):
    h, w = image.shape[:2]; desired = width/height; cur = w/h
    if cur > desired:
        new_w = int(desired * h); off = (w - new_w) // 2; image = image[:, off:off+new_w]
    else:
        new_h = int(w/desired); off = (h - new_h)//2; image = image[off:off+new_h, :]
    return cv2.resize(image, (width, height))

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
class GazeEngine:
    def __init__(self, cfg: GazeConfig, extractor: FeatureExtractor):
        self.cfg = cfg; self.ext = extractor; self.sess = SessionState()

    def update_config(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.cfg, k): setattr(self.cfg, k, v)

    # Calibration
    def calibration_add_point(self, screen_x_norm: float, screen_y_norm: float):
        if self.sess.last_feat is None: return False
        x = self.sess.last_feat.astype(np.float32)
        self.sess.calib_X.append(x.tolist())
        self.sess.calib_yx.append(float(screen_x_norm))
        self.sess.calib_yy.append(float(screen_y_norm))
        return True

    def calibration_finish(self):
        if len(self.sess.calib_X) < 6:
            self.sess.model_ready = False; self.sess.model_pipeline = None
            return {'ok': False, 'msg': 'ต้องเก็บคาลิเบรทอย่างน้อย 6 จุด'}
        X = np.array(self.sess.calib_X, dtype=np.float32)
        yx = np.array(self.sess.calib_yx, dtype=np.float32)
        yy = np.array(self.sess.calib_yy, dtype=np.float32)
        if Pipeline and PolynomialFeatures and Ridge:
            pipe = Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=True)), ('reg', Ridge(alpha=1.0))])
            pipe_x = pipe.fit(X, yx); pipe_y = pipe.fit(X, yy)
            self.sess.model_pipeline = (pipe_x, pipe_y)
        else:
            Phi = _poly_expand(X); lam = 1e-3
            A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
            bx = Phi.T @ yx; by = Phi.T @ yy
            wx = np.linalg.solve(A, bx); wy = np.linalg.solve(A, by)
            self.sess.model_pipeline = (wx, wy)
        self.sess.model_ready = True
        preds = [self._map_from_feat(np.array(v, dtype=np.float32)) for v in self.sess.calib_X]
        preds = np.array(preds); y_true = np.stack([yx, yy], axis=1)
        rmse = float(np.sqrt(np.mean((preds - y_true) ** 2)))
        return {'ok': True, 'rmse_norm': rmse, 'n': len(self.sess.calib_X)}

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
        if self.sess.model_ready and self.sess.model_pipeline is not None:
            if Pipeline and isinstance(self.sess.model_pipeline[0], Pipeline):
                pipe_x, pipe_y = self.sess.model_pipeline
                x = float(pipe_x.predict(feat_vec[None, :])[0]); y = float(pipe_y.predict(feat_vec[None, :])[0])
                return np.array([x, y], dtype=np.float32)
            else:
                wx, wy = self.sess.model_pipeline
                Phi = _poly_expand(feat_vec)
                x = float(Phi @ wx); y = float(Phi @ wy)
                return np.array([x, y], dtype=np.float32)
        ex, ey = float(feat_vec[0]), float(feat_vec[1])
        kx, ky = self.cfg.fallback_kx, self.cfg.fallback_ky
        return np.array([0.5 + (ex - 0.5) * kx, 0.5 + (ey - 0.5) * ky], dtype=np.float32)

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
        else:
            if (not feat.get("eye_open", True)) and self.sess.last_feat is not None:
                fv = self.sess.last_feat
            else:
                fv = _feat_vec(feat); self.sess.last_feat = fv
            pred = self._map_from_feat(fv)
        x, y = self._shape_and_smooth(float(pred[0]), float(pred[1]))
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
        x, y = self._shape_and_smooth(float(pred[0]), float(pred[1]))
        return x, y

    def get_report(self):
        rmse = None
        if self.sess.model_ready and len(self.sess.calib_X) >= 6:
            X = np.array(self.sess.calib_X, dtype=np.float32)
            yx = np.array(self.sess.calib_yx, dtype=np.float32)
            yy = np.array(self.sess.calib_yy, dtype=np.float32)
            preds = [self._map_from_feat(np.array(v, dtype=np.float32)) for v in self.sess.calib_X]
            preds = np.array(preds); y_true = np.stack([yx, yy], axis=1)
            rmse = float(np.sqrt(np.mean((preds - y_true) ** 2)))
        return {'model_ready': self.sess.model_ready, 'rmse_norm': rmse, 'n_samples': len(self.sess.calib_X)}

# ----------------------------- App State (persist across reruns) -----------------------------
class AppState:
    def __init__(self):
        # Modes / IO
        self.extractor_mode = 'Webcam/MediaPipe'
        self.mouse_enabled = False
        self.dwell_ms = 700
        self.dwell_radius = 40
        self.esp32_url = 'http://esp32.local/stream'
        self.esp32_q: Optional[queue.Queue] = None
        self.esp32_stop: Optional[threading.Event] = None
        self.synthetic_hud = False
        # Axes & mirror
        self.mirror_input = False
        self.invert_x = False
        self.invert_y = False
        # Shared gaze (from processor)
        self.shared_gaze_x = 0.5
        self.shared_gaze_y = 0.5
        # Calibration overlay (main-thread only)
        self.calib_overlay_active = False
        self.calib_targets: List[Tuple[float, float]] = []
        self.calib_idx = 0
        self.calib_dwell_ms = 800
        self.calib_radius_norm = 0.02  # ~2% of screen

# persist
if 'APP_STATE' not in st.session_state:
    st.session_state['APP_STATE'] = AppState()
APP_STATE: AppState = st.session_state['APP_STATE']
# dwell hold timer per-target
st.session_state.setdefault("calib_hold_start", None)
# overlay placeholder
if "calib_ph" not in st.session_state:
    st.session_state["calib_ph"] = st.empty()

# ----------------------------- Streamlit WebRTC Processor -----------------------------
class GazeProcessor(VideoProcessorBase):
    def __init__(self):
        self.engine: Optional[GazeEngine] = None
        self.mouse: Optional[MouseController] = None
        self.last_gaze = (0.5, 0.5)

    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # mirror webcam image before extract (fix X reversed)
        if APP_STATE.extractor_mode == 'Webcam/MediaPipe' and APP_STATE.mirror_input and cv2 is not None:
            img = cv2.flip(img, 1)

        if APP_STATE.extractor_mode == 'ESP32 (Orlosky)' and getattr(APP_STATE, 'synthetic_hud', False):
            img = np.full((480, 640, 3), 40, dtype=np.uint8)

        if self.engine is None:
            extractor = FeatureExtractor(use_mediapipe=(APP_STATE.extractor_mode == 'Webcam/MediaPipe'))
            cfg = GazeConfig()
            self.engine = GazeEngine(cfg, extractor)
            self.mouse = MouseController(enable=APP_STATE.mouse_enabled,
                                         dwell_ms=APP_STATE.dwell_ms,
                                         dwell_radius_px=APP_STATE.dwell_radius)

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

        if APP_STATE.invert_x: x = 1.0 - x
        if APP_STATE.invert_y: y = 1.0 - y
        self.last_gaze = (x, y)

        # share gaze to main thread (overlay)
        APP_STATE.shared_gaze_x = float(x)
        APP_STATE.shared_gaze_y = float(y)

        # mouse (optional)
        if self.mouse:
            self.mouse.set_enable(APP_STATE.mouse_enabled)
            self.mouse.dwell_ms = APP_STATE.dwell_ms
            self.mouse.dwell_radius_px = APP_STATE.dwell_radius
            self.mouse.update(x, y, do_click=True)

        # draw simple crosshair for tracking preview (not calibration)
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
    gain = st.sidebar.slider("Gain", 0.5, 2.5, 1.0, 0.05)
    gamma = st.sidebar.slider("Gamma", 0.5, 2.0, 1.0, 0.05)
    dead = st.sidebar.slider("Deadzone", 0.0, 0.1, 0.02, 0.005)

    st.sidebar.subheader("🖱️ Mouse Control")
    APP_STATE.mouse_enabled = st.sidebar.toggle("Enable Mouse Control", value=False)
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
        stop_esp = col_e2.button("Stop")
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
    st.sidebar.subheader("🧭 Calibration")
    npoints = st.sidebar.selectbox("Points", [9, 12], index=0)
    APP_STATE.calib_dwell_ms = st.sidebar.slider("Dwell per target (ms)", 400, 1500, 800, 50)

    if st.sidebar.button("Start Calibration"):
        if npoints == 9:
            grid = [(0.15,0.15),(0.5,0.15),(0.85,0.15),
                    (0.15,0.5),(0.5,0.5),(0.85,0.5),
                    (0.15,0.85),(0.5,0.85),(0.85,0.85)]
        else:
            grid = [(0.1,0.1),(0.5,0.1),(0.9,0.1),
                    (0.1,0.5),(0.5,0.5),(0.9,0.5),
                    (0.1,0.9),(0.5,0.9),(0.9,0.9),
                    (0.3,0.3),(0.7,0.3),(0.5,0.7)]
        random.shuffle(grid)
        APP_STATE.calib_targets = grid
        APP_STATE.calib_idx = 0
        APP_STATE.calib_overlay_active = True
        # radius_norm ≈ 40px on current screen
        sw, sh = _safe_size()
        APP_STATE.calib_radius_norm = max(0.012, 40 / max(sw, sh))
        st.session_state["calib_hold_start"] = None
        st.toast("เริ่มคาลิเบรท (Overlay)")

    # patch engine shaping in processor on next recv (simple global config)
    st.session_state.setdefault('cfg_patch', {'gain':gain, 'gamma':gamma, 'deadzone':dead})
    st.session_state['cfg_patch'] = {'gain':gain, 'gamma':gamma, 'deadzone':dead}

# ----------------------------- Overlay (Calibration in main thread) -----------------------------
def _render_calibration_overlay(ctx):
    """Full-screen overlay; add points based on normalized gaze from processor."""
    if not APP_STATE.calib_overlay_active:
        st.session_state["calib_ph"].empty()
        return

    # ไม่ต้อง autorefresh ขณะ overlay calibration

    # engine ready?
    eng = None
    if ctx and ctx.video_processor and ctx.video_processor.engine:
        eng = ctx.video_processor.engine
        # apply shaping tweaks from sidebar
        eng.update_config(
            gain=st.session_state['cfg_patch']['gain'],
            gamma=st.session_state['cfg_patch']['gamma'],
            deadzone=st.session_state['cfg_patch']['deadzone'],
        )

    targets = APP_STATE.calib_targets
    idx = APP_STATE.calib_idx
    dwell_ms = APP_STATE.calib_dwell_ms
    rnorm = APP_STATE.calib_radius_norm

    tx, ty = (0.5, 0.5)
    if 0 <= idx < len(targets):
        tx, ty = targets[idx]

    gx = APP_STATE.shared_gaze_x
    gy = APP_STATE.shared_gaze_y

    dist_norm = math.hypot(gx - tx, gy - ty)
    now = time.time(); started = st.session_state["calib_hold_start"]
    if dist_norm <= rnorm:
        if started is None: st.session_state["calib_hold_start"] = now; elapsed_ms = 0.0
        else: elapsed_ms = (now - started) * 1000.0
    else:
        st.session_state["calib_hold_start"] = None
        elapsed_ms = 0.0
    frac = max(0.0, min(1.0, elapsed_ms / float(max(1, dwell_ms))))

    # commit point if finished
    if frac >= 1.0 and eng is not None and 0 <= idx < len(targets):
        eng.calibration_add_point(tx, ty)
        APP_STATE.calib_idx += 1
        st.session_state["calib_hold_start"] = None
        if APP_STATE.calib_idx >= len(APP_STATE.calib_targets):
            rep = eng.calibration_finish()
            APP_STATE.calib_overlay_active = False
            st.toast(f"Calibration done: {'OK' if rep.get('ok') else 'NG'} | RMSE={rep.get('rmse_norm',None)}")
            st.session_state["calib_ph"].empty()
            return

    # draw overlay via placeholder (no flicker)
    deg = int(360 * frac); size = 32  # px
    with st.session_state["calib_ph"]:
        st.markdown(f"""
        <style>
            .calib-overlay {{
                position: fixed; inset: 0; z-index: 9999; background: rgba(0,0,0,0.80);
                display: flex; align-items: center; justify-content: center; color: white; text-align: center;
            }}
            .calib-target {{
                position: fixed; left: {tx*100:.3f}vw; top: {ty*100:.3f}vh; transform: translate(-50%, -50%);
                width: {size}px; height: {size}px; border-radius: 50%;
                border: 2px solid rgb(255,170,0);
                background: conic-gradient(rgba(255,170,0,1) {deg}deg, rgba(0,0,0,0) 0deg);
                box-shadow: 0 0 0 3px rgba(0,0,0,0.3) inset, 0 0 8px rgba(255,170,0,0.6);
            }}
            .calib-top {{ position: fixed; left:50%; top: 24px; transform: translateX(-50%); font-size: 20px; }}
            .calib-info {{ position: fixed; left:50%; bottom: 40px; transform: translateX(-50%); opacity:.9; }}
            .calib-close {{
                position: fixed; right:24px; top:24px; background:#0ea5e9; color:#fff;
                border-radius:8px; padding:8px 12px; text-decoration:none;
            }}
            .calib-bar {{
                position: fixed; left:50%; bottom: 20px; transform: translateX(-50%);
                width: 50vw; height: 10px; border: 1px solid #888;
                background: linear-gradient(90deg, rgba(255,170,0,1) {frac*100:.1f}%, rgba(0,0,0,0) 0%);
            }}
        </style>
        <div class="calib-overlay">
            <div class="calib-top">Calibration {min(idx+1, len(targets))}/{len(targets)}</div>
            <a class="calib-close" href="?close_overlay=1">ปิด Overlay</a>
            <div class="calib-target"></div>
            <div class="calib-info">จ้องเป้าหมายให้อยู่ในวงยอมรับ ~{int(rnorm*100)}% ของจอ จนวงแหวนเต็ม</div>
            <div class="calib-bar"></div>
            <form>
                <button type="submit" style="margin-top:40px;font-size:18px;padding:8px 16px;">เริ่ม Calibration</button>
            </form>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------- Main -----------------------------
def main():
    st.set_page_config(page_title="Gaze All-in-One", page_icon="👀", layout="wide")
    st.title("👀 Gaze All-in-One — Webcam / ESP32 / Calibration Overlay")
    st.caption("ถ้าพรีวิวกล้องดำ ตรวจสิทธิ์กล้อง & ติดตั้ง opencv-python / mediapipe")

    # Close overlay via query param
    try:
        qp = st.query_params
        if "close_overlay" in qp:
            APP_STATE.calib_overlay_active = False
            st.session_state["calib_hold_start"] = None
            qp.clear()  # remove param
            st.session_state["calib_ph"].empty()
    except Exception:
        pass

    _sidebar_controls()

    ctx = webrtc_streamer(
        key="gaze-stream",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False,
        video_processor_factory=GazeProcessor,
    )

    # Render calibration overlay (drives calibration; independent of video drawing)
    _render_calibration_overlay(ctx)

    # Diagnostics & controls
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Mouse", "ON" if APP_STATE.mouse_enabled else "OFF")
    with col2: st.metric("Mode", APP_STATE.extractor_mode)
    with col3: st.metric("ESP32", "RUNNING" if APP_STATE.esp32_stop is not None else "STOPPED")

    st.markdown("### Diagnostics")
    st.write({
        "overlay_active": APP_STATE.calib_overlay_active,
        "calib_idx": APP_STATE.calib_idx,
        "n_targets": len(APP_STATE.calib_targets),
        "mirror_input": APP_STATE.mirror_input,
        "invert_x": APP_STATE.invert_x,
        "invert_y": APP_STATE.invert_y,
        "gaze": (round(APP_STATE.shared_gaze_x,3), round(APP_STATE.shared_gaze_y,3)),
    })

    if ctx and ctx.video_processor and ctx.video_processor.engine:
        eng = ctx.video_processor.engine
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
**วิธีคาลิเบรท**
1) เลือกจำนวนจุด (9/12) → **Start Calibration**
2) จ้องเป้าแต่ละจุดจนวงแหวนเต็ม ระบบจะเก็บอัตโนมัติ
3) ครบทุกจุด ระบบฟิตโมเดลทันที → ใช้ tracking ต่อได้เลย
- แกน X วิ่งกลับทิศ? ติ๊ก **Invert X** หรือ **Mirror webcam image** แล้วคาลิเบรทใหม่
- ใช้ Drift เพื่อลด bias คงที่หลังคาลิเบรท
    """)

if __name__ == "__main__":
    main()