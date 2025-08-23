"""
Gaze All-in-One (ONEFILE) — Webcam/MediaPipe + ESP32 (Orlosky) + Calibration + Drift + Mouse Control + Streamlit UI

How to run:
  pip install streamlit streamlit-webrtc av opencv-python-headless numpy
  pip install mediapipe scikit-learn pynput pyautogui screeninfo
  # optional (Windows sometimes needs full opencv): pip install opencv-python

Run UI:
  streamlit run gaze_all_in_one.py

Notes:
- macOS: grant Accessibility permission to the terminal/IDE to allow mouse control
- If MediaPipe/Sklearn not available, the app falls back to simple mapping & numpy fitting
- ESP32 mode expects a stream that yields normalized (nx, ny) in [0..1]. See `iter_norm_from_esp32()` stub for guidance.
"""
from __future__ import annotations
import os, sys, time, math, threading, queue, random, json
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Deque
from collections import deque

import numpy as np

# --- Optional deps (graceful fallbacks) ---
try:
    import cv2
except Exception:
    cv2 = None  # UI still runs, but webcam won't work

try:
    import mediapipe as mp
except Exception:
    mp = None

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
except Exception:
    Ridge = None
    PolynomialFeatures = None
    Pipeline = None

# Mouse control backends
_MOUSE_BACKEND = None
try:
    from pynput.mouse import Controller as PynputMouseController, Button as PynputButton
    _MOUSE_BACKEND = 'pynput'
except Exception:
    try:
        import pyautogui as _pyautogui
        _MOUSE_BACKEND = 'pyautogui'
    except Exception:
        _MOUSE_BACKEND = None

# Screen size helper
try:
    from screeninfo import get_monitors
except Exception:
    get_monitors = None

# Streamlit UI
try:
    import streamlit as st
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
    from av import VideoFrame
except Exception as e:
    raise RuntimeError("This app requires streamlit + streamlit-webrtc + av. Install per header instructions.")


# ============================
# Utility & Filters
# ============================
class OneEuro:
    """Simple One Euro Filter for smoothing gaze; low-latency smoothing.
    Based on https://cristal.univ-lille.fr/~casiez/1euro/
    """
    def __init__(self, freq=60.0, mincutoff=2.0, beta=0.02, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def alpha(self, cutoff):
        te = 1.0 / max(1e-6, self.freq)
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x, t):
        if self.t_prev is None:
            self.t_prev = t
        dt = max(1e-6, t - self.t_prev)
        self.freq = 1.0 / dt
        self.t_prev = t

        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            return x
        # derivative of signal
        dx = (x - self.x_prev) * self.freq
        a_d = self.alpha(self.dcutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self.alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


# ============================
# Feature Extraction (MediaPipe or fallback)
# ============================
class FeatureExtractor:
    def __init__(self, use_mediapipe: bool = True):
        self.use_mediapipe = use_mediapipe and (mp is not None)
        if self.use_mediapipe:
            self.mp_face = mp.solutions.face_mesh
            self.mesh = self.mp_face.FaceMesh(static_image_mode=False,
                                              refine_landmarks=True,
                                              max_num_faces=1,
                                              min_detection_confidence=0.5,
                                              min_tracking_confidence=0.5)
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
        # Simple heuristic features: average of key eye landmarks + face center
        # Indices from MediaPipe FaceMesh for eye region (approx)
        left_ids = [33, 133, 159, 145]   # outer/inner corners + top/bottom lids approx
        right_ids = [362, 263, 386, 374]
        pts = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]
        def avg_xy(ids):
            xs = [pts[i][0] for i in ids]
            ys = [pts[i][1] for i in ids]
            return (sum(xs)/len(xs), sum(ys)/len(ys))
        lx, ly = avg_xy(left_ids)
        rx, ry = avg_xy(right_ids)
        cx = (lx + rx) / 2.0
        cy = (ly + ry) / 2.0
        # Normalize to [0,1] by frame size
        nx = cx / max(1, w)
        ny = cy / max(1, h)
        # rudimentary size/tilt features
        face_ids = [1, 9, 152, 33, 263]
        fx = sum([pts[i][0] for i in face_ids]) / len(face_ids) / max(1, w)
        fy = sum([pts[i][1] for i in face_ids]) / len(face_ids) / max(1, h)
        feat = {
            'eye_cx_norm': float(nx),
            'eye_cy_norm': float(ny),
            'face_cx_norm': float(fx),
            'face_cy_norm': float(fy),
        }
        return feat

    def extract(self, frame_bgr) -> Optional[dict]:
        if frame_bgr is None:
            return None
        if self.use_mediapipe and self.mesh is not None and cv2 is not None:
            try:
                return self._extract_mediapipe(frame_bgr)
            except Exception:
                return None
        # Fallback: center of frame (poor but usable before calibration)
        h, w = frame_bgr.shape[:2]
        return {'eye_cx_norm': 0.5, 'eye_cy_norm': 0.5, 'face_cx_norm': 0.5, 'face_cy_norm': 0.5}


# ============================
# Calibration & Mapping
# ============================
@dataclass
class GazeConfig:
    gain: float = 1.0
    gain_x: float = 1.0
    gain_y: float = 1.0
    gamma: float = 1.0
    deadzone: float = 0.02
    clamp_jump: float = 0.15
    smooth_fc: float = 2.0
    smooth_beta: float = 0.02
    fallback_kx: float = 1.0
    fallback_ky: float = 1.0
    screen_w: int = 1920
    screen_h: int = 1080

@dataclass
class SessionState:
    model_ready: bool = False
    model_pipeline: Optional[object] = None  # sklearn Pipeline or numpy coefs
    last_feat: Optional[np.ndarray] = None
    smooth_x: OneEuro = field(default_factory=lambda: OneEuro())
    smooth_y: OneEuro = field(default_factory=lambda: OneEuro())
    drift_dx: float = 0.0
    drift_dy: float = 0.0
    drift_alpha: float = 0.05  # EMA rate for drift correction
    drift_enabled: bool = False
    calib_X: List[List[float]] = field(default_factory=list)
    calib_yx: List[float] = field(default_factory=list)
    calib_yy: List[float] = field(default_factory=list)
    fps_hist: Deque[float] = field(default_factory=lambda: deque(maxlen=120))
    t_prev: Optional[float] = None


def _feat_vec(feat: dict) -> np.ndarray:
    return np.array([
        feat.get('eye_cx_norm', 0.5),
        feat.get('eye_cy_norm', 0.5),
        feat.get('face_cx_norm', 0.5),
        feat.get('face_cy_norm', 0.5),
        1.0,
    ], dtype=np.float32)


def _poly_expand(X: np.ndarray) -> np.ndarray:
    """Quadratic expansion (manual) if sklearn missing."""
    if X.ndim == 1:
        X = X[None, :]
    cols = [X]
    # pairwise products
    n = X.shape[1]
    for i in range(n):
        for j in range(i, n):
            cols.append((X[:, i] * X[:, j])[:, None])
    return np.concatenate(cols, axis=1)


class GazeEngine:
    def __init__(self, cfg: GazeConfig, extractor: FeatureExtractor):
        self.cfg = cfg
        self.ext = extractor
        self.sess = SessionState()

    def update_config(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)

    # -------- Calibration --------
    def calibration_add_point(self, screen_x_norm: float, screen_y_norm: float):
        if self.sess.last_feat is None:
            return False
        x = self.sess.last_feat.astype(np.float32)
        self.sess.calib_X.append(x.tolist())
        self.sess.calib_yx.append(float(screen_x_norm))
        self.sess.calib_yy.append(float(screen_y_norm))
        return True

    def calibration_finish(self):
        if len(self.sess.calib_X) < 6:
            self.sess.model_ready = False
            self.sess.model_pipeline = None
            return {'ok': False, 'msg': 'Not enough calibration samples (>=6).'}
        X = np.array(self.sess.calib_X, dtype=np.float32)
        yx = np.array(self.sess.calib_yx, dtype=np.float32)
        yy = np.array(self.sess.calib_yy, dtype=np.float32)
        if Pipeline and PolynomialFeatures and Ridge:
            pipe = Pipeline([
                ('poly', PolynomialFeatures(degree=2, include_bias=True)),
                ('reg', Ridge(alpha=1.0))
            ])
            pipe_x = pipe.fit(X, yx)
            pipe_y = pipe.fit(X, yy)
            self.sess.model_pipeline = (pipe_x, pipe_y)
        else:
            # manual quadratic expansion + ridge (lambda tiny)
            Phi = _poly_expand(X)
            lam = 1e-3
            A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
            bx = Phi.T @ yx
            by = Phi.T @ yy
            wx = np.linalg.solve(A, bx)
            wy = np.linalg.solve(A, by)
            self.sess.model_pipeline = (wx, wy)  # store weights
        self.sess.model_ready = True
        # compute rmse on calib
        preds = [self._map_from_feat(np.array(v, dtype=np.float32)) for v in self.sess.calib_X]
        preds = np.array(preds)
        y_true = np.stack([yx, yy], axis=1)
        rmse = float(np.sqrt(np.mean((preds - y_true)**2)))
        return {'ok': True, 'rmse_norm': rmse, 'n': len(self.sess.calib_X)}

    # -------- Drift --------
    def drift_start(self):
        self.sess.drift_enabled = True

    def drift_stop(self):
        self.sess.drift_enabled = False

    def drift_add(self, screen_x_norm: float, screen_y_norm: float):
        if self.sess.last_feat is None:
            return False
        pred = self._map_from_feat(self.sess.last_feat)
        ex = screen_x_norm - pred[0]
        ey = screen_y_norm - pred[1]
        a = self.sess.drift_alpha
        self.sess.drift_dx = (1 - a) * self.sess.drift_dx + a * ex
        self.sess.drift_dy = (1 - a) * self.sess.drift_dy + a * ey
        return True

    def drift_status(self):
        return {'enabled': self.sess.drift_enabled, 'dx': self.sess.drift_dx, 'dy': self.sess.drift_dy}

    # -------- Processing --------
    def _map_from_feat(self, feat_vec: np.ndarray) -> np.ndarray:
        if self.sess.model_ready and self.sess.model_pipeline is not None:
            if isinstance(self.sess.model_pipeline[0], Pipeline):
                pipe_x, pipe_y = self.sess.model_pipeline
                x = pipe_x.predict(feat_vec[None, :])[0]
                y = pipe_y.predict(feat_vec[None, :])[0]
                return np.array([x, y], dtype=np.float32)
            else:
                # numpy weights
                wx, wy = self.sess.model_pipeline
                Phi = _poly_expand(feat_vec)
                x = float(Phi @ wx)
                y = float(Phi @ wy)
                return np.array([x, y], dtype=np.float32)
        # Fallback: map eye center directly
        ex, ey = float(feat_vec[0]), float(feat_vec[1])
        kx, ky = self.cfg.fallback_kx, self.cfg.fallback_ky
        return np.array([0.5 + (ex - 0.5) * kx, 0.5 + (ey - 0.5) * ky], dtype=np.float32)

    def _shape_and_smooth(self, x: float, y: float) -> Tuple[float, float]:
        # Drift correction
        if self.sess.drift_enabled:
            x += self.sess.drift_dx
            y += self.sess.drift_dy
        # Gain & gamma
        gx = self.cfg.gain * self.cfg.gain_x
        gy = self.cfg.gain * self.cfg.gain_y
        x = 0.5 + (x - 0.5) * gx
        y = 0.5 + (y - 0.5) * gy
        if self.cfg.gamma != 1.0:
            def _gamma(v):
                # remap around center 0.5
                s = (v - 0.5)
                sign = 1.0 if s >= 0 else -1.0
                return 0.5 + sign * (abs(s) ** self.cfg.gamma)
            x = _gamma(x)
            y = _gamma(y)
        # Deadzone around center
        dz = self.cfg.deadzone
        def _dead(v):
            d = v - 0.5
            if abs(d) < dz:
                return 0.5
            return v
        x = _dead(x); y = _dead(y)
        # Clamp
        x = min(1.0, max(0.0, x))
        y = min(1.0, max(0.0, y))
        # OneEuro smooth
        t = time.time()
        x = float(self.sess.smooth_x.filter(x, t))
        y = float(self.sess.smooth_y.filter(y, t))
        return x, y

    def process_frame(self, frame_bgr) -> Tuple[float, float, dict, np.ndarray]:
        t0 = time.time()
        feat = self.ext.extract(frame_bgr)
        if feat is None:
            pred = np.array([0.5, 0.5], dtype=np.float32)
        else:
            fv = _feat_vec(feat)
            self.sess.last_feat = fv
            pred = self._map_from_feat(fv)
        x, y = self._shape_and_smooth(float(pred[0]), float(pred[1]))
        # FPS/latency
        t1 = time.time()
        if self.sess.t_prev is not None:
            dt = t1 - self.sess.t_prev
            if dt > 0:
                self.sess.fps_hist.append(1.0 / dt)
        self.sess.t_prev = t1
        metrics = {
            'fps': float(np.mean(self.sess.fps_hist)) if len(self.sess.fps_hist) else 0.0,
            'latency_ms': float((t1 - t0) * 1000.0),
            'model': 'calibrated' if self.sess.model_ready else 'fallback'
        }
        return x, y, metrics, frame_bgr

    def process_external(self, nx: float, ny: float) -> Tuple[float, float]:
        # Provide external normalized pupil center as features
        feat = {'eye_cx_norm': float(nx), 'eye_cy_norm': float(ny), 'face_cx_norm': 0.5, 'face_cy_norm': 0.5}
        fv = _feat_vec(feat)
        self.sess.last_feat = fv
        pred = self._map_from_feat(fv)
        x, y = self._shape_and_smooth(float(pred[0]), float(pred[1]))
        return x, y

    def get_report(self) -> dict:
        return {
            'model_ready': self.sess.model_ready,
            'drift': self.drift_status(),
            'fps_avg': float(np.mean(self.sess.fps_hist)) if len(self.sess.fps_hist) else 0.0,
        }


# ============================
# Mouse Controller
# ============================
class MouseController:
    def __init__(self, enable: bool = False, dwell_ms: int = 700, dwell_radius_px: int = 40):
        self.enable = enable
        self.dwell_ms = dwell_ms
        self.dwell_radius_px = dwell_radius_px
        self.last_px = None
        self.last_t = None
        self.backend = _MOUSE_BACKEND
        # screen size
        self.sw, self.sh = self._get_screen_size()
        if self.backend == 'pynput':
            self.mouse = PynputMouseController()
        elif self.backend == 'pyautogui':
            self.mouse = _pyautogui
        else:
            self.mouse = None

    def _get_screen_size(self):
        if get_monitors:
            try:
                m = get_monitors()[0]
                return int(m.width), int(m.height)
            except Exception:
                pass
        # fallback
        return 1920, 1080

    def set_enable(self, v: bool):
        self.enable = v

    def update(self, x_norm: float, y_norm: float, do_click=True):
        if not self.enable or self.mouse is None:
            return
        x = int(min(max(x_norm, 0.0), 1.0) * (self.sw - 1))
        y = int(min(max(y_norm, 0.0), 1.0) * (self.sh - 1))
        # Move
        if self.backend == 'pynput':
            self.mouse.position = (x, y)
        elif self.backend == 'pyautogui':
            self.mouse.moveTo(x, y)
        # Dwell click
        if not do_click:
            self.last_px = (x, y)
            self.last_t = time.time()
            return
        now = time.time()
        if self.last_px is None:
            self.last_px = (x, y); self.last_t = now; return
        dx = x - self.last_px[0]; dy = y - self.last_px[1]
        dist = math.hypot(dx, dy)
        if dist <= self.dwell_radius_px:
            if (now - self.last_t) * 1000.0 >= self.dwell_ms:
                self._click()
                self.last_t = now  # reset after click
        else:
            self.last_px = (x, y); self.last_t = now

    def _click(self):
        try:
            if self.backend == 'pynput':
                self.mouse.click(PynputButton.left, 1)
            elif self.backend == 'pyautogui':
                self.mouse.click()
        except Exception:
            pass


# ============================
# ESP32 Reader (stub)
# ============================
class ESP32Reader(threading.Thread):
    """Example reader. Replace body with real ESP32 CAM parsing.
    For demo: it yields random (nx, ny) near last value to simulate movement.
    """
    def __init__(self, url: str, out_q: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.url = url
        self.out_q = out_q
        self.stop_event = stop_event
        self._nx, self._ny = 0.5, 0.5

    def run(self):
        while not self.stop_event.is_set():
            # TODO: replace with actual stream parsing, e.g. MJPEG or WebSocket JSON
            self._nx = min(1.0, max(0.0, self._nx + np.random.randn() * 0.01))
            self._ny = min(1.0, max(0.0, self._ny + np.random.randn() * 0.01))
            try:
                self.out_q.put((self._nx, self._ny), timeout=0.01)
            except Exception:
                pass
            time.sleep(0.016)  # ~60Hz


# ============================
# Global App State (for calibration inside transformer)
# ============================
class AppState:
    def __init__(self):
        self.calib_active = False
        self.calib_targets: List[Tuple[float, float]] = []
        self.calib_idx = 0
        self.calib_dwell_ms = 800
        self.calib_radius = 40  # px for hit test in overlay space
        self.calib_started_at = 0.0
        self.calib_progress_ms = 0.0
        self.extractor_mode = 'Webcam/MediaPipe'
        self.mouse_enabled = False
        self.dwell_ms = 700
        self.dwell_radius = 40
        self.esp32_url = ''
        self.esp32_q: Optional[queue.Queue] = None
        self.esp32_stop: Optional[threading.Event] = None
        self.synthetic_hud = False

APP_STATE = AppState()


# ============================
# Streamlit Video Transformer
# ============================
class GazeProcessor(VideoProcessorBase):
    def __init__(self):
        self.engine: Optional[GazeEngine] = None
        self.mouse: Optional[MouseController] = None
        self.overlay_size = (640, 480)
        self.last_gaze = (0.5, 0.5)
        self.last_calib_hit_t = None

    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if APP_STATE.extractor_mode == 'ESP32 (Orlosky)' and getattr(APP_STATE, 'synthetic_hud', False):
            img = np.full((480, 640, 3), 40, dtype=np.uint8)
        if self.engine is None:
            # Lazy init with MediaPipe extractor for webcam path
            extractor = FeatureExtractor(use_mediapipe=(APP_STATE.extractor_mode == 'Webcam/MediaPipe'))
            cfg = GazeConfig()
            # update screen size from monitor
            mc_tmp = MouseController(enable=False)
            cfg.screen_w, cfg.screen_h = mc_tmp.sw, mc_tmp.sh
            self.engine = GazeEngine(cfg, extractor)
            self.mouse = MouseController(enable=APP_STATE.mouse_enabled,
                                         dwell_ms=APP_STATE.dwell_ms,
                                         dwell_radius_px=APP_STATE.dwell_radius)
        # Process per mode
        if APP_STATE.extractor_mode == 'Webcam/MediaPipe':
            x, y, metrics, img = self.engine.process_frame(img)
        else:
            # ESP32 mode: consume latest nx,ny if available
            x, y = self.last_gaze
            if APP_STATE.esp32_q is not None:
                try:
                    nx, ny = APP_STATE.esp32_q.get_nowait()
                    x, y = self.engine.process_external(nx, ny)
                except Exception:
                    pass
            metrics = self.engine.get_report()
        self.last_gaze = (x, y)
        # Mouse
        if self.mouse:
            self.mouse.set_enable(APP_STATE.mouse_enabled)
            self.mouse.dwell_ms = APP_STATE.dwell_ms
            self.mouse.dwell_radius_px = APP_STATE.dwell_radius
            self.mouse.update(x, y, do_click=True)
        # Draw HUD & Calibration overlay
        overlay = self._draw_overlay(img, x, y, metrics)
        return VideoFrame.from_ndarray(overlay, format="bgr24")

    def _draw_overlay(self, img, x, y, metrics):
        if cv2 is None:
            try:
                st.error("OpenCV (cv2) ไม่ได้ติดตั้ง — วาดวงแหวน calibration ไม่ได้")
            except Exception:
                pass
            return img
        h, w = img.shape[:2]
        # crosshair at gaze
        gx, gy = int(x * w), int(y * h)
        cv2.circle(img, (gx, gy), 8, (0, 255, 0), 2)
        cv2.line(img, (gx-15, gy), (gx+15, gy), (0, 255, 0), 1)
        cv2.line(img, (gx, gy-15), (gx, gy+15), (0, 255, 0), 1)
        # metrics text
        txt = f"model={metrics.get('model','')}, fps={metrics.get('fps',0):.1f}, lat={metrics.get('latency_ms',0):.0f}ms"
        cv2.putText(img, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        # Calibration overlay
        if APP_STATE.calib_active and 0 <= APP_STATE.calib_idx < len(APP_STATE.calib_targets):
            tx, ty = APP_STATE.calib_targets[APP_STATE.calib_idx]
            cx, cy = int(tx * w), int(ty * h)
            cv2.circle(img, (cx, cy), 12, (0, 170, 255), 2)
            # dwell progress
            now = time.time()
            if self.last_calib_hit_t is None:
                # test hit
                if math.hypot(gx - cx, gy - cy) <= APP_STATE.calib_radius:
                    self.last_calib_hit_t = now
            else:
                elapsed_ms = (now - self.last_calib_hit_t) * 1000.0
                frac = min(1.0, elapsed_ms / APP_STATE.calib_dwell_ms)
                cv2.ellipse(img, (cx, cy), (16, 16), 0, 0, int(360 * frac), (0, 170, 255), 3)
                if elapsed_ms >= APP_STATE.calib_dwell_ms:
                    # capture sample
                    self.engine.calibration_add_point(tx, ty)
                    APP_STATE.calib_idx += 1
                    self.last_calib_hit_t = None
                    if APP_STATE.calib_idx >= len(APP_STATE.calib_targets):
                        res = self.engine.calibration_finish()
                        APP_STATE.calib_active = False
                        # show result briefly
                        msg = f"Calibration done: n={res.get('n','?')} rmse={res.get('rmse_norm','-'):.3f}" if res.get('ok') else res.get('msg','fail')
                        try:
                            st.toast(msg)
                        except Exception:
                            pass
            # Reset timer if out of target
            if self.last_calib_hit_t is not None and math.hypot(gx - cx, gy - cy) > APP_STATE.calib_radius:
                self.last_calib_hit_t = None
        return img


# ============================
# Streamlit App
# ============================

def _sidebar_controls():
    st.sidebar.header("⚙️ Settings")
    mode = st.sidebar.selectbox("Extractor", ["Webcam/MediaPipe", "ESP32 (Orlosky)"])
    APP_STATE.synthetic_hud = st.sidebar.toggle("Use Synthetic HUD (no webcam)", value=False) if mode == "ESP32 (Orlosky)" else False
    APP_STATE.extractor_mode = mode

    st.sidebar.subheader("Smoothing & Shaping")
    smooth_fc = st.sidebar.slider("OneEuro fc", 0.5, 8.0, 2.0, 0.1)
    smooth_beta = st.sidebar.slider("OneEuro beta", 0.0, 0.2, 0.02, 0.005)
    gain = st.sidebar.slider("Gain", 0.5, 2.5, 1.0, 0.05)
    gamma = st.sidebar.slider("Gamma", 0.5, 2.0, 1.0, 0.05)
    dead = st.sidebar.slider("Deadzone", 0.0, 0.1, 0.02, 0.005)

    st.sidebar.subheader("🖱️ Mouse Control")
    APP_STATE.mouse_enabled = st.sidebar.toggle("Enable Mouse Control", value=False)
    APP_STATE.dwell_ms = st.sidebar.slider("Dwell Click (ms)", 200, 1500, 700, 50)
    APP_STATE.dwell_radius = st.sidebar.slider("Dwell Radius (px)", 10, 120, 40, 2)

    if mode == "ESP32 (Orlosky)":
        st.sidebar.subheader("ESP32")
        APP_STATE.esp32_url = st.sidebar.text_input("Stream URL / Config", value="http://esp32.local/stream")
        col_e1, col_e2 = st.sidebar.columns(2)
        start_esp = col_e1.button("Start ESP32")
        stop_esp = col_e2.button("Stop ESP32")
        if start_esp:
            if APP_STATE.esp32_stop is None:
                APP_STATE.esp32_q = queue.Queue(maxsize=4)
                APP_STATE.esp32_stop = threading.Event()
                t = ESP32Reader(APP_STATE.esp32_url, APP_STATE.esp32_q, APP_STATE.esp32_stop)
                t.start()
        if stop_esp and APP_STATE.esp32_stop is not None:
            APP_STATE.esp32_stop.set()
            APP_STATE.esp32_stop = None

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧭 Calibration")
    npoints = st.sidebar.selectbox("Points", [9, 12], index=0)
    dwell_ms = st.sidebar.slider("Dwell per point (ms)", 400, 1500, 800, 50)
    APP_STATE.calib_dwell_ms = dwell_ms
    start_cal = st.sidebar.button("Start Calibration")
    if start_cal:
        # generate grid targets
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
        APP_STATE.calib_active = True
        st.toast("Calibration started — look at targets until ring completes")

    # Debug helper: force center target
    if st.sidebar.button("Force Show Target (debug)"):
        APP_STATE.calib_targets = [(0.5, 0.5)]
        APP_STATE.calib_idx = 0
        APP_STATE.calib_active = True
        st.toast("Debug target shown at screen center")

    # Push shaping params to current (or future) engine via a small cache in session_state
    st.session_state.setdefault('cfg_patch', {})
    st.session_state['cfg_patch'] = {
        'smooth_fc': smooth_fc,
        'smooth_beta': smooth_beta,
        'gain': gain,
        'gain_x': 1.0,
        'gain_y': 1.0,
        'gamma': gamma,
        'deadzone': dead,
    }


def _apply_cfg_patch_to_processor(processor: GazeProcessor):
    if processor and processor.engine and 'cfg_patch' in st.session_state:
        processor.engine.sess.smooth_x.mincutoff = st.session_state['cfg_patch']['smooth_fc']
        processor.engine.sess.smooth_y.mincutoff = st.session_state['cfg_patch']['smooth_fc']
        processor.engine.sess.smooth_x.beta = st.session_state['cfg_patch']['smooth_beta']
        processor.engine.sess.smooth_y.beta = st.session_state['cfg_patch']['smooth_beta']
        processor.engine.update_config(
            gain=st.session_state['cfg_patch']['gain'],
            gamma=st.session_state['cfg_patch']['gamma'],
            deadzone=st.session_state['cfg_patch']['deadzone'],
        )


def main():
    st.set_page_config(page_title="Gaze All-in-One (ONEFILE)", page_icon="👀", layout="wide")
    st.title("👀 Gaze All-in-One — Webcam / ESP32 / Mouse / Calibration")
    st.caption("One-file demo. If camera preview is black, check camera permissions or install opencv-python.")

    _sidebar_controls()

    # Start webrtc streamer for preview + overlay (used for both modes as HUD surface)
    ctx = webrtc_streamer(
        key="gaze-stream",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False,
        video_processor_factory=GazeProcessor,
    )

    # Live status panel
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mouse", "ON" if APP_STATE.mouse_enabled else "OFF")
    with col2:
        st.metric("Mode", APP_STATE.extractor_mode)
    with col3:
        st.metric("ESP32", "RUNNING" if APP_STATE.esp32_stop is not None else "STOPPED")
    st.markdown("### Diagnostics")
    st.write({
        "calib_active": APP_STATE.calib_active,
        "calib_idx": APP_STATE.calib_idx,
        "n_targets": len(APP_STATE.calib_targets),
        "mode": APP_STATE.extractor_mode,
        "synthetic_hud": getattr(APP_STATE, "synthetic_hud", False),
    })

    # Apply config patch to current transformer
    if ctx and ctx.video_processor:
        _apply_cfg_patch_to_processor(ctx.video_processor)

    # Drift controls & Report
    if ctx and ctx.video_processor and ctx.video_processor.engine:
        eng = ctx.video_processor.engine
        st.markdown("---")
        colA, colB, colC, colD = st.columns([1,1,1,2])
        with colA:
            if st.button("Start Drift"):
                eng.drift_start(); st.toast("Drift enabled")
        with colB:
            if st.button("Stop Drift"):
                eng.drift_stop(); st.toast("Drift disabled")
        with colC:
            if st.button("Add Drift Sample (use current gaze vs known point)"):
                # simple: use center as known point example; change as needed
                eng.drift_add(0.5, 0.5)
        with colD:
            st.write(eng.get_report())

    st.markdown("""
    **Tips**
    - Do calibration until the orange ring completes at each target.
    - Enable mouse when ready; adjust dwell/radius in sidebar.
    - ESP32 mode uses a simulated stream in this demo. Replace `ESP32Reader.run()` with your parsing logic.
    """)


if __name__ == "__main__":
    main()