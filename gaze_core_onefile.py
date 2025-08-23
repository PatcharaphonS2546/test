#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gaze_core_onefile.py (fixed fallback + gain + quick-recal + head-pose-only)
----------------------------------------------------------------------------

- แก้ fallback: ยังไม่คาลิเบรตก็จะไม่ดูดไป (0,0) แต่ map จาก eye-frame → จอกลาง (0.5,0.5) + เกนเล็กน้อย
- เพิ่ม cfg.gain: ขยายความไวหลังโมเดล (ปลอดภัย ไม่ทำลายคาลิเบรต)
- คง Quick-Recal / Online Drift Correction และ Head-Pose-Only Normalization
- แก้ __main__ ให้รันได้จริง

ติดตั้งขั้นต่ำ:
    pip install numpy opencv-python-headless
(แม่นขึ้น)   pip install mediapipe scikit-learn
"""
from __future__ import annotations
import math
import time
import uuid
from dataclasses import dataclass, field
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2

# ------------------------------
# Optional imports
# ------------------------------
try:
    import mediapipe as mp  # type: ignore
    _HAS_MEDIAPIPE = True
except Exception:
    _HAS_MEDIAPIPE = False

try:
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures  # type: ignore
    from sklearn.linear_model import Ridge, RANSACRegressor  
    from sklearn.pipeline import make_pipeline            # type: ignore
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


# =========================
# Utilities
# =========================

def _now() -> float:
    return time.time()

def _decode_frame(frame_in: Any) -> np.ndarray:
    """รับได้ทั้ง ndarray (BGR) หรือ JPEG bytes; คืน ndarray BGR"""
    if isinstance(frame_in, np.ndarray):
        return frame_in
    if isinstance(frame_in, (bytes, bytearray, memoryview)):
        npbuf = np.frombuffer(frame_in, dtype=np.uint8)
        im = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError("Invalid JPEG bytes")
        return im
    raise TypeError("frame_in must be np.ndarray (BGR) or JPEG bytes")

def _lap_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def _clahe_gamma(gray: np.ndarray, clip: float = 2.0, tiles: int = 4, gamma: float = 0.9) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    eq = clahe.apply(gray)
    lut = np.array([(i / 255.0) ** gamma * 255 for i in range(256)], np.uint8)
    return cv2.LUT(eq, lut)

def _quality_score(gray: np.ndarray) -> float:
    """สเกล 0..1 (คงที่, ไม่อิง absolute)"""
    blur = _lap_var(gray) / 150.0
    mean = gray.mean()
    bright = 1 - abs(mean - 128) / 128
    q = 0.7 * np.clip(blur, 0, 1) + 0.3 * np.clip(bright, 0, 1)
    return float(np.clip(q, 0, 1))

# =========================
# One Euro Filter (temporal smoothing)
# =========================

class OneEuro:
    def __init__(self, fc: float = 1.0, beta: float = 0.007, dt: float = 1 / 30.0, mincut: float = 1.0):
        self.fc = fc
        self.beta = beta
        self.dt = dt
        self.mincut = mincut
        self.xp: Optional[float] = None
        self.dxp: float = 0.0

    def _alpha(self, f: float) -> float:
        return 1.0 / (1.0 + 1.0 / (2.0 * np.pi * f * self.dt))

    def step(self, x: float, dt: Optional[float] = None) -> float:
        if dt is not None and 1/120 <= dt <= 1/10:
            self.dt = dt
        if self.xp is None:
            self.xp = x
            return x
        dx = (x - self.xp) / self.dt
        ad = self._alpha(self.fc)
        self.dxp = ad * dx + (1 - ad) * self.dxp
        f = self.mincut + self.beta * abs(self.dxp)
        a = self._alpha(f)
        self.xp = a * x + (1 - a) * self.xp
        return self.xp

# =========================
# MediaPipe landmarks / indices
# =========================

LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
L_EYE_OUT, L_EYE_IN = 33, 133
R_EYE_IN,  R_EYE_OUT = 362, 263

# head pose indices (approx)
IDX_NOSE_TIP = 1
IDX_CHIN = 152
IDX_LEFT_EYE_OUT = 263     # user's left (image right)
IDX_RIGHT_EYE_OUT = 33     # user's right (image left)
IDX_LEFT_MOUTH = 291
IDX_RIGHT_MOUTH = 61

# canonical 3D model points (mm)
MODEL_3D = np.array([
    [0.0,   0.0,   0.0],    # nose tip
    [0.0, -63.6, -12.5],    # chin
    [-43.3, 32.7, -26.0],   # left eye outer corner
    [ 43.3, 32.7, -26.0],   # right eye outer corner
    [-28.9,-28.9, -24.1],   # left mouth corner
    [ 28.9,-28.9, -24.1],   # right mouth corner
], dtype=np.float32)

@dataclass
class EyeFeatures:
    x: float
    y: float
    scale: float

@dataclass
class PoseInfo:
    yaw: float
    pitch: float
    roll: float
    dist_z: float
    ok: bool

def _eye_frame_coords(p_outer: np.ndarray, p_inner: np.ndarray, p_center: np.ndarray) -> EyeFeatures:
    v = p_inner - p_outer
    d = float(np.linalg.norm(v) + 1e-6)
    ex = v / d
    ey = np.array([-ex[1], ex[0]], dtype=np.float32)
    origin = (p_outer + p_inner) / 2.0
    rel = p_center - origin
    x = float((rel @ ex) / d)
    y = float((rel @ ey) / d)
    return EyeFeatures(x=x, y=y, scale=d)

def _camera_matrix(w: int, h: int, fov_deg: float = 60.0) -> np.ndarray:
    f = 0.5 * w / math.tan(math.radians(fov_deg/2.0))
    return np.array([[f, 0, w/2.0],
                     [0, f, h/2.0],
                     [0, 0, 1.0]], dtype=np.float32)

def _rvec_to_ypr(rvec: np.ndarray) -> Tuple[float,float,float]:
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    pitch = math.degrees(math.atan2(-R[2,0], sy))
    yaw   = math.degrees(math.atan2(R[1,0], R[0,0]))
    roll  = math.degrees(math.atan2(R[2,1], R[2,2]))
    return yaw, pitch, roll

class MediaPipeExtractor:
    def __init__(self, static_mode: bool = False, max_faces: int = 1):
        if not _HAS_MEDIAPIPE:
            raise RuntimeError("mediapipe is not available")
        self._fm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._fm.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark

        def P(i: int) -> np.ndarray:
            return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)

        # eye
        L_outer, L_inner = P(L_EYE_OUT), P(L_EYE_IN)
        R_outer, R_inner = P(R_EYE_OUT), P(R_EYE_IN)
        L_iris = np.mean([P(i) for i in LEFT_IRIS], axis=0)
        R_iris = np.mean([P(i) for i in RIGHT_IRIS], axis=0)
        L = _eye_frame_coords(L_outer, L_inner, L_iris)
        R = _eye_frame_coords(R_outer, R_inner, R_iris)
        interoc = float(np.linalg.norm(((L_outer+L_inner)/2) - ((R_outer+R_inner)/2)))

        # pose
        pts_2d = np.stack([
            P(IDX_NOSE_TIP), P(IDX_CHIN),
            P(IDX_LEFT_EYE_OUT), P(IDX_RIGHT_EYE_OUT),
            P(IDX_LEFT_MOUTH), P(IDX_RIGHT_MOUTH)
        ], axis=0)
        cam_mtx = _camera_matrix(w, h, fov_deg=60.0)
        dist_coefs = np.zeros((4,1), dtype=np.float32)
        ok, rvec, tvec = cv2.solvePnP(MODEL_3D, pts_2d, cam_mtx, dist_coefs, flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            yaw, pitch, roll = _rvec_to_ypr(rvec)
            dist_z = float(tvec[2,0])
            pose = PoseInfo(yaw=float(yaw), pitch=float(pitch), roll=float(roll), dist_z=dist_z, ok=True)
        else:
            pose = PoseInfo(yaw=0.0, pitch=0.0, roll=0.0, dist_z=0.0, ok=False)

        return {"L": L, "R": R, "interoc": interoc, "pose": pose, "ok": True}

# Fallback extractor (no mediapipe)
class CenterExtractor:
    def extract(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        h, w = frame_bgr.shape[:2]
        d = 0.25 * w
        L = EyeFeatures(x=0.4, y=0.5, scale=d)
        R = EyeFeatures(x=0.6, y=0.5, scale=d)
        pose = PoseInfo(yaw=0.0, pitch=0.0, roll=0.0, dist_z=0.0, ok=False)
        return {"L": L, "R": R, "interoc": 0.4*w, "pose": pose, "ok": True}

# =========================
# Feature builder
# =========================

@dataclass
class Config:
    enable_preprocess: bool = True
    quality_thresh: float = 0.40
    clamp_jump: float = 0.18
    ridge_lambda: float = 1e-3
    ransac_iters: int = 300
    ransac_thresh: float = 0.055
    poly_degree: int = 2
    min_eye_width_px: int = 12
    min_interoc_px: int = 40
    return_debug: bool = False
    use_mediapipe: bool = True
    head_pose_only_norm: bool = False
    gain: float = 1.0               # << ขยายความไวผลลัพธ์หลังโมเดล (1.0 = เดิม)
    # --- New shaping & control params ---
    gain_x: float = 1.0           # per-axis gain multiplier (X)
    gain_y: float = 1.0           # per-axis gain multiplier (Y)
    deadzone: float = 0.0         # neutral zone around center (0..0.1)
    gamma: float = 1.0            # <1 boosts edges, >1 compresses edges
    fallback_kx: float = 0.8      # fallback horizontal gain (no calibration)
    fallback_ky: float = 0.8      # fallback vertical gain (no calibration)
    smooth_fc: float = 2.0        # OneEuro base cutoff frequency
    smooth_beta: float = 0.02     # OneEuro speed coefficient

def build_base_features(raw: Dict[str, Any], cfg: Config) -> np.ndarray:
    L: EyeFeatures = raw["L"]
    R: EyeFeatures = raw["R"]
    io: float = raw["interoc"]
    pose: PoseInfo = raw.get("pose", PoseInfo(0,0,0,0,False))  # type: ignore

    if cfg.head_pose_only_norm:
        # โหมดฟีเจอร์กระชับ: avg/diff + pose
        avgx = 0.5*(L.x+R.x); avgy = 0.5*(L.y+R.y)
        dfx  = (R.x-L.x);     dfy  = (R.y-L.y)
        base = np.array([
            avgx, avgy, dfx, dfy,      # eye summary in eye-frame
            pose.yaw, pose.pitch, pose.roll,
            1.0
        ], dtype=np.float32)            # dim = 8
    else:
        base = np.array([
            L.x, L.y, R.x, R.y,
            L.scale, R.scale, io,
            pose.yaw, pose.pitch, pose.roll,
            1.0
        ], dtype=np.float32)            # dim = 11
    return base

def build_manual_poly(feat_base: np.ndarray, use_head_pose_only: bool) -> np.ndarray:
    # สร้างฟีเจอร์โพลีโดยไม่พึ่ง sklearn (ใช้กรณี fallback)
    if use_head_pose_only:
        sel = [0,1,2,3,4,5,6]   # [avgx,avgy,dfx,dfy,yaw,pitch,roll]
    else:
        sel = [0,1,2,3,6,7,8,9] # [Lx,Ly,Rx,Ry,io,yaw,pitch,roll]
    v = feat_base[sel]
    xs = [1.0]  # bias
    xs.extend(v.tolist())
    # เพิ่มกำลังสองและครอสเทอม
    for i in range(len(v)):
        xs.append(v[i]*v[i])
        for j in range(i+1, len(v)):
            xs.append(v[i]*v[j])
    return np.asarray(xs, dtype=np.float32)

# =========================
# Calibration (Ridge + RANSAC)
# =========================

def _ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    XtX = X.T @ X + lam * np.eye(X.shape[1], dtype=np.float32)
    return np.linalg.solve(XtX, X.T @ Y)  # (D,2)

def _ransac_ridge_np(X: np.ndarray, Y: np.ndarray, lam: float, iters: int, thresh: float) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    N, D = X.shape
    best_inl = None
    best_W = None
    m = max(8, D)
    for _ in range(iters):
        idx = rng.choice(N, size=m, replace=False)
        W = _ridge_fit(X[idx], Y[idx], lam)
        pred = X @ W
        err = np.linalg.norm(pred - Y, axis=1)
        inl = err < thresh
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl, best_W = inl, W
    W = _ridge_fit(X[best_inl], Y[best_inl], lam)
    return W, best_inl

def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))

# =========================
# Session state & metrics
# =========================

@dataclass
class Metrics:
    frames_in: int = 0
    frames_proc: int = 0
    drops: int = 0
    lat_ms: List[float] = field(default_factory=list)
    first_ts: float = field(default_factory=_now)

    def fps_avg(self) -> float:
        elapsed = max(1e-6, _now() - self.first_ts)
        return self.frames_proc / elapsed

    def drop_rate(self) -> float:
        total = max(1, self.frames_in)
        return self.drops / total

    def p50(self) -> float:
        if not self.lat_ms:
            return 0.0
        return float(np.percentile(np.array(self.lat_ms), 50))

    def p95(self) -> float:
        if not self.lat_ms:
            return 0.0
        return float(np.percentile(np.array(self.lat_ms), 95))

@dataclass
class DriftData:
    active: bool = False
    maxlen: int = 50
    lam: float = 1e-2
    min_samples: int = 10
    ema: float = 0.5
    Xf: deque = field(default_factory=lambda: deque(maxlen=50))   # (Df,)
    R:  deque = field(default_factory=lambda: deque(maxlen=50))   # (2,)
    delta_coef: Optional[np.ndarray] = None  # (2, Df)
    df: Optional[int] = None

@dataclass
class CalibrationData:
    feats: List[np.ndarray] = field(default_factory=list)  # base features
    targs: List[np.ndarray] = field(default_factory=list)  # (2,)
    W_runtime: Optional[Any] = None   # ("sk_v2", coef, mean, scale, degree) / ("np_poly", W)
    W_report: Optional[Dict[str, Any]] = None
    rmse_norm: Optional[float] = None
    inliers: Optional[np.ndarray] = None

@dataclass
class Smoothers:
    x: OneEuro = field(default_factory=lambda: OneEuro(fc=2.0, beta=0.02, dt=1/30.0))
    y: OneEuro = field(default_factory=lambda: OneEuro(fc=2.0, beta=0.02, dt=1/30.0))

@dataclass
class SessionState:
    sid: str
    created_at: float = field(default_factory=_now)
    metrics: Metrics = field(default_factory=Metrics)
    calib: CalibrationData = field(default_factory=CalibrationData)
    drift: DriftData = field(default_factory=DriftData)
    smooth: Smoothers = field(default_factory=Smoothers)
    last_feat: Optional[np.ndarray] = None     # base feature ล่าสุด
    last_pred: Optional[Tuple[float, float]] = None
    last_ts: Optional[float] = None

# =========================
# Core Engine
# =========================

class GazeEngine:
    """
    ใช้งาน:
      - start_session()
      - process_frame(session_id, frame, ts?) -> gaze (x,y) 0..1 + quality + timing
      - calibration_start/add_point/finish
      - quick-recal drift: drift_start/add/stop/status
      - metrics_snapshot(), get_report()
    """
    def __init__(self, config: Optional[Config] = None):
        self.cfg = config or Config()
        self._sessions: Dict[str, SessionState] = {}
        if self.cfg.use_mediapipe and _HAS_MEDIAPIPE:
            self._extractor = MediaPipeExtractor()
        else:
            self._extractor = CenterExtractor()

    # -------- Session lifecycle --------
    def start_session(self, sid: Optional[str] = None) -> str:
        sid = sid or uuid.uuid4().hex
        self._sessions[sid] = SessionState(sid=sid)
        # Apply smoother params from config
        s = self._sessions[sid]
        try:
            s.smooth.x.fc = float(getattr(self.cfg, "smooth_fc", s.smooth.x.fc))
            s.smooth.y.fc = float(getattr(self.cfg, "smooth_fc", s.smooth.y.fc))
            s.smooth.x.beta = float(getattr(self.cfg, "smooth_beta", s.smooth.x.beta))
            s.smooth.y.beta = float(getattr(self.cfg, "smooth_beta", s.smooth.y.beta))
        except Exception:
            pass
        return sid

    def get_session(self, sid: str) -> SessionState:
        s = self._sessions.get(sid)
        if not s:
            raise KeyError(f"session {sid} not found")
        return s

    def reset_session(self, sid: str):
        self._sessions[sid] = SessionState(sid=sid)

    def update_config(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
        # propagate smoother params dynamically
        if "smooth_fc" in kwargs or "smooth_beta" in kwargs:
            for s in self._sessions.values():
                try:
                    if "smooth_fc" in kwargs:
                        s.smooth.x.fc = float(self.cfg.smooth_fc)
                        s.smooth.y.fc = float(self.cfg.smooth_fc)
                    if "smooth_beta" in kwargs:
                        s.smooth.x.beta = float(self.cfg.smooth_beta)
                        s.smooth.y.beta = float(self.cfg.smooth_beta)
                except Exception:
                    continue

    # -------- Calibration API --------
    def calibration_start(self, sid: str):
        s = self.get_session(sid)
        s.calib = CalibrationData()

    def calibration_add_point(self, sid: str, screen_x: float, screen_y: float) -> int:
        s = self.get_session(sid)
        if s.last_feat is None:
            raise RuntimeError("No feature captured yet. Call process_frame before adding a point.")
        s.calib.feats.append(s.last_feat.copy())
        s.calib.targs.append(np.array([screen_x, screen_y], dtype=np.float32))
        return len(s.calib.feats)

    def calibration_finish(self, sid: str) -> dict:
        s = self.get_session(sid)
        Xb = np.asarray(s.calib.feats)
        Y  = np.asarray(s.calib.targs)
        if len(Xb) < 20:
            raise RuntimeError("Not enough calibration points")

        if _HAS_SKLEARN:
            # เส้นทางเดิม (sklearn)
            scaler = StandardScaler()
            poly = PolynomialFeatures(self.cfg.poly_degree)
            base = make_pipeline(scaler, poly, Ridge(alpha=self.cfg.ridge_lambda))
            ransac = RANSACRegressor(estimator=base,
                                    min_samples=max(8, int(0.5 * Xb.shape[0])),
                                    residual_threshold=self.cfg.ransac_thresh,
                                    random_state=0)
            ransac.fit(Xb, Y)
            inlier_mask = ransac.inlier_mask_
            Y_pred = ransac.predict(Xb)
            rmse_norm = float(np.sqrt(np.mean((Y_pred - Y)**2)))

            s.calib.W_runtime = (
                "sk_v2",
                ransac.estimator_.named_steps['ridge'].coef_,
                ransac.estimator_.named_steps['ridge'].intercept_,
                ransac.estimator_.named_steps['standardscaler'].mean_,
                ransac.estimator_.named_steps['standardscaler'].scale_,
                self.cfg.poly_degree
            )
            s.calib.W_report = {
                "type": "sk_v2",
                "W": ransac.estimator_.named_steps['ridge'].coef_.tolist(),
                "intercept": ransac.estimator_.named_steps['ridge'].intercept_.tolist(),
                "scaler_mean": ransac.estimator_.named_steps['standardscaler'].mean_.tolist(),
                "scaler_scale": ransac.estimator_.named_steps['standardscaler'].scale_.tolist(),
                "poly_degree": self.cfg.poly_degree,
                "rmse_norm": rmse_norm,
                "inliers": int(inlier_mask.sum()),
                "points": int(len(Xb)),
                "head_pose_only_norm": self.cfg.head_pose_only_norm,
            }
            s.calib.inliers = inlier_mask
            s.calib.rmse_norm = rmse_norm
        else:
            # Fallback (ไม่มี sklearn): ใช้ manual poly + RANSAC ridge แบบ numpy
            Xf = np.stack([build_manual_poly(f, self.cfg.head_pose_only_norm) for f in Xb], axis=0)
            W, inlier_mask = _ransac_ridge_np(
                Xf, Y,
                lam=self.cfg.ridge_lambda,
                iters=self.cfg.ransac_iters,
                thresh=self.cfg.ransac_thresh
            )
            Y_pred = Xf @ W
            rmse_norm = float(np.sqrt(np.mean((Y_pred - Y)**2)))

            s.calib.W_runtime = ("np_poly", W.astype(np.float32))
            s.calib.W_report = {
                "type": "np_poly",
                "W": W.tolist(),
                "rmse_norm": rmse_norm,
                "inliers": int(inlier_mask.sum()),
                "points": int(len(Xb)),
                "head_pose_only_norm": self.cfg.head_pose_only_norm,
            }
            s.calib.inliers = inlier_mask
            s.calib.rmse_norm = rmse_norm

        return {
            "ok": True,
            "rmse_norm": s.calib.rmse_norm,
            "inliers": int(s.calib.inliers.sum()),
            "points": int(len(Xb)),
        }


    # -------- Quick-Recal / Online Drift API --------
    def drift_start(self, sid: str, maxlen: int = 50, lam: float = 1e-2, min_samples: int = 10, ema: float = 0.5):
        s = self.get_session(sid)
        s.drift = DriftData(active=True, maxlen=maxlen, lam=lam, min_samples=min_samples, ema=ema,
                            Xf=deque(maxlen=maxlen), R=deque(maxlen=maxlen), delta_coef=None, df=None)

    def drift_add(self, sid: str, screen_x: float, screen_y: float) -> Dict[str, Any]:
        """
        เรียกทันทีที่รู้ตำแหน่งจริง ณ เฟรมล่าสุด (เช่น ผู้ใช้คลิกที่หน้าจอ)
        ใช้ฟีเจอร์ล่าสุด s.last_feat → สร้าง Xf → residual = y_true - y_pred_base → อัปเดต Δ
        """
        s = self.get_session(sid)
        if not s.drift.active:
            return {"ok": False, "reason": "drift not active"}
        if s.calib.W_runtime is None:
            return {"ok": False, "reason": "no calibration"}
        if s.last_feat is None:
            return {"ok": False, "reason": "no last feature (call process_frame first)"}

        feat_base = s.last_feat
        # สร้าง Xf และ y_pred_base
        Xf = self._to_Xf(s, feat_base)   # (1, Df)
        pred_base = self._predict_base_from_Xf(s, Xf)  # (1,2)
        y_true = np.array([[screen_x, screen_y]], dtype=np.float32)
        resid = (y_true - pred_base).astype(np.float32)  # (1,2)

        # เก็บ buffer
        s.drift.Xf.append(Xf.reshape(-1))   # (Df,)
        s.drift.R.append(resid.reshape(-1)) # (2,)
        s.drift.df = Xf.shape[1]

        updated = False
        if len(s.drift.Xf) >= s.drift.min_samples:
            X = np.stack(list(s.drift.Xf), axis=0)  # (N, Df)
            R = np.stack(list(s.drift.R),  axis=0)  # (N, 2)
            # Δ (Df x 2) = (X^T X + λI)^-1 X^T R
            XtX = X.T @ X + s.drift.lam * np.eye(X.shape[1], dtype=np.float32)
            Delta = np.linalg.solve(XtX, X.T @ R)   # (Df,2)
            Delta = Delta.astype(np.float32)

            # EMA กับของเดิม
            if s.drift.delta_coef is None:
                s.drift.delta_coef = Delta.T        # (2,Df)
            else:
                s.drift.delta_coef = (1.0 - s.drift.ema) * s.drift.delta_coef + s.drift.ema * Delta.T
            updated = True

        return {"ok": True, "buffer": len(s.drift.Xf), "updated": updated}

    def drift_stop(self, sid: str):
        s = self.get_session(sid)
        s.drift.active = False

    def drift_status(self, sid: str) -> Dict[str, Any]:
        s = self.get_session(sid)
        return {
            "active": s.drift.active,
            "buffer": len(s.drift.Xf),
            "df": s.drift.df,
            "has_delta": s.drift.delta_coef is not None
        }

    # -------- Processing --------
    def process_frame(self, sid, frame_in, ts=None):
        s = self.get_session(sid)
        ts = ts or _now()
        t_all0 = time.perf_counter()

        # 1) decode + preprocess + quality
        t0 = time.perf_counter()
        frame = _decode_frame(frame_in)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_norm = _clahe_gamma(gray) if self.cfg.enable_preprocess else gray
        q_pre = _quality_score(gray_norm)
        if q_pre < self.cfg.quality_thresh:
            s.metrics.frames_in += 1; s.metrics.drops += 1
            lat = (time.perf_counter()-t_all0)*1000.0
            s.metrics.lat_ms.append(lat)
            return {"ts": ts, "x": (s.last_pred or (0.5,0.5))[0], "y": (s.last_pred or (0.5,0.5))[1],
                    "quality": float(q_pre), "fps_proc": 0.0, "latency_ms": float(lat),
                    "timings_ms": {"preprocess": (time.perf_counter()-t0)*1000.0, "extract": 0.0, "estimate": 0.0, "total": lat},
                    "dropped": True}

        # 2) extract features (MediaPipe/Center)
        t1 = time.perf_counter()
        raw = self._extractor.extract(frame)
        t_ext = (time.perf_counter()-t1)*1000.0
        if raw is None:
            s.metrics.frames_in += 1; s.metrics.drops += 1
            lat = (time.perf_counter()-t_all0)*1000.0
            s.metrics.lat_ms.append(lat)
            return {"ts": ts, "x": (s.last_pred or (0.5,0.5))[0], "y": (s.last_pred or (0.5,0.5))[1],
                    "quality": float(q_pre), "fps_proc": 0.0, "latency_ms": float(lat),
                    "timings_ms": {"preprocess": (time.perf_counter()-t0)*1000.0, "extract": t_ext, "estimate": 0.0, "total": lat},
                    "dropped": True}

        L, R = raw["L"], raw["R"]; pose = raw.get("pose")
        s.last_feat = build_base_features(raw, self.cfg)
        avgx = 0.5*(L.x+R.x); avgy = 0.5*(L.y+R.y)

        # 3) ส่งเข้าท่อเดียวกับ external (รวม shaping/smooth/metrics ในฟังก์ชันนั้น)
        t2 = time.perf_counter()
        res = self.process_external(sid, avgx, avgy, pose.yaw, pose.pitch, pose.roll, ts=ts)
        t_est = (time.perf_counter()-t2)*1000.0

        # 4) เสริม timing/quality รวม แล้วคืนค่า
        total = (time.perf_counter()-t_all0)*1000.0
        s.metrics.lat_ms.append(total)
        res["quality"] = float(q_pre)
        res["timings_ms"] = {"preprocess": (time.perf_counter()-t0)*1000.0, "extract": t_ext, "estimate": t_est, "total": total}
        return res

        
    def process_external(self, sid: str, avgx: float, avgy: float, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0, ts: Optional[float] = None) -> Dict[str, Any]:
        """
        ใช้เมื่อมีฟีเจอร์ normalized (เช่น (avgx,avgy) จาก ESP32/Orlosky) แล้ว
        คืนผลลัพธ์เหมือน process_frame(): {'x','y','gaze',(x,y),'quality','fps_proc','latency_ms','timings_ms'}
        - avgx,avgy ควรอยู่ราว [-0.5, +0.5]
        - ถ้ามีโมเดลคาลิเบรตแล้วจะใช้โมเดล + drift correction; ไม่มีก็ fallback mapping
        """
        import time, math
        t0 = time.perf_counter()
        s = self.get_session(sid)
        ts = _now() if ts is None else ts

        # base features (8D) เหมือน head_pose_only_norm
        feat_base = np.array([avgx, avgy, 0.0, 0.0, yaw, pitch, roll, 1.0], dtype=np.float32)
        s.last_feat = feat_base.copy()

        # 1) คำนวณ gaze พื้นฐาน
        if s.calib.W_runtime is not None:
            Xf = self._to_Xf(s, feat_base)             # (1, Df)
            pred = self._predict_base_from_Xf(s, Xf)   # (1, 2)
            gx, gy = float(pred[0, 0]), float(pred[0, 1])
            # Drift correction
            if s.drift.active and s.drift.delta_coef is not None and Xf.shape[1] == s.drift.delta_coef.shape[1]:
                corr = (Xf @ s.drift.delta_coef.T).astype(np.float32)
                gx += float(corr[0, 0])
                gy += float(corr[0, 1])
            quality = 1.0
        else:
            # ไม่มีคาลิเบรต → map ตรงกลางจอด้วยเกนเล็กน้อย (ตั้งค่าได้)
            kx = float(getattr(self.cfg, 'fallback_kx', 0.8))
            ky = float(getattr(self.cfg, 'fallback_ky', 0.8))
            gx = 0.5 + kx * avgx
            gy = 0.5 + ky * avgy
            quality = 1.0

        # 2) shaping: deadzone + gamma + per-axis gain
        gtot_x = float(getattr(self.cfg, 'gain', 1.0)) * float(getattr(self.cfg, 'gain_x', 1.0))
        gtot_y = float(getattr(self.cfg, 'gain', 1.0)) * float(getattr(self.cfg, 'gain_y', 1.0))
        dz = float(getattr(self.cfg, 'deadzone', 0.0))
        gm = float(getattr(self.cfg, 'gamma', 1.0))

        def _shape(val: float, gaxis: float) -> float:
            v = val - 0.5
            if abs(v) < dz:
                v = 0.0
            else:
                v = np.sign(v) * (abs(v) ** gm)
            return 0.5 + gaxis * v

        gx = _shape(gx, gtot_x)
        gy = _shape(gy, gtot_y)

        # 3) anti-jump clamp + temporal smooth
        if s.last_pred is not None:
            dx = gx - s.last_pred[0]
            dy = gy - s.last_pred[1]
            if abs(dx) > self.cfg.clamp_jump:
                gx = s.last_pred[0] + math.copysign(self.cfg.clamp_jump, dx)
            if abs(dy) > self.cfg.clamp_jump:
                gy = s.last_pred[1] + math.copysign(self.cfg.clamp_jump, dy)

        dt = None
        if s.last_ts is not None:
            dt = max(1e-6, ts - s.last_ts)

        gx = float(np.clip(s.smooth.x.step(gx, dt=dt), 0.0, 1.0))
        gy = float(np.clip(s.smooth.y.step(gy, dt=dt), 0.0, 1.0))
        s.last_pred = (gx, gy)
        s.last_ts = ts

        # metrics
        latency_ms = (time.perf_counter() - t0) * 1000.0
        s.metrics.frames_in += 1
        s.metrics.frames_proc += 1
        s.metrics.lat_ms.append(latency_ms)

        return {
            'ts': ts,
            'x': gx,
            'y': gy,
            'gaze': (gx, gy),
            'quality': float(quality),
            'fps_proc': 0.0,
            'latency_ms': float(latency_ms),
            'timings_ms': {'external': float(latency_ms)},
        }
    
    # -------- Helpers for transform/predict --------
    def _to_Xf(self, s: SessionState, feat_base: np.ndarray) -> np.ndarray:
        """แปลง base feature -> Xf ที่ใช้ในโมเดลคาลิเบรชัน"""
        if s.calib.W_runtime is None:
            return feat_base.reshape(1, -1)
        if s.calib.W_runtime[0] == "sk_v2":
            _, coef, intercept, mean, scale, degree = s.calib.W_runtime
            Xs = (feat_base - mean) / (scale + 1e-8)
            if _HAS_SKLEARN:
                poly = PolynomialFeatures(degree=int(degree), include_bias=True)
                poly.fit(np.zeros((1, Xs.shape[0]), dtype=np.float32))
                Xf = poly.transform(Xs.reshape(1, -1)).astype(np.float32)
            else:
                Xf = np.concatenate([np.ones((1,1), np.float32), Xs.reshape(1,-1)], axis=1)
            return Xf
        elif s.calib.W_runtime[0] == "np_poly":
            _, W = s.calib.W_runtime
            Xf = build_manual_poly(feat_base, self.cfg.head_pose_only_norm).reshape(1, -1)
            return Xf.astype(np.float32)
        else:
            return feat_base.reshape(1, -1)

    def _predict_base_from_Xf(self, s: SessionState, Xf: np.ndarray) -> np.ndarray:
        if s.calib.W_runtime is None:
            return np.array([[0.5, 0.5]], dtype=np.float32)
        if s.calib.W_runtime[0] == "sk_v2":
            _, coef, intercept, _, _, _ = s.calib.W_runtime
            return (Xf @ coef.T + intercept.reshape(1, -1)).astype(np.float32)  # (1,2)
        elif s.calib.W_runtime[0] == "np_poly":
            _, W = s.calib.W_runtime
            return (Xf @ W).astype(np.float32)
        return np.array([[0.5,0.5]], dtype=np.float32)

    # -------- Metrics & Report --------
    def metrics_snapshot(self, sid: str) -> Dict[str, Any]:
        s = self.get_session(sid)
        return {
            "frames_in": s.metrics.frames_in,
            "frames_proc": s.metrics.frames_proc,
            "drop_rate": s.metrics.drop_rate(),
            "fps_avg": s.metrics.fps_avg(),
            "latency_ms_p50": s.metrics.p50(),
            "latency_ms_p95": s.metrics.p95(),
            "rmse_norm": s.calib.rmse_norm,
        }

    def get_report(self, sid: str) -> Dict[str, Any]:
        s = self.get_session(sid)
        return {
            "session_id": sid,
            "created_at": s.created_at,
            "metrics": self.metrics_snapshot(sid),
            "calibration": {
                "points": len(s.calib.feats),
                "rmse_norm": s.calib.rmse_norm,
                "inliers": int(s.calib.inliers.sum()) if s.calib.inliers is not None else None,
                "W": s.calib.W_report,
                "head_pose_only_norm": bool(self.cfg.head_pose_only_norm),
            },
            "drift": {
                "active": s.drift.active,
                "buffer": len(s.drift.Xf),
                "lam": s.drift.lam,
                "min_samples": s.drift.min_samples,
                "ema": s.drift.ema,
                "has_delta": s.drift.delta_coef is not None,
                "df": s.drift.df,
            }
        }


if __name__ == "__main__":
    # Minimal dry-run
    eng = GazeEngine(Config(head_pose_only_norm=False, gain=1.0))
    sid = eng.start_session()
    dummy = np.zeros((360,640,3), np.uint8)
    for _ in range(3):
        print(eng.process_frame(sid, dummy))
    print("metrics:", eng.metrics_snapshot(sid))