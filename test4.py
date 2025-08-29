# test3.py
# 👁️ Vision Mouse — Pointer-only eye control (Webcam/MediaPipe + ESP32-CAM)
# - 9-point calibration (no auto-extend)
# - One-Euro filter smoothing
# - Metrics recorder + bar charts (Streamlit) และปุ่มบันทึกกราฟเป็นไฟล์ PNG (สำหรับโหมด Webcam)
# - Evaluation CSV export
#
# Run: streamlit run test3.py

from __future__ import annotations
import math, time, threading, queue, os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import streamlit as st

# Optional deps
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
    import pandas as pd
except Exception:
    pd = None
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # ใช้บันทึกไฟล์ PNG; แนะนำให้ติดตั้ง matplotlib

# --- Optional sound deps for Thai Soundboard ---
try:
    import simpleaudio as sa  # best for .wav
except Exception:
    sa = None
try:
    from playsound import playsound as _playsound  # fallback for .mp3
except Exception:
    _playsound = None

from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from av import VideoFrame


# ----------------------------- Helpers -----------------------------
def _safe_size() -> Tuple[int, int]:
    if pyautogui:
        try:
            return pyautogui.size()
        except Exception:
            pass
    return 1920, 1080



def play_sound_file(path: str) -> bool:
    """Play an audio file non-blocking on the server machine.
    Returns True if a backend accepted the file.
    """
    try:
        if sa and path.lower().endswith((".wav", ".wave")):
            wave_obj = sa.WaveObject.from_wave_file(path)
            wave_obj.play()
            return True
        elif _playsound:
            import threading as _th
            _th.Thread(target=_playsound, args=(path,), daemon=True).start()
            return True
    except Exception as e:
        print("play_sound failed:", e)
    return False

def _inside_rect(nx: float, ny: float, rect) -> bool:
    x0, y0, x1, y1 = rect
    return (x0 <= nx <= x1) and (y0 <= ny <= y1)

# ----------------------------- One Euro Filter -----------------------------
class OneEuroFilter:
    def __init__(self, freq=120.0, mincutoff=1.0, beta=0.01, dcutoff=1.0):
        self.freq = float(freq)
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _alpha(self, cutoff):
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
        a_d = self._alpha(self.dcutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


# ----------------------------- Mouse (Pointer only) -----------------------------
class MouseController:
    def __init__(self, enable: bool = False, rate_limit_ms: int = 33):
        self.enable = bool(enable)
        self.rate_ms = int(rate_limit_ms)
        self.sw, self.sh = _safe_size()
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


# ----------------------------- Metrics Recorder -----------------------------
def _dist_px(p, q):
    if p is None or q is None:
        return None
    dx = float(p[0]) - float(q[0])
    dy = float(p[1]) - float(q[1])
    return math.hypot(dx, dy)

@dataclass
class Sample:
    ts_ms: float
    latency_ms: float | None
    pred_px: tuple[int,int] | None
    target_px: tuple[int,int] | None
    point_id: int | None

class MetricsRecorder:
    def __init__(self):
        self.samples: list[Sample] = []
        self._maxlen = 120000

    def record(self, *, latency_ms, pred_px, target_px=None, point_id=None):
        ts_ms = time.time() * 1000.0
        if len(self.samples) >= self._maxlen:
            self.samples = self.samples[-self._maxlen//2:]
        self.samples.append(Sample(ts_ms, latency_ms, pred_px, target_px, point_id))

    def _df(self):
        if not self.samples or pd is None:
            return None
        rows = []
        for s in self.samples:
            err = _dist_px(s.pred_px, s.target_px) if (s.pred_px and s.target_px) else None
            rows.append({
                "ts_ms": s.ts_ms,
                "latency_ms": s.latency_ms,
                "pred_x": None if s.pred_px is None else s.pred_px[0],
                "pred_y": None if s.pred_px is None else s.pred_px[1],
                "tgt_x": None if s.target_px is None else s.target_px[0],
                "tgt_y": None if s.target_px is None else s.target_px[1],
                "point_id": s.point_id,
                "err_px": err,
            })
        return pd.DataFrame(rows)

    def summarize(self):
        if pd is None:
            return None, None
        df = self._df()
        if df is None or df.empty:
            return (
                pd.DataFrame({"metric":["Latency(ms)","Jitter(px)","MAE(px)","RMSE(px)"], "value":[np.nan]*4}),
                pd.DataFrame(columns=["point_id","jitter_px","mae_px","rmse_px","n"])
            )
        df_t = df.dropna(subset=["tgt_x","tgt_y","err_px"])
        lat_mean = float(np.nanmean(df["latency_ms"])) if "latency_ms" in df and len(df) else np.nan
        if df_t.empty:
            return (
                pd.DataFrame({"metric":["Latency(ms)","Jitter(px)","MAE(px)","RMSE(px)"], "value":[lat_mean,np.nan,np.nan,np.nan]}),
                pd.DataFrame(columns=["point_id","jitter_px","mae_px","rmse_px","n"])
            )
        mae = float(np.mean(df_t["err_px"]))
        rmse = float(np.sqrt(np.mean(np.square(df_t["err_px"]))))
        jitter = float(np.std(df_t["err_px"]))
        df_global = pd.DataFrame({
            "metric": ["Latency(ms)","Jitter(px)","MAE(px)","RMSE(px)"],
            "value":  [lat_mean,      jitter,       mae,       rmse]
        })
        g = df_t.groupby("point_id")["err_px"]
        df_points = pd.DataFrame({
            "point_id": g.count().index.astype(int),
            "jitter_px": g.std().values,
            "mae_px": g.mean().values,
            "rmse_px": np.sqrt(g.apply(lambda s: np.mean(np.square(s))).values),
            "n": g.count().values
        }).sort_values("point_id").reset_index(drop=True)
        return df_global, df_points

    def save_csv(self, prefix="metrics"):
        if pd is None:
            return None, None
        df_global, df_points = self.summarize()
        ts = time.strftime("%Y%m%d-%H%M%S")
        p1 = f"{prefix}_global_{ts}.csv"
        p2 = f"{prefix}_points_{ts}.csv"
        df_global.to_csv(p1, index=False, encoding="utf-8-sig")
        df_points.to_csv(p2, index=False, encoding="utf-8-sig")
        return p1, p2

    # ---------- NEW: บันทึกรูป PNG สำหรับโหมด Webcam ----------
    def export_webcam_charts(self, *, export_dir="exports", deg_per_px: float | None = None) -> list[str]:
        """
        สร้างไฟล์กราฟ PNG 6 แบบสำหรับผลจากโหมด Webcam:
        - Latency (mean±std), Summary Jitter, Per-point MAE, Per-point Jitter, Per-point RMSE, Summary Accuracy
        คืนค่า: รายการ path ของไฟล์ที่บันทึกสำเร็จ
        """
        if pd is None or plt is None:
            raise RuntimeError("ต้องติดตั้ง pandas และ matplotlib ก่อน (pip install pandas matplotlib)")

        os.makedirs(export_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")

        df = self._df()
        if df is None or df.empty:
            raise RuntimeError("ยังไม่มีข้อมูลเมตริกให้บันทึก (ลองคาลิเบรต/ใช้งานโหมด Webcam ให้มีข้อมูลก่อน)")

        df_global, df_points = self.summarize()
        saved = []

        # 1) Latency mean ± std
        lat_mean = float(np.nanmean(df["latency_ms"])) if "latency_ms" in df and len(df) else np.nan
        lat_std  = float(np.nanstd(df["latency_ms"])) if "latency_ms" in df and len(df) else np.nan
        fig = plt.figure(figsize=(10,6))
        plt.bar(["Latency_mean_ms"], [lat_mean], label="Latency", color="orange")
        plt.errorbar([0], [lat_mean], yerr=[lat_std], fmt='k|', lw=2)
        plt.title("Latency (Webcam): mean ± std")
        plt.ylabel("Milliseconds")
        plt.tight_layout()
        p = f"{export_dir}/webcam_latency_{ts}.png"; fig.savefig(p, dpi=160); plt.close(fig); saved.append(p)

        # 2) Summary Jitter metrics (จาก per-point std)
        if not df_points.empty:
            jitter_vals = df_points["jitter_px"].dropna().values
            j_mean = float(np.mean(jitter_vals)) if jitter_vals.size else np.nan
            j_median = float(np.median(jitter_vals)) if jitter_vals.size else np.nan
            j_std = float(np.std(jitter_vals)) if jitter_vals.size else np.nan
        else:
            j_mean = j_median = j_std = np.nan
        fig = plt.figure(figsize=(10,6))
        plt.bar(["Jitter_mean_px","Jitter_median_px","Jitter_std_px"], [j_mean, j_median, j_std], color="orange")
        plt.title("Summary Jitter Metrics (Webcam)")
        plt.ylabel("Pixels")
        plt.tight_layout()
        p = f"{export_dir}/webcam_jitter_summary_{ts}.png"; fig.savefig(p, dpi=160); plt.close(fig); saved.append(p)

        # 3) Per-Point MAE
        fig = plt.figure(figsize=(10,6))
        if not df_points.empty:
            plt.bar(df_points["point_id"].astype(int).astype(str), df_points["mae_px"], color="orange")
        plt.title("Per-Point Accuracy: MAE (Webcam)")
        plt.xlabel("Point Index (0–8)")
        plt.ylabel("MAE (pixels)")
        plt.tight_layout()
        p = f"{export_dir}/webcam_mae_per_point_{ts}.png"; fig.savefig(p, dpi=160); plt.close(fig); saved.append(p)

        # 4) Per-Point Jitter (std)
        fig = plt.figure(figsize=(10,6))
        if not df_points.empty:
            plt.bar(df_points["point_id"].astype(int).astype(str), df_points["jitter_px"], color="orange")
        plt.title("Per-Point Stability: Jitter Mean (Webcam)")
        plt.xlabel("Point Index (0–8)")
        plt.ylabel("Jitter Mean (pixels)")
        plt.tight_layout()
        p = f"{export_dir}/webcam_jitter_mean_per_point_{ts}.png"; fig.savefig(p, dpi=160); plt.close(fig); saved.append(p)

        # 5) Per-Point RMSE
        fig = plt.figure(figsize=(10,6))
        if not df_points.empty:
            plt.bar(df_points["point_id"].astype(int).astype(str), df_points["rmse_px"], color="orange")
        plt.title("Per-Point Accuracy: RMSE (Webcam)")
        plt.xlabel("Point Index (0–8)")
        plt.ylabel("RMSE (pixels)")
        plt.tight_layout()
        p = f"{export_dir}/webcam_rmse_per_point_{ts}.png"; fig.savefig(p, dpi=160); plt.close(fig); saved.append(p)

        # 6) Summary Accuracy (MAE/RMSE in px และแปลงเป็น deg ถ้ามี deg/px)
        mae_px = float(df_global.loc[df_global["metric"]=="MAE(px)","value"].values[0])
        rmse_px = float(df_global.loc[df_global["metric"]=="RMSE(px)","value"].values[0])
        labels = ["MAE_px","RMSE_px"]; values = [mae_px, rmse_px]
        if (deg_per_px is not None) and (deg_per_px > 0):
            labels += ["MAE_deg","RMSE_deg"]
            values += [mae_px * deg_per_px, rmse_px * deg_per_px]
        fig = plt.figure(figsize=(10,6))
        plt.bar(labels, values, color="orange")
        plt.title("Summary Accuracy Metrics (Webcam)")
        plt.ylabel("Value")
        plt.tight_layout()
        p = f"{export_dir}/webcam_accuracy_summary_{ts}.png"; fig.savefig(p, dpi=160); plt.close(fig); saved.append(p)

        return saved


# ใช้ MetricsRecorder ตัวเดียวที่อยู่ใน session_state เสมอ
if "METRICS" not in st.session_state:
    st.session_state["METRICS"] = MetricsRecorder()
METRICS: MetricsRecorder = st.session_state["METRICS"]


# ----------------------------- Feature Extractor -----------------------------
class FeatureExtractor:
    LEFT_EYE_IDS = [33, 133, 159, 145]
    RIGHT_EYE_IDS = [362, 263, 386, 374]
    LEFT_IRIS_IDS = [468, 469, 470, 471, 472]
    RIGHT_IRIS_IDS = [473, 474, 475, 476, 477]

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

    def extract(self, frame_bgr) -> Optional[dict]:
        if frame_bgr is None:
            return None
        if not self.use_mediapipe or self.mesh is None:
            # Fallback: center + low quality
            return {
                "eye_cx_norm": 0.5, "eye_cy_norm": 0.5,
                "face_cx_norm": 0.5, "face_cy_norm": 0.5,
                "eye_open": True, "quality": 0.2,
            }

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(frame_rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0]
        pts = [(lm.landmark[i].x * w, lm.landmark[i].y * h, lm.landmark[i].z) for i in range(len(lm.landmark))]
        l_iris = self._avg_xy(self.LEFT_IRIS_IDS, pts)
        r_iris = self._avg_xy(self.RIGHT_IRIS_IDS, pts)
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
            eye_cx_norm, eye_cy_norm = 0.5, 0.5
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
            "eye_cx_norm": eye_cx_norm, "eye_cy_norm": eye_cy_norm,
            "face_cx_norm": face_cx_norm, "face_cy_norm": face_cy_norm,
            "eye_open": eye_open, "quality": float(quality),
        }


# ----------------------------- Simple Polynomial Mapper -----------------------------
def _feat_vec(feat: dict) -> np.ndarray:
    ex = float(feat.get("eye_cx_norm", 0.5))
    ey = float(feat.get("eye_cy_norm", 0.5))
    fx = 0.5 + (float(feat.get("face_cx_norm", 0.5)) - 0.5) * 0.3
    fy = 0.5 + (float(feat.get("face_cy_norm", 0.5)) - 0.5) * 0.3
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

def _fit_ridge_manual(Phi, y, lam=1e-3):
    A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
    b = Phi.T @ y
    w = np.linalg.solve(A, b)
    return w.reshape(-1)

def _predict_poly(wx, wy, X):
    Phi = _poly_expand(X)
    x = Phi @ wx
    y = Phi @ wy
    return np.stack([x, y], axis=1).astype(np.float32)


# ----------------------------- ESP32 lightweight iris -----------------------------
def _crop_ratio(img, width=640, height=480):
    if cv2 is None:
        return img
    h, w = img.shape[:2]
    desired = width / height
    cur = w / h
    if cur > desired:
        new_w = int(desired * h); off = (w - new_w) // 2
        img = img[:, off:off+new_w]
    else:
        new_h = int(w / desired); off = (h - new_h) // 2
        img = img[off:off+new_h, :]
    return cv2.resize(img, (width, height))

def _iris_center_norm(frame_bgr):
    if cv2 is None or frame_bgr is None:
        return None
    f = _crop_ratio(frame_bgr, 640, 480)
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    win = 24; step = 12
    best = 1e18; bx = by = None
    H, W = gray.shape[:2]
    for y in range(0, H - win, step):
        for x in range(0, W - win, step):
            s = float(gray[y:y+win, x:x+win].sum())
            if s < best:
                best, bx, by = s, x + win//2, y + win//2
    if bx is None:
        return None
    return float(bx)/W, float(by)/H

class ESP32Reader(threading.Thread):
    def __init__(self, url: str, out_q: queue.Queue, stop_evt: threading.Event):
        super().__init__(daemon=True)
        self.url = url
        self.out_q = out_q
        self.stop_evt = stop_evt

    def run(self):
        if cv2 is None:
            return
        while not self.stop_evt.is_set():
            cap = cv2.VideoCapture(self.url)
            if not cap.isOpened():
                time.sleep(1.0); continue
            try:
                while not self.stop_evt.is_set():
                    cap.grab()
                    ok, frame = cap.read()
                    if not ok: break
                    res = _iris_center_norm(frame)
                    if res is not None:
                        nx, ny = res
                        try:
                            self.out_q.put((frame, nx, ny), timeout=0.005)
                        except Exception:
                            pass
            finally:
                cap.release()


# ----------------------------- Engine -----------------------------
@dataclass
class CalibrationReport:
    n_points: int
    rmse_px: Optional[float]
    rmse_cv_px: Optional[float]
    uniformity: float
    width: int
    height: int

class Session:
    def __init__(self):
        self.smooth_x = OneEuroFilter(120, 1.0, 0.01, 1.0)
        self.smooth_y = OneEuroFilter(120, 1.0, 0.01, 1.0)
        self.last_feat: Optional[np.ndarray] = None
        self.last_quality = 0.5
        self.t_prev = None
        self.fps_hist = []
        # calibration buffers
        self.X = []
        self.yx = []
        self.yy = []
        # model
        self.wx = None
        self.wy = None
        self.model_ready = False
        # report
        self.report: Optional[CalibrationReport] = None
        # ui metrics
        self.metrics = {"fps": 0.0, "latency_ms": 0.0}

class GazeEngine:
    def __init__(self, screen_w: int, screen_h: int, extractor: FeatureExtractor):
        self.sw, self.sh = screen_w, screen_h
        self.ext = extractor
        self.sess = Session()
        self.gain = 1.0
        self.gamma = 1.0
        self.deadzone = 0.02

    def _map(self, fv: np.ndarray) -> np.ndarray:
        ex, ey = float(fv[0]), float(fv[1])
        base = np.array([0.5 + (ex - 0.5) * 1.6, 0.5 + (ey - 0.5) * 1.4], dtype=np.float32)
        if self.sess.model_ready and (self.sess.wx is not None):
            pm = _predict_poly(self.sess.wx, self.sess.wy, fv[None, :])[0]
            alpha_x, alpha_y = 0.10, 0.20
            px = (1.0 - alpha_x) * pm[0] + alpha_x * base[0]
            py = (1.0 - alpha_y) * pm[1] + alpha_y * base[1]
        else:
            px, py = base[0], base[1]
        return np.array([px, py], dtype=np.float32)

    def _shape(self, x: float, y: float) -> Tuple[float, float]:
        x = 0.5 + (x - 0.5) * self.gain
        y = 0.5 + (y - 0.5) * self.gain
        if self.gamma != 1.0:
            def _gm(v):
                s = (v - 0.5)
                sign = 1.0 if s >= 0 else -1.0
                return 0.5 + sign * (abs(s) ** self.gamma)
            x, y = _gm(x), _gm(y)
        dz = self.deadzone
        def _dz(v):
            d = v - 0.5
            return 0.5 if abs(d) < dz else v
        x, y = _dz(x), _dz(y)
        x = float(self.sess.smooth_x.filter(min(1.0, max(0.0, x))))
        y = float(self.sess.smooth_y.filter(min(1.0, max(0.0, y))))
        return x, y

    def process_frame(self, frame_bgr):
        t0 = time.time()
        feat = self.ext.extract(frame_bgr)
        if feat is None:
            fv = np.array([0.5, 0.5, 0.5, 0.5, 1.0], dtype=np.float32)
            q = 0.2
        else:
            fv = _feat_vec(feat)
            q = float(feat.get("quality", 0.5))
        self.sess.last_feat = fv
        self.sess.last_quality = q

        for f in (self.sess.smooth_x, self.sess.smooth_y):
            f.mincutoff = max(0.3, 1.6 - 1.2 * q)
            f.beta = 0.01 + 0.12 * (1.0 - q)

        pred = self._map(fv)
        x, y = self._shape(float(pred[0]), float(pred[1]))

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
        self.sess.metrics = {"fps": fps, "latency_ms": latency_ms}
        return x, y

    def calibration_add(self, sx, sy):
        if self.sess.last_feat is None:
            return False
        self.sess.X.append(self.sess.last_feat.astype(np.float32))
        self.sess.yx.append(float(sx)); self.sess.yy.append(float(sy))
        return True

    def calibration_finish(self):
        n = len(self.sess.X)
        if n < 6:
            self.sess.model_ready = False
            self.sess.wx = self.sess.wy = None
            self.sess.report = None
            return {"ok": False, "msg": "ต้องเก็บคาลิเบรทอย่างน้อย 6 จุด"}
        X = np.array(self.sess.X, dtype=np.float32)
        Y = np.stack([np.array(self.sess.yx, dtype=np.float32), np.array(self.sess.yy, dtype=np.float32)], axis=1)
        Phi = _poly_expand(X)
        lam = 1e-3
        self.sess.wx = _fit_ridge_manual(Phi, Y[:,0], lam)
        self.sess.wy = _fit_ridge_manual(Phi, Y[:,1], lam)
        self.sess.model_ready = True

        preds = _predict_poly(self.sess.wx, self.sess.wy, X)
        dx = (preds[:,0] - Y[:,0]) * self.sw
        dy = (preds[:,1] - Y[:,1]) * self.sh
        rmse_train_px = float(np.sqrt(np.mean(dx*dx + dy*dy)))

        # simple 3-fold CV
        def _cv_rmse_px():
            n = len(X)
            if n < 6: return float("inf")
            k = 3
            idx = np.arange(n); np.random.shuffle(idx)
            folds = np.array_split(idx, k)
            acc = []
            for i in range(k):
                te = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
                wx = _fit_ridge_manual(_poly_expand(X[tr]), Y[tr,0], lam)
                wy = _fit_ridge_manual(_poly_expand(X[tr]), Y[tr,1], lam)
                pp = _predict_poly(wx, wy, X[te])
                dx = (pp[:,0] - Y[te,0]) * self.sw
                dy = (pp[:,1] - Y[te,1]) * self.sh
                acc.append(float(np.sqrt(np.mean(dx*dx + dy*dy))))
            return float(np.mean(acc))
        rmse_cv_px = _cv_rmse_px()

        # uniformity ~ convex hull area approx
        pts = Y.copy()
        if len(pts) >= 3:
            pts = pts[np.lexsort((pts[:,1], pts[:,0]))]
            def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
            lower, upper = [], []
            for p in pts:
                while len(lower)>=2 and cross(lower[-2], lower[-1], p) <= 0: lower.pop()
                lower.append(p)
            for p in pts[::-1]:
                while len(upper)>=2 and cross(upper[-2], upper[-1], p) <= 0: upper.pop()
                upper.append(p)
            hull = np.vstack([lower[:-1], upper[:-1]])
            area = 0.0
            for i in range(len(hull)):
                x1,y1 = hull[i]; x2,y2 = hull[(i+1)%len(hull)]
                area += x1*y2 - x2*y1
            uniformity = float(max(0.0, min(1.0, abs(area)/2.0)))
        else:
            uniformity = 0.0

        self.sess.report = CalibrationReport(
            n_points=n, rmse_px=rmse_train_px, rmse_cv_px=rmse_cv_px, uniformity=uniformity, width=self.sw, height=self.sh
        )
        return {"ok": True, "n": n, "rmse_px": rmse_train_px, "rmse_cv_px": rmse_cv_px, "uniformity": uniformity}

    def get_report(self):
        rep = self.sess.report
        if rep is None:
            return {"model_ready": self.sess.model_ready, "rmse_norm": None, "n_samples": len(self.sess.X)}
        return {"model_ready": True, "rmse_px": rep.rmse_px, "rmse_cv_px": rep.rmse_cv_px,
                "uniformity": rep.uniformity, "points": rep.n_points}

    def get_last_metrics(self):
        return dict(self.sess.metrics)


# ----------------------------- App State -----------------------------
class AppState:
    def __init__(self):
        self.mode = "Webcam/MediaPipe"
        self.mirror = False
        self.invert_x = False
        self.invert_y = False
        self.gain = 1.0
        self.gamma = 1.0
        self.deadzone = 0.02
        # screen
        sw, sh = _safe_size()
        self.use_screen_override = False
        self.screen_w = sw
        self.screen_h = sh
        # OPTIONAL: degree per pixel (สำหรับคูณแปลงเป็นองศาเวลาบันทึกรูป)
        self.deg_per_px = 0.0
        # mouse
        self.mouse_enabled = False
        # calibration
        self.calib_overlay = False
        self.targets: List[Tuple[float,float]] = []
        self.idx = 0
        self.dwell_ms = 1000
        self.radius_norm = 0.02
        # ESP32
        self.esp32_url = "http://172.20.10.3:81/stream"
        self.esp_q = None
        self.esp_stop = None
        # shared gaze + metrics
        self.gx = 0.5
        self.gy = 0.5
        self.ui_fps = 0.0
        self.ui_lat = 0.0
        # >>> ADD: countdown config/state <<<
        self.countdown_secs = 3     # จำนวนวิินาทีก่อนเริ่มคาลิเบรต
        self.countdown_active = False
        self.countdown_end = 0.0
        # --- Soundboard (2x3 Thai) ---
        self.soundboard_on = False
        self.sound_labels = ["สวัสดี", "ใช่", "ไม่ใช่", "ขอบคุณ", "ช่วยด้วย", "ไปห้องน้ำ"]
        self.sound_files = [
            "sounds/s1.wav",
            "sounds/s2.wav",
            "sounds/s3.wav",
            "sounds/s4.wav",
            "sounds/s5.wav",
            "sounds/s6.wav",
        ]
        self.sound_dwell_ms = 3000      # dwell 3s before speaking
        self.sound_cooldown_ms = 1500   # cooldown to avoid repeats
        self._sound_current_idx = None
        self._sound_start = 0.0
        self._sound_last_play = [-1.0]*6



if "APP" not in st.session_state:
    st.session_state["APP"] = AppState()
APP: AppState = st.session_state["APP"]


# ----------------------------- Video Processor -----------------------------
class Processor(VideoProcessorBase):
    def __init__(self):
        self.engine = None
        self.mouse = MouseController(False)
        self.calib_hold_start = None
        self.pool = []

    def _ensure(self):
        if self.engine is not None:
            return
        sw = APP.screen_w if APP.use_screen_override else _safe_size()[0]
        sh = APP.screen_h if APP.use_screen_override else _safe_size()[1]
        extractor = FeatureExtractor(use_mediapipe=(APP.mode == "Webcam/MediaPipe"))
        self.engine = GazeEngine(sw, sh, extractor)

    def recv(self, frame: VideoFrame) -> VideoFrame:


            # --- ESP32 mode: get frame from queue ---
            if APP.mode.startswith("ESP32") and APP.esp_q is not None and cv2 is not None:
                img_esp = None; nx = ny = None
                try:
                    item = APP.esp_q.get_nowait()
                    if isinstance(item, tuple) and len(item) == 3:
                        img_esp, nx, ny = item
                except Exception:
                    pass
                if img_esp is not None:
                    img = img_esp
                else:
                    # fallback: black image
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                img = frame.to_ndarray(format="bgr24")
                if APP.mode == "Webcam/MediaPipe" and APP.mirror and cv2 is not None:
                    img = cv2.flip(img, 1)

            self._ensure()

            # update engine params
            self.engine.gain = float(APP.gain)
            self.engine.gamma = float(APP.gamma)
            self.engine.deadzone = float(APP.deadzone)

            # process per frame
            if APP.mode == "Webcam/MediaPipe":
                x, y = self.engine.process_frame(img)
            else:
                if APP.esp_q is not None:
                    try:
                        nx, ny = APP.esp_q.get_nowait()
                        fv = np.array([nx, ny, 0.5, 0.5, 1.0], dtype=np.float32)
                        self.engine.sess.last_feat = fv
                        x, y = self.engine._map(fv)
                        x, y = self.engine._shape(float(x), float(y))
                        self.engine.sess.last_quality = 0.5
                        m = self.engine.get_last_metrics()
                        m["latency_ms"] = (m.get("latency_ms", 16.0) * 0.9) + 1.6
                        self.engine.sess.metrics = m
                    except Exception:
                        x, y = APP.gx, APP.gy
                else:
                    x, y = APP.gx, APP.gy

            if APP.invert_x: x = 1.0 - x
            if APP.invert_y: y = 1.0 - y

            APP.gx, APP.gy = float(x), float(y)
            m = self.engine.get_last_metrics()
            APP.ui_fps = float(m.get("fps", 0.0)); APP.ui_lat = float(m.get("latency_ms", 0.0))

            # >>> ADD: Countdown overlay <<<
            if APP.countdown_active and cv2 is not None:
                remaining = APP.countdown_end - time.time()
                if remaining <= 0:
                    # countdown จบ → เริ่มคาลิเบรตจริง
                    APP.countdown_active = False
                    APP.calib_overlay = True
                    self.calib_hold_start = None
                    self.pool = []
                else:
                    # วาด overlay 3..2..1..GO แล้ว return เฟรมนี้ไปก่อน
                    h, w = img.shape[:2]
                    overlay = img.copy()
                    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
                    sec = int(math.ceil(remaining))
                    # ถ้าเหลือน้อยกว่า 0.3 วินาที ให้ขึ้น GO
                    if 0 < remaining < 0.3:
                        text = "GO"
                    else:
                        text = str(sec)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 4.0
                    thick = 8
                    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
                    cv2.putText(img, text, (w // 2 - tw // 2, h // 2 + th // 3),
                                font, scale, (0, 200, 255), thick, cv2.LINE_AA)
                    cv2.putText(img, "Starting calibration...",
                                (max(20, w // 2 - 280), min(h - 20, h // 2 + th + 40)),
                                font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                    # ยังไม่เริ่มคาลิเบรต/ไม่เก็บจุดในรอบนี้
                    return VideoFrame.from_ndarray(img, format="bgr24")

            # Calibration overlay + data collection
            if APP.calib_overlay and cv2 is not None:
                tx, ty = APP.targets[APP.idx] if (0 <= APP.idx < len(APP.targets)) else (0.5, 0.5)
                h, w = img.shape[:2]
                gx_i, gy_i = int(tx*w), int(ty*h)
                cv2.circle(img, (gx_i, gy_i), 18, (0,170,255), 2)

                now = time.time()
                if self.engine.sess.last_feat is not None:
                    if self.calib_hold_start is None:
                        self.calib_hold_start = now; self.pool = []
                    self.pool.append(self.engine.sess.last_feat.copy())
                    elapsed = (now - self.calib_hold_start)*1000.0
                else:
                    self.calib_hold_start = None; self.pool = []; elapsed = 0.0
                frac = max(0.0, min(1.0, elapsed / float(max(1, APP.dwell_ms))))
                end_angle = int(360 * frac)
                cv2.ellipse(img, (gx_i, gy_i), (22, 22), 0, 0, end_angle, (0,170,255), 3)
                cv2.putText(img, f"Calibration {APP.idx+1}/{len(APP.targets)}",
                            (24,48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

                if frac >= 1.0 and 0 <= APP.idx < len(APP.targets):
                    if len(self.pool) >= 5:
                        self.engine.sess.last_feat = np.mean(np.stack(self.pool, axis=0), axis=0).astype(np.float32)
                    self.engine.calibration_add(tx, ty)
                    APP.idx += 1; self.calib_hold_start = None; self.pool = []
                    if APP.idx >= len(APP.targets):
                        rep = self.engine.calibration_finish()
                        APP.calib_overlay = False
                        txt = ("Calibration OK · "
                               f"RMSE={rep.get('rmse_px',0):.0f}px · "
                               f"CV={rep.get('rmse_cv_px',0):.0f}px · "
                               f"U={rep.get('uniformity',0):.2f}")
                        cv2.rectangle(img, (10,10), (10+900, 10+40), (0,0,0), -1)
                        cv2.putText(img, txt, (18,38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,230,255), 2)

            # Mouse pointer only
            self.mouse.set_enable(APP.mouse_enabled)
            self.mouse.update(APP.gx, APP.gy)

            # Crosshair + metrics
            if cv2 is not None:
                h, w = img.shape[:2]
                cx, cy = int(APP.gx*w), int(APP.gy*h)
                cv2.circle(img, (cx, cy), 8, (0,255,0), 2)
                cv2.line(img, (cx-15, cy), (cx+15, cy), (0,255,0), 1)
                cv2.line(img, (cx, cy-15), (cx, cy+15), (0,255,0), 1)
                cv2.putText(img, f"FPS {APP.ui_fps:4.1f} | Latency {APP.ui_lat:4.1f} ms",
                    (w - 460, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            
            # --- Soundboard overlay (2x3) ---
            if APP.soundboard_on and (not APP.calib_overlay) and (not APP.countdown_active) and cv2 is not None:
                rows, cols = 2, 3
                H, W = img.shape[:2]
                hover_idx = None

                # draw grid + labels
                for r in range(rows):
                    for c in range(cols):
                        idx = r*cols + c
                        x0, x1 = c/cols, (c+1)/cols
                        y0, y1 = r/rows, (r+1)/rows
                        pt1 = (int(x0*W), int(y0*H))
                        pt2 = (int(x1*W), int(y1*H))
                        inside = _inside_rect(APP.gx, APP.gy, (x0, y0, x1, y1))
                        color = (40, 40, 40); thick = 2
                        if inside:
                            color = (0, 200, 255); thick = 4
                            hover_idx = idx
                        cv2.rectangle(img, pt1, pt2, color, thick)
                        label = APP.sound_labels[idx] if idx < len(APP.sound_labels) else f"Button {idx+1}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                        cx = int((x0 + x1)/2 * W) - tw//2
                        cy = int((y0 + y1)/2 * H) + th//3
                        cv2.putText(img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3, cv2.LINE_AA)

                # dwell logic → play sound
                now = time.time()
                if hover_idx is None:
                    APP._sound_current_idx = None
                    APP._sound_start = 0.0
                else:
                    if APP._sound_current_idx != hover_idx:
                        APP._sound_current_idx = hover_idx
                        APP._sound_start = now
                    elapsed_ms = (now - APP._sound_start) * 1000.0

                    # progress arc at top of the hovered button
                    c = hover_idx % cols
                    r = hover_idx // cols
                    x0, x1 = c/cols, (c+1)/cols
                    y0 = r/rows
                    pcx = int((x0 + x1)/2 * W)
                    pcy = int(y0 * H + 40)
                    frac = max(0.0, min(1.0, elapsed_ms / float(max(1, APP.sound_dwell_ms))))
                    cv2.ellipse(img, (pcx, pcy), (28,28), 0, 0, int(360*frac), (0,200,255), 4)

                    last = APP._sound_last_play[hover_idx] if hover_idx < len(APP._sound_last_play) else -1.0
                    if (elapsed_ms >= APP.sound_dwell_ms) and ((now - last) >= APP.sound_cooldown_ms/1000.0):
                        path = APP.sound_files[hover_idx] if hover_idx < len(APP.sound_files) else None
                        if path:
                            try:
                                import threading as _th
                                _th.Thread(target=play_sound_file, args=(path,), daemon=True).start()
                            except Exception:
                                pass
                        if hover_idx < len(APP._sound_last_play):
                            APP._sound_last_play[hover_idx] = now
                        # require moving gaze away before repeating same button
                        APP._sound_current_idx = None
                        APP._sound_start = 0.0
# --- record metrics each frame ---
            sw = APP.screen_w if APP.use_screen_override else _safe_size()[0]
            sh = APP.screen_h if APP.use_screen_override else _safe_size()[1]
            pred_px = (int(APP.gx*sw), int(APP.gy*sh))
            target_px = None; point_id = None
            if APP.calib_overlay and (0 <= APP.idx < len(APP.targets)):
                tx, ty = APP.targets[APP.idx]
                target_px = (int(tx*sw), int(ty*sh))
                point_id = int(APP.idx)
            METRICS.record(latency_ms=APP.ui_lat, pred_px=pred_px, target_px=target_px, point_id=point_id)

            return VideoFrame.from_ndarray(img, format="bgr24")


# ----------------------------- Sidebar -----------------------------
def sidebar():
    st.sidebar.title("⚙️ Controls")
    APP.mode = st.sidebar.selectbox("Mode", ["Webcam/MediaPipe", "ESP32 (HTTP stream)"], index=0)
    APP.mouse_enabled = st.sidebar.toggle("Enable Mouse Control (pointer only — no click)", value=APP.mouse_enabled)

    st.sidebar.subheader("🖥️ Display / Screen")
    APP.use_screen_override = st.sidebar.checkbox("Override screen size (px)", value=APP.use_screen_override)
    colsw, colsh = st.sidebar.columns(2)
    APP.screen_w = int(colsw.number_input("Width", min_value=320, max_value=10000, value=int(APP.screen_w)))
    APP.screen_h = int(colsh.number_input("Height", min_value=240, max_value=10000, value=int(APP.screen_h)))
    sw_os, sh_os = _safe_size()
    st.sidebar.caption(f"OS reports: {sw_os}×{sh_os}px")

    st.sidebar.subheader("🎛 Shaping")
    APP.gain = float(st.sidebar.slider("Gain", 0.5, 2.5, APP.gain, 0.05))
    APP.gamma = float(st.sidebar.slider("Gamma", 0.5, 2.0, APP.gamma, 0.05))
    APP.deadzone = float(st.sidebar.slider("Deadzone", 0.0, 0.1, APP.deadzone, 0.005))

    st.sidebar.subheader("🪞 Mirror / Invert")
    APP.mirror = st.sidebar.checkbox("Mirror webcam image", value=APP.mirror)
    APP.invert_x = st.sidebar.checkbox("Invert X (x→1−x)", value=APP.invert_x)
    APP.invert_y = st.sidebar.checkbox("Invert Y (y→1−y)", value=APP.invert_y)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Optional: Degrees")
    APP.deg_per_px = float(st.sidebar.number_input("deg per px (คูณเพื่อแปลงค่า px→deg)", min_value=0.0, max_value=1.0, value=float(APP.deg_per_px), step=0.001, format="%.3f"))
    st.sidebar.caption("ถ้าไม่ทราบ ปล่อยไว้ 0.000 ได้ (จะไม่สร้างกราฟหน่วยองศา)")
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔊 Thai Soundboard (2×3)")
    APP.soundboard_on = st.sidebar.toggle("Enable gaze soundboard overlay", value=APP.soundboard_on)
    APP.sound_dwell_ms = int(st.sidebar.slider("Dwell to speak (ms)", 1000, 5000, APP.sound_dwell_ms, 250))
    APP.sound_cooldown_ms = int(st.sidebar.slider("Cooldown (ms)", 500, 5000, APP.sound_cooldown_ms, 250))
    for i in range(6):
        c1, c2 = st.sidebar.columns([1,2])
        APP.sound_labels[i] = c1.text_input(f"Label {i+1}", APP.sound_labels[i], key=f"sb_lbl_{i}")
        APP.sound_files[i] = c2.text_input(f"File {i+1}", APP.sound_files[i], key=f"sb_file_{i}")
    st.sidebar.caption("หมายเหตุ: เสียงจะออกทางเครื่องที่รันแอป Streamlit (ฝั่งเซิร์ฟเวอร์)")


    if APP.mode.startswith("ESP32"):
        st.sidebar.subheader("ESP32-CAM")
        APP.esp32_url = st.sidebar.text_input("Stream URL", value=APP.esp32_url)
        c1, c2 = st.sidebar.columns(2)
        start = c1.button("Start")
        stop = c2.button("Stop")
        if start:
            if APP.esp_stop is None:
                APP.esp_q = queue.Queue(maxsize=4)
                APP.esp_stop = threading.Event()
                t = ESP32Reader(APP.esp32_url, APP.esp_q, APP.esp_stop)
                t.start(); st.toast("ESP32 reader started")
        if stop and APP.esp_stop is not None:
            APP.esp_stop.set(); APP.esp_stop = None
            st.toast("ESP32 reader stopped")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧭 Calibration (9 points)")
    st.sidebar.caption("จำนวนจุดคาลิเบรต: 9 (ตามเอกสาร)")
    # >>> ADD: ตั้งค่า countdown <<<
    APP.countdown_secs = int(st.sidebar.slider("Countdown before start (s)", 0, 5, APP.countdown_secs, 1))
    APP.dwell_ms = st.sidebar.slider("Dwell per target (ms)", 400, 2000, APP.dwell_ms, 50)
    if st.sidebar.button("Start Calibration"):
        grid = [
            (0.50, 0.50),
            (0.30, 0.50), (0.70, 0.50),
            (0.50, 0.30), (0.50, 0.70),
            (0.20, 0.20), (0.80, 0.20), (0.20, 0.80), (0.80, 0.80),
        ]
        APP.targets = grid
        APP.idx = 0
        # >>> CHANGED: เริ่มจาก countdown ก่อน <<<
        APP.calib_overlay = False
        APP.countdown_active = True
        APP.countdown_end = time.time() + max(0, APP.countdown_secs)
        sw, sh = _safe_size()
        APP.radius_norm = max(0.012, 40 / max(sw, sh))
        st.toast(f"เริ่มคาลิเบรทใน {APP.countdown_secs} วินาที…")


# ----------------------------- Overlay CSS -----------------------------
def _overlay_css(ctx):
    if APP.calib_overlay:
        st.markdown("""
        <style>
          section[data-testid="stSidebar"] { display:none!important; }
          header, footer { display:none!important; }
          video { position:fixed!important; inset:0!important; width:100vw!important; height:100vh!important;
                  object-fit:cover!important; z-index:9999!important; }
        </style>
        """, unsafe_allow_html=True)


# ----------------------------- Main UI -----------------------------
def main():
    st.set_page_config(page_title="Vision Mouse", page_icon="👁️", layout="wide")
    st.title("👁️ Vision Mouse — Eye-controlled pointer (Webcam / ESP32)")
    st.caption("โปรเจกต์นี้ขยับเมาส์อย่างเดียว (ไม่คลิก) · หากพรีวิวไม่ขึ้น ตรวจสิทธิ์กล้อง และติดตั้ง opencv-python / mediapipe")

    sidebar()
    ctx = webrtc_streamer(
        key="vm-stream", mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video":{"width":{"ideal":640},"height":{"ideal":480},"frameRate":{"ideal":60,"min":30}}, "audio":False},
        async_processing=False, video_processor_factory=Processor,
    )
    _overlay_css(ctx)

    # Metrics strip
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Mode", APP.mode)
    with c2: st.metric("Mouse", "ON" if APP.mouse_enabled else "OFF")
    with c3: st.metric("FPS", f"{APP.ui_fps:.1f}")
    with c4: st.metric("Latency (ms)", f"{APP.ui_lat:.1f}")
    st.caption(f"📦 Samples collected: {len(METRICS.samples)}")

    # Calibration report
    if ctx and ctx.video_processor and ctx.video_processor.engine:
        eng: GazeEngine = ctx.video_processor.engine
        rep = eng.get_report()
        st.subheader("Calibration Report")
        if not rep.get("model_ready", False):
            st.info("Model: **Uncalibrated** — โปรเจกต์นี้ขยับเมาส์อย่างเดียว (ไม่คลิก)")
        else:
            st.markdown(
                f"- RMSE(train): **{rep['rmse_px']:.0f}px** · RMSE(CV): **{rep['rmse_cv_px']:.0f}px** · "
                f"Uniformity: **{rep['uniformity']:.2f}** · Points: **{rep['points']}**"
            )

    # --- Charts: Global + Per-point (บนหน้าเว็บ) ---
    if pd is None:
        st.warning("ต้องติดตั้ง pandas เพื่อตาราง/กราฟเมตริก:  pip install pandas")
    else:
        st.subheader("Metrics (Global)")
        df_global, df_points = METRICS.summarize()
        c1, c2 = st.columns([2,1])
        with c1:
            st.bar_chart(df_global, x="metric", y="value", height=260)
        with c2:
            st.dataframe(df_global.style.format({"value": "{:.2f}"}), use_container_width=True)

        st.subheader("Metrics per Calibration Point")
        if df_points.empty:
            st.info("ยังไม่มีข้อมูล per-point — รวบรวมขณะคาลิเบรตหรือรอบ evaluation")
        else:
            st.caption("แต่ละแท่งคำนวณจากตัวอย่างที่มี target จริงในจุดนั้น (n ≥ 1)")
            st.write("**Jitter (px) ต่อจุด**")
            st.bar_chart(df_points[["point_id","jitter_px"]].set_index("point_id"), height=220)
            st.write("**RMSE (px) ต่อจุด**")
            st.bar_chart(df_points[["point_id","rmse_px"]].set_index("point_id"), height=220)
            st.write("**MAE (px) ต่อจุด**")
            st.bar_chart(df_points[["point_id","mae_px"]].set_index("point_id"), height=220)
            with st.expander("ดูตารางตัวเลขต่อจุด"):
                st.dataframe(
                    df_points.rename(columns={"point_id":"Point","jitter_px":"Jitter(px)","rmse_px":"RMSE(px)","mae_px":"MAE(px)","n":"N"})
                             .style.format({"Jitter(px)":"{:.2f}","RMSE(px)":"{:.2f}","MAE(px)":"{:.2f}"}),
                    use_container_width=True
                )

        cA, cB = st.columns([1,3])
        with cA:
            if st.button("💾 Save metrics to CSV"):
                p1, p2 = METRICS.save_csv(prefix="metrics")
                if p1 and p2:
                    st.success(f"Saved: {p1}, {p2}")

        # ---------- NEW: ปุ่มบันทึกกราฟ PNG สำหรับโหมด Webcam ----------
        with cB:
            if plt is None:
                st.warning("ต้องติดตั้ง matplotlib เพื่อบันทึกรูป PNG:  pip install matplotlib")
            else:
                if st.button("📷 Export Webcam charts (PNG)"):
                    if len(METRICS.samples) == 0:
                        st.error("ยังไม่มีตัวอย่างเฟรม — เริ่มสตรีม/ขยับสายตาให้มีเฟรมก่อน")
                    else:
                        try:
                            paths = METRICS.export_webcam_charts(
                                export_dir="exports",
                                deg_per_px=(APP.deg_per_px if APP.deg_per_px > 0 else None)
                            )
                            st.success("Saved PNG: " + ", ".join(paths))
                        except Exception as e:
                            st.error(f"Export failed: {e}")

    st.markdown("""
**ขั้นตอนใช้งานย่อย**
- กด **Start Calibration** เพื่อคาลิเบรต 9 จุด (มีวงแหวนบอกเวลาค้าง)
- จบคาลิเบรตแล้วจะมีรายงาน **RMSE / CV / Uniformity / Points**
- แถว **Metrics** ด้านบนและกราฟด้านล่างจะแสดง **Latency / Jitter / MAE / RMSE** (รวม และต่อจุด)
- ปุ่ม **Export Webcam charts (PNG)** จะบันทึกกราฟ 6 แบบลงโฟลเดอร์ `exports/`
- ใส่ค่า **deg per px** ใน Sidebar หากต้องการกราฟหน่วยองศาใน Summary Accuracy
- โหมดนี้ **pointer only — ไม่คลิก** ตามเอกสาร Project_Final
    """)

if __name__ == "__main__":
    main()