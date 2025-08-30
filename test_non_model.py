"""
test_non_model.py
👁️ Vision Mouse — Pointer-only eye control (Webcam/MediaPipe + ESP32-CAM)
- 9-point calibration (no auto-extend)
- One-Euro filter smoothing
- Metrics recorder + bar charts (Streamlit) และปุ่มบันทึกกราฟเป็นไฟล์ PNG (สำหรับโหมด Webcam)
- Evaluation CSV export

Run: streamlit run test3.py
"""


# --- Standard Library Imports ---
from __future__ import annotations
import math
import time
import threading as _th
import queue
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

# --- Third-party Imports ---
import numpy as np
import streamlit as st


# --- Optional Dependencies ---
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

# --- Streamlit WebRTC & AV ---
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from av import VideoFrame

# --- Other Imports ---
from collections import deque
import pygame


# ----------------------------- Helpers -----------------------------
def _safe_size() -> Tuple[int, int]:
    """Get screen size safely, fallback to 1920x1080 if pyautogui is unavailable."""
    if pyautogui:
        try:
            return pyautogui.size()
        except Exception:
            pass
    return 1920, 1080



def play_sound_file(path: str) -> bool:
    """Play an audio file non-blocking on the server machine. Returns True if a backend accepted the file."""
    try:
        # ใช้ simpleaudio สำหรับ .wav
        if sa and path.lower().endswith((".wav", ".wave")):
            wave_obj = sa.WaveObject.from_wave_file(path)
            wave_obj.play()
            return True
        # ใช้ pygame สำหรับ .mp3
        elif path.lower().endswith('.mp3'):
            try:
                pygame.mixer.init()
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                return True
            except Exception as e:
                print(f"pygame play failed: {e}")
        # fallback: playsound (อาจ error ได้)
        elif _playsound:
            _th.Thread(target=_playsound, args=(path,), daemon=True).start()
            return True
    except Exception as e:
        print("play_sound_file failed:", e)
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

# ----------------------------- CalibrationReport -----------------------------
class CalibrationReport:
    def __init__(self, n_points=0, rmse_px=0.0, rmse_cv_px=0.0, uniformity=1.0, width=1, height=1):
        self.n_points = n_points
        self.rmse_px = rmse_px
        self.rmse_cv_px = rmse_cv_px
        self.uniformity = uniformity
        self.width = width
        self.height = height

# ----------------------------- Session -----------------------------
# ----------------------------- Session -----------------------------
class Session:
    def __init__(self):
        # buffers (เดิม)
        self.X = []
        self.yx = []
        self.yy = []

        # calibration (ใหม่)
        self.calib_features = []   # [ex, ey, yaw, pitch]
        self.calib_targets  = []   # [sx, sy]

        # running state
        self.last_feat: Optional[np.ndarray] = None   # [ex,ey,yaw,pitch]
        self.last_quality: float = 0.0

        # smoothing & perf
        self.smooth_x = OneEuroFilter(freq=120.0, mincutoff=1.3, beta=0.02, dcutoff=1.2)
        self.smooth_y = OneEuroFilter(freq=120.0, mincutoff=1.3, beta=0.02, dcutoff=1.2)
        self.fps_hist: list[float] = []
        self.t_prev: Optional[float] = None

        # model/report
        self.affine: Optional[np.ndarray] = None      # (5,2)
        self.model_ready: bool = False
        self.report: Optional['CalibrationReport'] = None

        # metrics
        self.metrics = {"fps": 0.0, "latency_ms": 0.0}

# ----------------------------- Feature Extractor -----------------------------
class FeatureExtractor:
    LEFT_EYE_IDS  = [33, 133, 159, 145]
    RIGHT_EYE_IDS = [362, 263, 386, 374]
    LEFT_IRIS_IDS  = [468, 469, 470, 471, 472]
    RIGHT_IRIS_IDS = [473, 474, 475, 476, 477]

    def __init__(self, use_mediapipe: bool = True):
        self.use_mediapipe = use_mediapipe and (mp is not None) and (cv2 is not None)
        self.face_landmarks = None
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
        # baselines
        self.earL_base = None
        self.earR_base = None

    def close(self):
        if self.mesh is not None:
            self.mesh.close()

    @staticmethod
    def _avg_xy(ids, pts):
        xs = [pts[i][0] for i in ids if i < len(pts)]
        ys = [pts[i][1] for i in ids if i < len(pts)]
        if not xs or not ys: return None
        return (sum(xs)/len(xs), sum(ys)/len(ys))

    @staticmethod
    def _eye_box_metrics(ids, pts):
        xs = [pts[i][0] for i in ids if i < len(pts)]
        ys = [pts[i][1] for i in ids if i < len(pts)]
        if not xs or not ys: return None
        x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
        w = max(6.0, x1-x0); h = max(2.0, y1-y0)
        ear = h / max(6.0, w)  # eye-aspect ratio (ยิ่งเล็กยิ่งปิด)
        return (x0, x1, y0, y1, w, h, ear)

    @staticmethod
    def _norm_in_box(p, box):
        if (p is None) or (box is None): return (0.5, 0.5)
        x0, x1, y0, y1, w, h, _ = box
        nx = (p[0]-x0)/w; ny = (p[1]-y0)/h
        return (min(1.0, max(0.0, nx)), min(1.0, max(0.0, ny)))

    def extract(self, frame_bgr) -> Optional[dict]:
        # fallback helper
        def _fallback(q=0.2, open_=False):
            return dict(
                eye_cx_norm=0.5, eye_cy_norm=0.5,
                face_cx_norm=0.5, face_cy_norm=0.5,
                eye_open=bool(open_), quality=float(q),
                yaw=0.0, pitch=0.0
            )

        if frame_bgr is None:
            return _fallback(0.2, True)

        if not self.use_mediapipe or self.mesh is None:
            # ไม่มี mediapipe → คืน fallback ชัดเจน (อย่า return None)
            return _fallback(0.1, False)

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(frame_rgb)

        if not res.multi_face_landmarks:
            self.face_landmarks = None
            return _fallback(0.05, False)

        lm = res.multi_face_landmarks[0]
        self.face_landmarks = lm
        lmk = lm.landmark
        pts = [(p.x * w, p.y * h, p.z) for p in lmk]

        # --- head pose (แบบคร่าวๆจาก mediapipe index) ---
        dx = lmk[263].x - lmk[33].x
        dy = lmk[263].y - lmk[33].y
        yaw = math.degrees(math.atan2(dy, dx))
        pitch = math.degrees(math.atan2(lmk[1].y - lmk[168].y, lmk[1].z - lmk[168].z))

        # --- ตาดำ + กล่องตา + normalize ภายในกล่องตา ---
        l_iris = self._avg_xy(self.LEFT_IRIS_IDS, pts)
        r_iris = self._avg_xy(self.RIGHT_IRIS_IDS, pts)
        l_box  = self._eye_box_metrics(self.LEFT_EYE_IDS, pts)
        r_box  = self._eye_box_metrics(self.RIGHT_EYE_IDS, pts)

        l_n = self._norm_in_box(l_iris, l_box) if (l_iris and l_box) else (0.5, 0.5)
        r_n = self._norm_in_box(r_iris, r_box) if (r_iris and r_box) else (0.5, 0.5)

        earL = l_box[6] if l_box else None
        earR = r_box[6] if r_box else None
        eye_open = bool((earL or 0.0) > 0.20 or (earR or 0.0) > 0.20)

        # confidence ของแต่ละตา: blend จาก EAR และขนาดกล่อง (เชิงประจักษ์)
        def _eye_conf(ear, base, box, iris_ok):
            if (ear is None) or (box is None) or (not iris_ok):
                return 0.0
            # bootstrap baseline จากค่าเปิดตาในช่วงแรกๆ
            if base is None:
                base = 0.26
            ratio = max(0.0, min(1.4, ear / max(1e-6, 0.8*base)))
            _, _, _, _, bw, bh, _ = box
            area_norm = (bw*bh) / max(1.0, (w*h))
            area_boost = min(1.0, area_norm / 0.02)
            return max(0.0, min(1.0, 0.7*ratio + 0.3*area_boost))

        cL = _eye_conf(earL, self.earL_base, l_box, l_iris is not None)
        cR = _eye_conf(earR, self.earR_base, r_box, r_iris is not None)

        # update baseline แบบ EMA
        for ear, attr in [(earL, "earL_base"), (earR, "earR_base")]:
            if ear is not None:
                cur = getattr(self, attr)
                setattr(self, attr, 0.9*cur + 0.1*ear if cur is not None else ear)

        # combine ex/ey (ถ่วงน้ำหนักด้วยความเชื่อมั่นของตา)
        if (cL + cR) <= 1e-6:
            ex, ey = 0.5, 0.5
        else:
            ex = float((l_n[0]*cL + r_n[0]*cR) / (cL + cR))
            ey = float((l_n[1]*cL + r_n[1]*cR) / (cL + cR))

        # ศูนย์กลางหน้า (เผื่อใช้)
        face_ids = [1, 9, 152, 33, 263]
        fxs = [pts[i][0] for i in face_ids if i < len(pts)]
        fys = [pts[i][1] for i in face_ids if i < len(pts)]
        fcx = float(sum(fxs)/len(fxs)/max(1, w)) if fxs else 0.5
        fcy = float(sum(fys)/len(fys)/max(1, h)) if fys else 0.5

        # คุณภาพรวม: max(conf) * margin (กันหลุดมุม)
        margin = min(ex, 1-ex, ey, 1-ey)
        quality = max(cL, cR) * (0.6 + 0.4*max(0.0, min(1.0, margin*2)))

        return dict(
            eye_cx_norm=ex, eye_cy_norm=ey,
            face_cx_norm=fcx, face_cy_norm=fcy,
            eye_open=eye_open, quality=float(quality),
            yaw=float(yaw), pitch=float(pitch),
        )

# ----------------------------- AutoGain & Math Utils -----------------------------
class AutoGain:
    def __init__(self, target=0.48, window_sec=2.5, ema_alpha=0.05):
        self.target = float(target)
        self.window_sec = float(window_sec)
        self.ema_alpha = float(ema_alpha)
        self.samples = deque()
        self.gain = 1.0
        self.frozen = False

    def reset(self):
        self.samples.clear()
        self.gain = 1.0

    def set_frozen(self, frozen: bool):
        self.frozen = bool(frozen)

    def update(self, abs_val, now_ts):
        self.samples.append((now_ts, float(abs_val)))
        cut = now_ts - self.window_sec
        while self.samples and self.samples[0][0] < cut:
            self.samples.popleft()
        if self.frozen or len(self.samples) < 8:
            return self.gain
        vals = [v for _, v in self.samples]
        vals.sort()
        idx = int(0.95 * (len(vals)-1))
        perc = max(1e-4, vals[idx])
        target_gain = self.target / perc
        self.gain = (1.0 - self.ema_alpha)*self.gain + self.ema_alpha*target_gain
        return self.gain

def clamp01(x: float) -> float:
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return x

def signed_gamma(u: float, gamma: float) -> float:
    v = u - 0.5
    s = 1.0 if v >= 0 else -1.0
    a = abs(v)
    a_g = a ** max(1e-6, gamma)
    return clamp01(0.5 + s * a_g)

class GazeEngine:
    def process_frame_new(self, frame_bgr):
        # ตัวอย่าง stub: return process_frame เดิม
        return self.process_frame(frame_bgr)
    def __init__(self, screen_w: int, screen_h: int, extractor: FeatureExtractor):
        # head-pose compensation (ตัวคูณ/สเกล)
        self.k_yaw = 0.35;  self.c_yaw = 30.0   # deg scale
        self.k_pitch = 0.30; self.c_pitch = 25.0
        # bias/manual
        self.bias_x = 0.0; self.bias_y = 0.0
        # auto-gain (ต่อแกน)
        self.auto_x = AutoGain(target=0.48, window_sec=2.5, ema_alpha=0.05)
        self.auto_y = AutoGain(target=0.48, window_sec=2.5, ema_alpha=0.05)
        self.autogain_frozen = False
        # shaping
        self.gamma_x = 1.15; self.gamma_y = 1.15
        self.gain = 1.0
        self.gamma = 1.0
        self.deadzone = 0.02

        self.sw, self.sh = screen_w, screen_h
        self.ext = extractor
        self.sess = Session()

    # ------- public controls (เรียกจาก Processor) -------
    def set_comp_params(self, k_yaw, c_yaw, k_pitch, c_pitch):
        self.k_yaw = float(k_yaw); self.c_yaw = max(1.0, float(c_yaw)*100) if c_yaw <= 1 else float(c_yaw)
        self.k_pitch = float(k_pitch); self.c_pitch = max(1.0, float(c_pitch)*100) if c_pitch <= 1 else float(c_pitch)

    def set_bias(self, bx, by):
        self.bias_x = float(bx); self.bias_y = float(by)

    def reset_auto_gains(self):
        self.auto_x.reset(); self.auto_y.reset()

    def get_last_metrics(self):
        return dict(self.sess.metrics)

    # ------- internals -------
    def _apply_pose_comp(self, ex, ey, yaw, pitch):
        # ชดเชยทิศทางศีรษะให้ ex/ey (linear scale + clamp)
        offx = self.k_yaw   * (yaw   / self.c_yaw)
        offy = self.k_pitch * (-pitch / self.c_pitch)
        ex = 0.5 + (ex - 0.5) + offx + self.bias_x
        ey = 0.5 + (ey - 0.5) + offy + self.bias_y
        return float(min(1.0, max(0.0, ex))), float(min(1.0, max(0.0, ey)))

    def _auto_gain(self, ex, ey, q):
        # ปรับกำลังต่อแกนจากสถิติล่าสุด (ยิ่งมองไกลศูนย์บ่อย → ลด gain เพื่อกันสะบัด)
        now = time.time()
        if not self.autogain_frozen:
            self.auto_x.update(abs(ex - 0.5), now)
            self.auto_y.update(abs(ey - 0.5), now)
        gx = self.auto_x.gain; gy = self.auto_y.gain
        ex = 0.5 + (ex - 0.5) * gx
        ey = 0.5 + (ey - 0.5) * gy
        return ex, ey

    def _shape(self, x: float, y: float, q: float) -> Tuple[float, float]:
        # global gain/gamma + deadzone + OneEuro
        x = 0.5 + (x - 0.5) * self.gain
        y = 0.5 + (y - 0.5) * self.gain

        if self.gamma != 1.0:
            def _gm(v, g):
                s = (v - 0.5); sign = 1.0 if s >= 0 else -1.0
                return 0.5 + sign * (abs(s) ** g)
            x = _gm(x, self.gamma); y = _gm(y, self.gamma)

        # deadzone
        dz = self.deadzone
        def _dz(v):
            d = v - 0.5
            return 0.5 if abs(d) < dz else v
        x, y = _dz(x), _dz(y)

        # ปรับความหนืดตามคุณภาพ (คุณภาพต่ำ → หนืดขึ้นเล็กน้อย)
        for f in (self.sess.smooth_x, self.sess.smooth_y):
            f.mincutoff = max(0.3, 1.6 - 1.2 * q)
            f.beta = 0.01 + 0.12 * (1.0 - q)

        x = float(self.sess.smooth_x.filter(min(1.0, max(0.0, x))))
        y = float(self.sess.smooth_y.filter(min(1.0, max(0.0, y))))
        return x, y

    def _map(self, fv: np.ndarray) -> np.ndarray:
        # affine mapping: [ex, ey, yaw, pitch, 1] @ A → [px, py]
        ex, ey, yaw, pitch = [float(v) for v in fv]
        # pose compensation + bias
        ex, ey = self._apply_pose_comp(ex, ey, yaw, pitch)
        # auto-gain
        ex, ey = self._auto_gain(ex, ey, self.sess.last_quality)

        if self.sess.model_ready and (self.sess.affine is not None):
            X_aug = np.array([ex, ey, yaw, pitch, 1.0], dtype=np.float32)  # (5,)
            px, py = (X_aug @ self.sess.affine).tolist()
        else:
            # fallback mapping ก่อนคาลิเบรต
            px = 0.5 + (ex - 0.5) * 1.6
            py = 0.5 + (ey - 0.5) * 1.4

        return np.array([clamp01(px), clamp01(py)], dtype=np.float32)

    def process_frame(self, frame_bgr):
        t0 = time.time()
        feat = self.ext.extract(frame_bgr)

        if feat is None:
            fv = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32); q = 0.2
        else:
            fv = np.array([
                float(feat.get("eye_cx_norm", 0.5)),
                float(feat.get("eye_cy_norm", 0.5)),
                float(feat.get("yaw", 0.0)),
                float(feat.get("pitch", 0.0)),
            ], dtype=np.float32)
            q = float(feat.get("quality", 0.5))

        self.sess.last_feat = fv
        self.sess.last_quality = q

        pred = self._map(fv)
        x, y = self._shape(float(pred[0]), float(pred[1]), q)

        # metrics
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

    # ---------- Calibration ----------
    def calibration_add(self, sx, sy):
        if self.sess.last_feat is None or len(self.sess.last_feat) < 4:
            return False
        # เก็บเฉพาะเฟรมที่คุณภาพโอเค
        if self.sess.last_quality < 0.15:
            return False
        fv = [float(v) for v in self.sess.last_feat[:4]]
        self.sess.calib_features.append(fv)
        self.sess.calib_targets.append([float(sx), float(sy)])
        return True

    def calibration_finish(self):
        X = np.asarray(self.sess.calib_features, dtype=np.float32)
        Y = np.asarray(self.sess.calib_targets, dtype=np.float32)
        n = len(X)
        if n < 5:
            self.sess.model_ready = False
            self.sess.affine = None
            self.sess.report = None
            return {"ok": False, "msg": "ต้องเก็บคาลิเบรทอย่างน้อย 5 จุด"}

        # ridge regression (เล็กน้อยกันโอเวอร์ฟิต) และ weighting ด้วยคุณภาพ (ถ้ามี)
        W = np.ones((n, 1), dtype=np.float32)  # จะอัปเกรดเป็นคุณภาพได้ ถ้าบันทึกไว้รายจุด
        X_aug = np.hstack([X, np.ones((n, 1), dtype=np.float32)])  # (n,5)
        lam = 1e-3
        A = np.linalg.solve(X_aug.T @ (W * X_aug) + lam * np.eye(5, dtype=np.float32),
                            X_aug.T @ (W * Y))  # (5,2)
        self.sess.affine = A
        self.sess.model_ready = True

        # train RMSE (px)
        Yp = X_aug @ A
        dx = (Yp[:, 0] - Y[:, 0]) * self.sw
        dy = (Yp[:, 1] - Y[:, 1]) * self.sh
        rmse_train_px = float(np.sqrt(np.mean(dx*dx + dy*dy)))

        # 5-fold CV (ง่ายๆแบบ round-robin)
        K = min(5, n)
        idx = np.arange(n)
        rmses = []
        for k in range(K):
            test = (idx % K) == k
            train = ~test
            Xa, Ya = X_aug[train], Y[train]
            Xb, Yb = X_aug[test],  Y[test]
            if len(Xa) < 5 or len(Xb) == 0:  # กันเคสจุดน้อยมาก
                continue
            Ak = np.linalg.solve(Xa.T @ Xa + lam * np.eye(5, dtype=np.float32),
                                 Xa.T @ Ya)
            Ybk = Xb @ Ak
            dx = (Ybk[:, 0] - Yb[:, 0]) * self.sw
            dy = (Ybk[:, 1] - Yb[:, 1]) * self.sh
            rmses.append(np.sqrt(np.mean(dx*dx + dy*dy)))
        rmse_cv_px = float(np.mean(rmses)) if rmses else rmse_train_px

        # uniformity: RMSE เฉลี่ยต่อจุด / RMSE รวม
        # จัดกลุ่มด้วยพิกัดเป้าหมาย (sx,sy)
        from collections import defaultdict
        buckets = defaultdict(list)
        for (sx, sy), (px, py) in zip(Y.tolist(), (Yp).tolist()):
            err = math.hypot((px - sx)*self.sw, (py - sy)*self.sh)
            buckets[(round(sx,3), round(sy,3))].append(err)
        per_point_rmse = []
        for v in buckets.values():
            per_point_rmse.append(np.sqrt(np.mean(np.square(v))))
        uniformity = float(np.mean(per_point_rmse) / max(1e-6, rmse_train_px)) if per_point_rmse else 1.0

        self.sess.report = CalibrationReport(
            n_points=n, rmse_px=rmse_train_px, rmse_cv_px=rmse_cv_px,
            uniformity=uniformity, width=self.sw, height=self.sh
        )
        return {"ok": True, "n": n, "rmse_px": rmse_train_px,
                "rmse_cv_px": rmse_cv_px, "uniformity": uniformity}

    def get_report(self):
        rep = self.sess.report
        if rep is None:
            return {"model_ready": self.sess.model_ready, "rmse_norm": None, "n_samples": len(self.sess.X)}
        return {"model_ready": True,
                "rmse_px": rep.rmse_px, "rmse_cv_px": rep.rmse_cv_px,
                "uniformity": rep.uniformity, "points": rep.n_points}

# ----------------------------- App State -----------------------------
class AppState:
    def __init__(self):

        # --- math lift / compensation defaults ---
        self.k_yaw = 0.35
        self.c_yaw = 0.08
        self.k_pitch = 0.30
        self.c_pitch = 0.06

        # --- bias / autogain / shaping ---
        self.bias_x = 0.0
        self.bias_y = 0.0
        self.autogain_freeze = False
        self.autogain_reset = False
        self.gamma_x = 1.15
        self.gamma_y = 1.15

        # --- saccade-aware smoothing toggle ---
        self.saccade_aware = True
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
        self.sound_icons = [
            "C:/icon/question.png",
            "C:/icon/pain.png",
            "C:/icon/emergency.png",
            "C:/icon/toilet.png",
            "C:/icon/temperature.png",
            "C:/icon/cutlery.png",
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


            img = frame.to_ndarray(format="bgr24")
            if APP.mode == "Webcam/MediaPipe" and APP.mirror and cv2 is not None:
                img = cv2.flip(img, 1)

            self._ensure()

            # --- sync math-lift params from APP ---
            try:
                if hasattr(self.engine, "set_comp_params"):
                    self.engine.set_comp_params(APP.k_yaw, APP.c_yaw, APP.k_pitch, APP.c_pitch)
                if hasattr(self.engine, "set_bias"):
                    self.engine.set_bias(APP.bias_x, APP.bias_y)
                if hasattr(self.engine, "auto_x"):
                    self.engine.autogain_frozen = bool(APP.autogain_freeze)
                    if bool(APP.autogain_reset):
                        self.engine.reset_auto_gains()
                        APP.autogain_reset = False
                if hasattr(self.engine, "gamma_x"):
                    self.engine.gamma_x = float(APP.gamma_x)
                    self.engine.gamma_y = float(APP.gamma_y)
            except Exception:
                pass

            # update engine params
            self.engine.gain = float(APP.gain)
            self.engine.gamma = float(APP.gamma)
            self.engine.deadzone = float(APP.deadzone)

            # process per frame
            if APP.mode == "Webcam/MediaPipe":
                x, y = self.engine.process_frame_new(img)
                # clamp ค่า x, y ให้อยู่ในช่วง 0.0 - 1.0
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
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
                        # วาดพื้นหลังสีสำหรับแต่ละช่อง icon
                        bg_colors = [
                            (255, 220, 220), # ช่อง 1
                            (220, 255, 220), # ช่อง 2
                            (220, 220, 255), # ช่อง 3
                            (255, 255, 220), # ช่อง 4
                            (220, 255, 255), # ช่อง 5
                            (255, 220, 255), # ช่อง 6
                        ]
                        bg_color = bg_colors[idx % len(bg_colors)]
                        x0_px, y0_px = int(x0*W), int(y0*H)
                        x1_px, y1_px = int(x1*W), int(y1*H)
                        cv2.rectangle(img, (x0_px, y0_px), (x1_px, y1_px), bg_color, -1)

                        # วางไอคอนแทนข้อความ (ใช้ cache)
                        icon_path = APP.sound_icons[idx] if idx < len(APP.sound_icons) else None
                        icon_w, icon_h = int((x1-x0)*W*0.7), int((y1-y0)*H*0.7)
                        cx = int((x0 + x1)/2 * W) - icon_w//2
                        cy = int((y0 + y1)/2 * H) - icon_h//2
                        # ป้องกันขอบภาพ
                        cx = max(0, min(cx, W-icon_w))
                        cy = max(0, min(cy, H-icon_h))
                        # cache icon
                        if not hasattr(self, '_icon_cache'):
                            self._icon_cache = dict()
                        cache_key = f"{icon_path}_{icon_w}_{icon_h}"
                        icon = self._icon_cache.get(cache_key, None)
                        if icon is None and icon_path and cv2 is not None:
                            raw_icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
                            if raw_icon is not None:
                                icon = cv2.resize(raw_icon, (icon_w, icon_h))
                                self._icon_cache[cache_key] = icon
                        if icon is not None:
                            # ถ้าเป็น PNG มี alpha channel
                            if icon.shape[2] == 4:
                                alpha = icon[:,:,3] / 255.0
                                for c in range(3):
                                    img[cy:cy+icon_h, cx:cx+icon_w, c] = (
                                        alpha * icon[:,:,c] + (1-alpha) * img[cy:cy+icon_h, cx:cx+icon_w, c]
                                    )
                            else:
                                img[cy:cy+icon_h, cx:cx+icon_w] = icon
                        else:
                            label = APP.sound_labels[idx] if idx < len(APP.sound_labels) else f"Button {idx+1}"
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                            tx = int((x0 + x1)/2 * W) - tw//2
                            ty = int((y0 + y1)/2 * H) + th//3
                            cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3, cv2.LINE_AA)

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
                                _th.Thread(target=play_sound_file, args=(path,), daemon=True).start()
                            except Exception:
                                pass
                        if hover_idx < len(APP._sound_last_play):
                            APP._sound_last_play[hover_idx] = now
                        # require moving gaze away before repeating same button
                        APP._sound_current_idx = None
                        APP._sound_start = 0.0

                # --- วาด gaze point หลัง overlay soundboard ---
                cx, cy = int(APP.gx*W), int(APP.gy*H)
                cv2.circle(img, (cx, cy), 8, (0,255,0), 2)
                cv2.line(img, (cx-15, cy), (cx+15, cy), (0,255,0), 1)
                cv2.line(img, (cx, cy-15), (cx, cy+15), (0,255,0), 1)
                cv2.putText(img, f"FPS {APP.ui_fps:4.1f} | Latency {APP.ui_lat:4.1f} ms",
                    (W - 460, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
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
    # --- Compensation controls ---
    st.sidebar.subheader("🎯 Compensation (yaw/pitch)")
    APP.k_yaw   = float(st.sidebar.slider("k_yaw",   0.0, 1.0, APP.k_yaw,   0.01))
    APP.c_yaw   = float(st.sidebar.slider("c_yaw",   0.0, 0.5, APP.c_yaw,   0.01))
    APP.k_pitch = float(st.sidebar.slider("k_pitch", 0.0, 1.0, APP.k_pitch, 0.01))
    APP.c_pitch = float(st.sidebar.slider("c_pitch", 0.0, 0.5, APP.c_pitch, 0.01))

    st.sidebar.subheader("🧮 Bias & Auto-Gain")
    c1, c2, c3 = st.sidebar.columns(3)
    if c1.button("Set Center Bias"):
        APP.bias_x = 0.9*APP.bias_x + 0.1*(0.5 - float(APP.gx))
        APP.bias_y = 0.9*APP.bias_y + 0.1*(0.5 - float(APP.gy))
        st.toast("Center bias updated")
    if c2.button("Reset Auto-Gain"):
        APP.autogain_reset = True
    APP.autogain_freeze = c3.toggle("Freeze Auto-Gain", value=APP.autogain_freeze)

    st.sidebar.subheader("📈 Shaping (per-axis)")
    cols = st.sidebar.columns(2)
    APP.gamma_x = float(cols[0].slider("Gamma X", 0.5, 2.0, APP.gamma_x, 0.05))
    APP.gamma_y = float(cols[1].slider("Gamma Y", 0.5, 2.0, APP.gamma_y, 0.05))
    APP.saccade_aware = st.sidebar.toggle("Saccade-aware smoothing", value=APP.saccade_aware)

    st.sidebar.title("⚙️ Controls")
    APP.mode = st.sidebar.selectbox("Mode", ["Webcam/MediaPipe"], index=0)
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
    st.title("👁️ Vision Mouse — Eye-controlled pointer (Webcam)")
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