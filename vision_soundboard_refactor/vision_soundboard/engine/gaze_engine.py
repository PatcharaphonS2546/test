
from __future__ import annotations
import time, math
import numpy as np

from .filters import OneEuroFilter
from .autogain import AutoGain
from .math_utils import clamp01, signed_gamma
from ..calibration.report import CalibrationReport
from ..utils.screen import safe_screen_size

class Session:
    def __init__(self):
        self.calib_features = []   # [ex, ey, yaw, pitch]
        self.calib_targets  = []   # [sx, sy]
        self.last_feat = None
        self.last_quality: float = 0.0
        self.smooth_x = OneEuroFilter(freq=60.0, mincutoff=0.5, beta=0.005, dcutoff=1.2)
        self.smooth_y = OneEuroFilter(freq=60.0, mincutoff=0.5, beta=0.005, dcutoff=1.2)
        self.fps_hist: list[float] = []
        self.t_prev = None
        self.affine = None      # (5,2)
        self.model_ready: bool = False
        self.report: CalibrationReport | None = None
        self.metrics = {"fps": 0.0, "latency_ms": 0.0}

class GazeEngine:
    def __init__(self, extractor, screen_w: int | None=None, screen_h: int | None=None):
        self.k_yaw = 0.35;  self.c_yaw = 30.0
        self.k_pitch = 0.30; self.c_pitch = 25.0
        self.bias_x = 0.0; self.bias_y = 0.0
        self.auto_x = AutoGain(target=0.48, window_sec=2.5, ema_alpha=0.05)
        self.auto_y = AutoGain(target=0.48, window_sec=2.5, ema_alpha=0.05)
        self.autogain_frozen = False
        self.gamma_x = 1.15; self.gamma_y = 1.15
        self.gain = 1.0
        self.gamma = 1.0
        self.deadzone = 0.02

        self.ext = extractor
        sw, sh = safe_screen_size()
        self.sw = screen_w or sw
        self.sh = screen_h or sh
        self.sess = Session()

    def set_comp_params(self, k_yaw, c_yaw, k_pitch, c_pitch):
        self.k_yaw = float(k_yaw); self.c_yaw = float(c_yaw) if c_yaw > 1 else float(c_yaw)*100
        self.k_pitch = float(k_pitch); self.c_pitch = float(c_pitch) if c_pitch > 1 else float(c_pitch)*100

    def set_bias(self, bx, by):
        self.bias_x = float(bx); self.bias_y = float(by)

    def reset_auto_gains(self):
        self.auto_x.reset(); self.auto_y.reset()

    def get_last_metrics(self):
        return dict(self.sess.metrics)

    def _apply_pose_comp(self, ex, ey, yaw, pitch):
        q = max(0.0, min(1.0, self.sess.last_quality if hasattr(self.sess, 'last_quality') else 1.0))
        offx = self.k_yaw * math.tanh(yaw / self.c_yaw) * q
        offy = self.k_pitch * math.tanh(-pitch / self.c_pitch) * q
        ex = 0.5 + (ex - 0.5) + offx + self.bias_x
        ey = 0.5 + (ey - 0.5) + offy + self.bias_y
        return float(min(1.0, max(0.0, ex))), float(min(1.0, max(0.0, ey)))

    def _auto_gain(self, ex, ey, q):
        now = time.time()
        if not self.autogain_frozen:
            self.auto_x.update(abs(ex - 0.5), now)
            self.auto_y.update(abs(ey - 0.5), now)
        gx = self.auto_x.gain; gy = self.auto_y.gain
        ex = 0.5 + (ex - 0.5) * gx
        ey = 0.5 + (ey - 0.5) * gy
        return ex, ey

    def _shape(self, x: float, y: float, q: float):
        x = 0.5 + (x - 0.5) * self.gain
        y = 0.5 + (y - 0.5) * self.gain
        if getattr(self, "gamma_x", 1.0) != 1.0:
            x = signed_gamma(x, self.gamma_x)
        if getattr(self, "gamma_y", 1.0) != 1.0:
            y = signed_gamma(y, self.gamma_y)
        if self.gamma != 1.0:
            def _gm(v, g):
                s = (v - 0.5); sign = 1.0 if s >= 0 else -1.0
                return 0.5 + sign * (abs(s) ** g)
            x = _gm(x, self.gamma); y = _gm(y, self.gamma)
        dz = self.deadzone
        def _dz(v):
            d = v - 0.5
            return 0.5 if abs(d) < dz else v
        x, y = _dz(x), _dz(y)
        for f in (self.sess.smooth_x, self.sess.smooth_y):
            f.mincutoff = max(0.3, 1.6 - 1.2 * q)
            f.beta = 0.01 + 0.12 * (1.0 - q)
        x = float(self.sess.smooth_x.filter(min(1.0, max(0.0, x))))
        y = float(self.sess.smooth_y.filter(min(1.0, max(0.0, y))))
        return x, y

    def _map(self, fv: np.ndarray) -> np.ndarray:
        ex, ey, yaw, pitch = [float(v) for v in fv]
        ex, ey = self._apply_pose_comp(ex, ey, yaw, pitch)
        ex, ey = self._auto_gain(ex, ey, self.sess.last_quality)
        if self.sess.model_ready and (self.sess.affine is not None):
            X_aug = np.array([ex, ey, yaw, pitch, 1.0], dtype=np.float32)
            px, py = (X_aug @ self.sess.affine).tolist()
        else:
            px = 0.5 + (ex - 0.5) * 1.6
            py = 0.5 + (ey - 0.5) * 1.4
        py = 0.5 + (py - 0.5) * 1.15
        return np.array([clamp01(px), clamp01(py)], dtype=np.float32)

    def process_frame(self, frame_bgr, extractor):
        t0 = time.time()
        feat = extractor.extract(frame_bgr) if extractor else None
        if feat is None or feat.get("quality", 0.0) < 0.01:
            fv = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32); q = 0.0
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
        if self.sess.last_quality < 0.08:
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
            return {"ok": False, "msg": "Need at least 5 points"}

        W = np.ones((n, 1), dtype=np.float32)
        X_aug = np.hstack([X, np.ones((n, 1), dtype=np.float32)])
        lam_xy, lam_pose, lam_bias = 1e-3, 5e-2, 1e-6
        R = np.diag([lam_xy, lam_xy, lam_pose, lam_pose, lam_bias]).astype(np.float32)
        A = np.linalg.solve(X_aug.T @ (W * X_aug) + R, X_aug.T @ (W * Y))
        self.sess.affine = A
        self.sess.model_ready = True

        Yp = X_aug @ A
        dx = (Yp[:, 0] - Y[:, 0]) * self.sw
        dy = (Yp[:, 1] - Y[:, 1]) * self.sh
        rmse_train_px = float(np.sqrt(np.mean(dx*dx + dy*dy)))

        K = min(5, n)
        idx = np.arange(n)
        rmses = []
        for k in range(K):
            test = (idx % K) == k
            train = ~test
            Xa, Ya = X_aug[train], Y[train]
            Xb, Yb = X_aug[test],  Y[test]
            if len(Xa) < 5 or len(Xb) == 0:
                continue
            Ak = np.linalg.solve(Xa.T @ Xa + lam_xy * np.eye(5, dtype=np.float32),
                                 Xa.T @ Ya)
            Ybk = Xb @ Ak
            dx = (Ybk[:, 0] - Yb[:, 0]) * self.sw
            dy = (Ybk[:, 1] - Yb[:, 1]) * self.sh
            rmses.append(np.sqrt(np.mean(dx*dx + dy*dy)))
        rmse_cv_px = float(np.mean(rmses)) if rmses else rmse_train_px

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
            return {"model_ready": self.sess.model_ready, "rmse_norm": None, "n_samples": len(self.sess.calib_features)}
        return {"model_ready": True,
                "rmse_px": rep.rmse_px, "rmse_cv_px": rep.rmse_cv_px,
                "uniformity": rep.uniformity, "points": rep.n_points}
