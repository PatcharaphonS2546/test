
import math, time
import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

from av import VideoFrame  # type: ignore
from streamlit_webrtc import VideoProcessorBase  # type: ignore

from ..engine.gaze_engine import GazeEngine
from ..engine.feature_extractor import FeatureExtractor
from ..engine.mouse import MouseController
from ..audio.sound import play_sound_file
from ..utils.geom import inside_rect

class Processor(VideoProcessorBase):
    def __init__(self, app_state, metrics):
        self.APP = app_state
        self.METRICS = metrics
        self.engine = None
        self.mouse = MouseController(False)
        self.calib_hold_start = None
        self.pool = []
        self._icon_cache = dict()

    def _ensure(self):
        if self.engine is not None:
            return
        extractor = FeatureExtractor(use_mediapipe=(self.APP.mode == "Webcam/MediaPipe"))
        self.engine = GazeEngine(extractor, screen_w=self.APP.screen_w, screen_h=self.APP.screen_h)

    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if self.APP.mode == "Webcam/MediaPipe" and self.APP.mirror and cv2 is not None:
            img = cv2.flip(img, 1)

        self._ensure()

        try:
            if hasattr(self.engine, "set_comp_params"):
                self.engine.set_comp_params(self.APP.k_yaw, self.APP.c_yaw, self.APP.k_pitch, self.APP.c_pitch)
            if hasattr(self.engine, "set_bias"):
                self.engine.set_bias(self.APP.bias_x, self.APP.bias_y)
            if hasattr(self.engine, "auto_x"):
                self.engine.autogain_frozen = bool(self.APP.autogain_freeze)
                if bool(self.APP.autogain_reset):
                    self.engine.reset_auto_gains()
                    self.APP.autogain_reset = False
            if hasattr(self.engine, "gamma_x"):
                self.engine.gamma_x = float(self.APP.gamma_x)
                self.engine.gamma_y = float(self.APP.gamma_y)
        except Exception:
            pass

        self.engine.gain = float(self.APP.gain)
        self.engine.gamma = float(self.APP.gamma)
        self.engine.deadzone = float(self.APP.deadzone)

        if self.APP.mode == "Webcam/MediaPipe":
            try:
                x, y = self.engine.process_frame(img, self.engine.ext)
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
            except Exception:
                x, y = 0.5, 0.5
        else:
            x, y = self.APP.gx, self.APP.gy

        if self.APP.invert_x: x = 1.0 - x
        if self.APP.invert_y: y = 1.0 - y

        self.APP.gx, self.APP.gy = float(x), float(y)
        m = self.engine.get_last_metrics()
        self.APP.ui_fps = float(m.get("fps", 0.0)); self.APP.ui_lat = float(m.get("latency_ms", 0.0))

        if self.APP.countdown_active and cv2 is not None:
            remaining = self.APP.countdown_end - time.time()
            if remaining <= 0:
                self.APP.countdown_active = False
                self.APP.calib_overlay = True
                self.calib_hold_start = None
                self.pool = []
            else:
                h, w = img.shape[:2]
                overlay = img.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
                sec = int(math.ceil(remaining))
                text = "GO" if 0 < remaining < 0.3 else str(sec)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 4.0; thick = 8
                (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
                cv2.putText(img, text, (w // 2 - tw // 2, h // 2 + th // 3),
                            font, scale, (0, 200, 255), thick, cv2.LINE_AA)
                cv2.putText(img, "Starting calibration...",
                            (max(20, w // 2 - 280), min(h - 20, h // 2 + th + 40)),
                            font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                return VideoFrame.from_ndarray(img, format="bgr24")

        if self.APP.calib_overlay and cv2 is not None:
            tx, ty = self.APP.targets[self.APP.idx] if (0 <= self.APP.idx < len(self.APP.targets)) else (0.5, 0.5)
            h, w = img.shape[:2]
            gx_i, gy_i = int(tx*w), int(ty*h)
            overlay = img.copy()
            cv2.rectangle(overlay, (gx_i-32, gy_i-32), (gx_i+32, gy_i+32), (0,170,255), -1)
            cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
            cv2.circle(img, (gx_i, gy_i), 22, (0,170,255), 3)

            now = time.time()
            if self.engine.sess.last_feat is not None:
                if self.calib_hold_start is None:
                    self.calib_hold_start = now; self.pool = []
                feat = self.engine.sess.last_feat.copy()
                quality = self.engine.sess.last_quality
                if quality > 0.08:
                    self.pool.append(feat)
                elapsed = (now - self.calib_hold_start)*1000.0
            else:
                self.calib_hold_start = None; self.pool = []; elapsed = 0.0
            frac = max(0.0, min(1.0, elapsed / float(max(1, self.APP.dwell_ms))))
            end_angle = int(360 * frac)
            cv2.ellipse(img, (gx_i, gy_i), (28, 28), 0, 0, end_angle, (0,255,180), 5)
            cv2.putText(img, f"คาลิเบรตจุดที่ {self.APP.idx+1}/{len(self.APP.targets)}",
                        (24,48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 3)
            cv2.putText(img, "กรุณาตั้งหัวตรงและมองจุดเป้าหมาย",
                        (24,90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

            if frac >= 1.0 and 0 <= self.APP.idx < len(self.APP.targets):
                if len(self.pool) >= 5:
                    import numpy as np
                    avg_feat = np.mean(np.stack(self.pool, axis=0), axis=0).astype(np.float32)
                    self.engine.sess.last_feat = avg_feat
                self.engine.calibration_add(tx, ty)
                self.APP.idx += 1; self.calib_hold_start = None; self.pool = []
                if self.APP.idx >= len(self.APP.targets):
                    rep = self.engine.calibration_finish()
                    self.APP.calib_overlay = False
                    txt = ("✅ Calibration สำเร็จ · " f"RMSE={rep.get('rmse_px',0):.0f}px · " f"CV={rep.get('rmse_cv_px',0):.0f}px · " f"U={rep.get('uniformity',0):.2f}")
                    cv2.rectangle(img, (10,10), (10+900, 10+50), (0,0,0), -1)
                    cv2.putText(img, txt, (18,48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,230,255), 3)

        # Mouse pointer only
        self.mouse.set_enable(self.APP.mouse_enabled)
        self.mouse.update(self.APP.gx, self.APP.gy)

        if cv2 is not None:
            h, w = img.shape[:2]
            cx, cy = int(self.APP.gx*w), int(self.APP.gy*h)
            cv2.circle(img, (cx, cy), 8, (0,255,0), 2)
            cv2.line(img, (cx-15, cy), (cx+15, cy), (0,255,0), 1)
            cv2.line(img, (cx, cy-15), (cx, cy+15), (0,255,0), 1)
            cv2.putText(img, f"FPS {self.APP.ui_fps:4.1f} | Latency {self.APP.ui_lat:4.1f} ms",
                (w - 460, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Soundboard overlay (2x3)
        if self.APP.soundboard_on and (not self.APP.calib_overlay) and (not self.APP.countdown_active) and cv2 is not None:
            rows, cols = 2, 3
            H, W = img.shape[:2]
            hover_idx = None
            for r in range(rows):
                for c in range(cols):
                    idx = r*cols + c
                    x0, x1 = c/cols, (c+1)/cols
                    y0, y1 = r/rows, (r+1)/rows
                    pt1 = (int(x0*W), int(y0*H))
                    pt2 = (int(x1*W), int(y1*H))
                    inside = inside_rect(self.APP.gx, self.APP.gy, (x0, y0, x1, y1))
                    # กำหนดสีปุ่มจาก self.APP.sound_colors ถ้ามี
                    if hasattr(self.APP, "sound_colors") and idx < len(self.APP.sound_colors):
                        base_color = self.APP.sound_colors[idx]
                    else:
                        base_color = (40, 40, 40)
                    color = base_color; thick = 2
                    if inside:
                        color = (0, 200, 255); thick = 4
                        hover_idx = idx
                    x0_px, y0_px = int(x0*W), int(y0*H)
                    x1_px, y1_px = int(x1*W), int(y1*H)
                    # สีพื้นปุ่ม (ใช้ base_color) ก่อน
                    cv2.rectangle(img, (x0_px, y0_px), (x1_px, y1_px), base_color, -1)
                    # วาดกรอบปุ่ม (color/thick) หลังสุด
                    cv2.rectangle(img, pt1, pt2, color, thick)
                    icon_path = self.APP.sound_icons[idx] if idx < len(self.APP.sound_icons) else None
                    icon_w, icon_h = int((x1-x0)*W*0.7), int((y1-y0)*H*0.7)
                    cx = int((x0 + x1)/2 * W) - icon_w//2
                    cy = int((y0 + y1)/2 * H) - icon_h//2
                    cx = max(0, min(cx, W-icon_w))
                    cy = max(0, min(cy, H-icon_h))
                    cache_key = f"{icon_path}_{icon_w}_{icon_h}"
                    icon = self._icon_cache.get(cache_key, None)
                    if icon is None and icon_path and cv2 is not None:
                        raw_icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
                        if raw_icon is not None:
                            icon = cv2.resize(raw_icon, (icon_w, icon_h))
                            self._icon_cache[cache_key] = icon
                    if icon is not None:
                        if cy+icon_h <= img.shape[0] and cx+icon_w <= img.shape[1]:
                            if icon.shape[2] == 4:
                                alpha = icon[:,:,3] / 255.0
                                for c in range(3):
                                    img[cy:cy+icon_h, cx:cx+icon_w, c] = (
                                        alpha * icon[:,:,c] + (1-alpha) * img[cy:cy+icon_h, cx:cx+icon_w, c]
                                    )
                            else:
                                img[cy:cy+icon_h, cx:cx+icon_w] = icon
                    else:
                        label = self.APP.sound_labels[idx] if idx < len(self.APP.sound_labels) else f"Button {idx+1}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                        tx = int((x0 + x1)/2 * W) - tw//2
                        ty = int((y0 + y1)/2 * H) + th//3
                        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3, cv2.LINE_AA)

            now = time.time()
            if hover_idx is None:
                self.APP._sound_current_idx = None
                self.APP._sound_start = 0.0
            else:
                if self.APP._sound_current_idx != hover_idx:
                    self.APP._sound_current_idx = hover_idx
                    self.APP._sound_start = now
                elapsed_ms = (now - self.APP._sound_start) * 1000.0
                c = hover_idx % cols
                r = hover_idx // cols
                x0, x1 = c/cols, (c+1)/cols
                y0 = r/rows
                pcx = int((x0 + x1)/2 * W)
                pcy = int(y0 * H + 40)
                frac = max(0.0, min(1.0, elapsed_ms / float(max(1, self.APP.sound_dwell_ms))))
                cv2.ellipse(img, (pcx, pcy), (28,28), 0, 0, int(360*frac), (0,200,255), 4)
                last = self.APP._sound_last_play[hover_idx] if hover_idx < len(self.APP._sound_last_play) else -1.0
                if (elapsed_ms >= self.APP.sound_dwell_ms) and ((now - last) >= self.APP.sound_cooldown_ms/1000.0):
                    path = self.APP.sound_files[hover_idx] if hover_idx < len(self.APP.sound_files) else None
                    if path:
                        play_sound_file(path)
                    if hover_idx < len(self.APP._sound_last_play):
                        self.APP._sound_last_play[hover_idx] = now
                    self.APP._sound_current_idx = None
                    self.APP._sound_start = 0.0

            cx, cy = int(self.APP.gx*W), int(self.APP.gy*H)
            cv2.circle(img, (cx, cy), 8, (0,255,0), 2)
            cv2.line(img, (cx-15, cy), (cx+15, cy), (0,255,0), 1)
            cv2.line(img, (cx, cy-15), (cx, cy+15), (0,255,0), 1)
            cv2.putText(img, f"FPS {self.APP.ui_fps:4.1f} | Latency {self.APP.ui_lat:4.1f} ms",
                (W - 460, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # record metrics
        sw, sh = self.engine.sw, self.engine.sh
        pred_px = (int(self.APP.gx*sw), int(self.APP.gy*sh))
        target_px = None; point_id = None
        if self.APP.calib_overlay and (0 <= self.APP.idx < len(self.APP.targets)):
            tx, ty = self.APP.targets[self.APP.idx]
            target_px = (int(tx*sw), int(ty*sh))
            point_id = int(self.APP.idx)
        self.METRICS.record(latency_ms=self.APP.ui_lat, pred_px=pred_px, target_px=target_px, point_id=point_id)

        return VideoFrame.from_ndarray(img, format="bgr24")
