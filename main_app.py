#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal, reliable MVP for webcam gaze tracking with Streamlit-WebRTC.
- Uses the existing GazeEngine (gaze_core_onefile.py)
- Built-in 9/12-point calibration with dwell timing
- Optional mouse control via mouse_controller.py

Run:
    streamlit run main_app.py

Requires:
    pip install streamlit streamlit-webrtc av opencv-python-headless numpy
    # Optional
    pip install mediapipe scikit-learn pynput pyautogui screeninfo

Notes:
- Designed to run even when mediapipe/scikit-learn are missing (engine has fallbacks).
- Keeps async_processing=False to avoid threading issues.
"""
from __future__ import annotations
import json, math, random, time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import av, cv2, numpy as np, streamlit as st
from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer
from Orlosky3DEyeTracker import iter_norm_from_esp32, STREAM_URL

# ---------------------------
# Import engine + optional mouse
# ---------------------------
try:
    from gaze_core_onefile import GazeEngine, Config  # local file
except Exception as e:
    raise SystemExit("\n[ERROR] Cannot import gaze_core_onefile.GazeEngine.\nPlace gaze_core_onefile.py next to this file.\n" + str(e))

try:
    import mouse_controller as mc
    _HAS_MOUSE = True
except Exception:
    _HAS_MOUSE = False
    mc = None

# ---------------------------
# Small helpers
# ---------------------------

def get_default_screen_size() -> Tuple[int, int]:
    try:
        from screeninfo import get_monitors
        mons = get_monitors()
        if mons:
            return int(mons[0].width), int(mons[0].height)
    except Exception:
        pass
    try:
        import pyautogui
        sz = pyautogui.size()
        return int(sz.width), int(sz.height)
    except Exception:
        pass
    return 1920, 1080


def draw_crosshair(img: np.ndarray, cx: int, cy: int, size: int = 18, color=(0, 165, 255), thick=2):
    cv2.line(img, (cx - size, cy), (cx + size, cy), color, thick, cv2.LINE_AA)
    cv2.line(img, (cx, cy - size), (cx, cy + size), color, thick, cv2.LINE_AA)


def make_screen_canvas(
    screen_w: int,
    screen_h: int,
    gaze_xy: Optional[Tuple[float, float]],
    max_canvas_w: int = 640,
    trail: Optional[deque] = None,
    calib_target: Optional[Tuple[float, float]] = None,
    calib_progress: Optional[Tuple[int, int]] = None,
    dwell_progress: float = 0.0,
) -> np.ndarray:
    scale = max_canvas_w / float(screen_w)
    canvas_w = max_canvas_w
    canvas_h = int(round(screen_h * scale))
    canvas = np.full((canvas_h, canvas_w, 3), (20, 20, 20), dtype=np.uint8)
    cv2.rectangle(canvas, (1, 1), (canvas_w - 2, canvas_h - 2), (90, 90, 90), 1)

    if trail:
        for i, (tx, ty) in enumerate(list(trail)[-60:]):
            tx = int(round(np.clip(tx, 0, 1) * (canvas_w - 1)))
            ty = int(round(np.clip(ty, 0, 1) * (canvas_h - 1)))
            r = max(1, int(6 * (i / max(1, len(trail)))))
            cv2.circle(canvas, (tx, ty), r, (60, 60, 200), -1, cv2.LINE_AA)

    if calib_target is not None:
        cx = int(round(np.clip(calib_target[0], 0, 1) * (canvas_w - 1)))
        cy = int(round(np.clip(calib_target[1], 0, 1) * (canvas_h - 1)))
        progress_color = (0, 255, int(255 * (1 - dwell_progress)))
        cv2.circle(canvas, (cx, cy), 18, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), int(18 * dwell_progress), progress_color, -1, cv2.LINE_AA)
        draw_crosshair(canvas, cx, cy, size=22, color=(0, 165, 255), thick=1)
        msg = "Look at the center of the target"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(
            canvas,
            msg,
            ((canvas_w - tw) // 2, canvas_h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
            cv2.LINE_AA,
        )

    if gaze_xy is not None:
        gx = int(round(np.clip(gaze_xy[0], 0, 1) * (canvas_w - 1)))
        gy = int(round(np.clip(gaze_xy[1], 0, 1) * (canvas_h - 1)))
        cv2.circle(canvas, (gx, gy), 12, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.drawMarker(canvas, (gx, gy), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)

    if calib_progress:
        i, n = calib_progress
        msg = f"Calibrating: Point {i}/{n}"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(
            canvas,
            msg,
            ((canvas_w - tw) // 2, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return canvas


def grid_points(num_points: int, margin: float = 0.1) -> List[Tuple[float, float]]:
    margin = float(np.clip(margin, 0.02, 0.2))
    if num_points == 9:
        xs = [margin, 0.5, 1 - margin]
        ys = [margin, 0.5, 1 - margin]
        pts = [(x, y) for y in ys for x in xs]
        order_idx = [0, 2, 6, 8, 1, 3, 5, 7, 4]
        pts = [pts[i] for i in order_idx]
    elif num_points == 12:
        xs = [margin, (1 - margin) / 3 + margin * 0, 2 * (1 - margin) / 3 + margin * 0, 1 - margin]
        ys = [margin, 0.5, 1 - margin]
        pts = [(x, y) for y in ys for x in xs]
    else:
        raise ValueError("num_points must be 9 or 12")
    return pts


# ---------------------------
# Video Processor
# ---------------------------
@dataclass
class CalibState:
    active: bool = False
    points: Optional[List[Tuple[float, float]]] = None
    dwell_ms: int = 800
    idx: int = 0
    started_at: float = 0.0
    last_step_at: float = 0.0
    recorded_in_step: bool = False
    report: Optional[dict] = None


class GazeProcessor(VideoProcessorBase):
    def __init__(self):
        # Engine + session
        self.eng = GazeEngine(Config(return_debug=False))
        self.sid = self.eng.start_session()
        # Runtime
        self.frame_count = 0
        self.trail = deque(maxlen=120)
        self.gaze = (0.5, 0.5)
        self.quality = 0.0
        self.drop_rate = 0.0
        self.fps = 0.0
        self.lat_ms = 0.0
        # Screen
        self.screen_w, self.screen_h = get_default_screen_size()
        # Shaping defaults (mirrors engine defaults)
        self.cfg_cache = dict(
            quality_thresh=0.40,
            head_pose_only_norm=False,
            gain=1.0,
            gain_x=1.0,
            gain_y=1.0,
            gamma=1.0,
            deadzone=0.0,
            clamp_jump=0.18,
            smooth_fc=2.0,
            smooth_beta=0.02,
            fb_kx=0.8,
            fb_ky=0.8,
        )
        self._apply_cfg()
        # Mouse control
        self.mouse = mc.MouseController() if _HAS_MOUSE else None
        if self.mouse:
            self.mouse.set_config(False, False, 0.8, 30, 0.35)
        # Calibration
        self.calib = CalibState()
        # Input mode
        self.input_mode = "webrtc"
        self._orlo_iter = None
        self.orlo_url = STREAM_URL

    # ---- UI hooks ----
    def set_screen_size(self, w: int, h: int):
        self.screen_w, self.screen_h = int(w), int(h)

    def set_quality_thresh(self, q: float):
        self.cfg_cache["quality_thresh"] = float(q)
        self._apply_cfg()

    def set_head_pose_norm(self, enabled: bool):
        self.cfg_cache["head_pose_only_norm"] = bool(enabled)
        self._apply_cfg()

    def set_gain(self, g: float):
        self.cfg_cache["gain"] = float(g)
        self._apply_cfg()

    def set_gain_axes(self, gx: float, gy: float):
        self.cfg_cache["gain_x"] = float(gx)
        self.cfg_cache["gain_y"] = float(gy)
        self._apply_cfg()

    def set_shape(self, gamma: float, deadzone: float):
        self.cfg_cache["gamma"] = float(gamma)
        self.cfg_cache["deadzone"] = float(deadzone)
        self._apply_cfg()

    def set_clamp_jump(self, cj: float):
        self.cfg_cache["clamp_jump"] = float(cj)
        self._apply_cfg()

    def set_smooth(self, fc: float, beta: float):
        self.cfg_cache["smooth_fc"] = float(fc)
        self.cfg_cache["smooth_beta"] = float(beta)
        self._apply_cfg()

    def set_fallback_gain(self, kx: float, ky: float):
        self.cfg_cache["fb_kx"] = float(kx)
        self.cfg_cache["fb_ky"] = float(ky)
        self._apply_cfg()

    def set_mouse_config(self, enabled: bool, dwell_enabled: bool, dwell_time: float, dwell_radius: int, smooth_alpha: float):
        if self.mouse:
            self.mouse.set_config(enabled, dwell_enabled, dwell_time, dwell_radius, smooth_alpha)

    def start_calibration(self, points: List[Tuple[float, float]], dwell_ms: int = 800, shuffle: bool = False):
        if shuffle:
            pts = points[:]
            random.shuffle(pts)
        else:
            pts = points
        self.calib = CalibState(active=True, points=pts, dwell_ms=int(dwell_ms), idx=0, started_at=time.time())
        self.eng.calibration_start(self.sid)

    def stop_calibration(self):
        self.calib.active = False

    def set_input_mode(self, mode, orlo_url=None):
        self.input_mode = mode
        if mode == "orlosky":
            self.orlo_url = orlo_url or STREAM_URL
            self._orlo_iter = iter_norm_from_esp32(self.orlo_url)
        else:
            self._orlo_iter = None

    # ---- internals ----
    def _apply_cfg(self):
        self.eng.update_config(
            quality_thresh=self.cfg_cache["quality_thresh"],
            head_pose_only_norm=self.cfg_cache["head_pose_only_norm"],
            gain=self.cfg_cache["gain"],
            gain_x=self.cfg_cache["gain_x"],
            gain_y=self.cfg_cache["gain_y"],
            gamma=self.cfg_cache["gamma"],
            deadzone=self.cfg_cache["deadzone"],
            clamp_jump=self.cfg_cache["clamp_jump"],
            smooth_fc=self.cfg_cache["smooth_fc"],
            smooth_beta=self.cfg_cache["smooth_beta"],
            fallback_kx=self.cfg_cache["fb_kx"],
            fallback_ky=self.cfg_cache["fb_ky"],
        )

    # ---- main frame ----
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # --- Robust input mode switching ---
        if self.input_mode == "orlosky":
            if self._orlo_iter is None:
                self._orlo_iter = iter_norm_from_esp32(self.orlo_url)
            try:
                norm = next(self._orlo_iter)
                if norm is None:
                    # Stream error or frame not detected
                    self.quality = 0.0
                    # Keep last gaze
                    img = frame.to_ndarray(format="bgr24")
                    cv2.putText(img, "[Orlosky: No Data]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    return av.VideoFrame.from_ndarray(img, format="bgr24")
                avgx, avgy = norm[:2]
                # Send to engine for gaze estimation
                result = self.eng.process_external(self.sid, avgx, avgy)
                if 'gaze' in result:
                    self.gaze = (float(result['gaze'][0]), float(result['gaze'][1]))
                else:
                    self.gaze = (float(result.get('x', 0.5)), float(result.get('y', 0.5)))
                self.quality = float(result.get('quality', 1.0))
                img = frame.to_ndarray(format="bgr24")
                gx = int(self.gaze[0] * (img.shape[1] - 1))
                gy = int(self.gaze[1] * (img.shape[0] - 1))
                cv2.circle(img, (gx, gy), 12, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(img, f"Orlosky Q={self.quality:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            except StopIteration:
                # Stream ended, restart iterator
                self._orlo_iter = iter_norm_from_esp32(self.orlo_url)
                self.quality = 0.0
                img = frame.to_ndarray(format="bgr24")
                cv2.putText(img, "[Orlosky: Stream Restarted]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            except Exception as e:
                # Any error: fallback, keep last gaze
                self.quality = 0.0
                img = frame.to_ndarray(format="bgr24")
                cv2.putText(img, f"[Orlosky Error] {str(e)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
        else:
            img = frame.to_ndarray(format="bgr24")
            try:
                out = self.eng.process_frame(self.sid, img)
                self.frame_count += 1
                self.gaze = (float(out.get("x", 0.5)), float(out.get("y", 0.5)))
                self.quality = float(out.get("quality", 0.0))
                self.lat_ms = float(out.get("latency_ms", 0.0))
                self.fps = float(out.get("fps_proc", 0.0))
                ms = self.eng.metrics_snapshot(self.sid)
                self.drop_rate = float(ms.get("drop_rate", 0.0))
            except Exception as e:
                # If the engine throws, show a diagnostic frame
                img = np.full_like(img, 50)
                cv2.putText(img, f"ENGINE ERROR: {e}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Add to trail and (optional) move mouse
        self.trail.append(self.gaze)
        if self.mouse and self.mouse.enabled:
            self.mouse.update(self.gaze[0], self.gaze[1])

        # Calibration state machine (dwell-based auto sample)
        calib_target = None
        dwell_progress = 0.0
        if self.calib.active and self.calib.points:
            # Current target
            calib_target = self.calib.points[self.calib.idx]
            # Dwell timer
            now = time.time()
            if self.calib.last_step_at == 0.0:
                self.calib.last_step_at = now
            dt = (now - self.calib.last_step_at) * 1000.0
            dwell_progress = float(np.clip(dt / max(1.0, self.calib.dwell_ms), 0.0, 1.0))
            # When dwell finishes, record a sample and advance
            if dt >= self.calib.dwell_ms and not self.calib.recorded_in_step:
                try:
                    self.eng.calibration_add_point(self.sid, calib_target[0], calib_target[1])
                    self.calib.recorded_in_step = True
                except Exception:
                    # last_feat may be missing if the first frames were dropped; just skip this cycle
                    self.calib.recorded_in_step = False
                    self.calib.last_step_at = now  # retry dwell
            # Move to next point after brief pause post-record
            if self.calib.recorded_in_step and dt >= self.calib.dwell_ms + 300:
                self.calib.idx += 1
                self.calib.recorded_in_step = False
                self.calib.last_step_at = time.time()
                if self.calib.idx >= len(self.calib.points):
                    try:
                        self.calib.report = self.eng.calibration_finish(self.sid)
                    except Exception as e:
                        self.calib.report = {"ok": False, "error": str(e)}
                    self.calib.active = False

        # Overlay visuals
        h, w = img.shape[:2]
        gx = int(np.clip(self.gaze[0] * w, 15, w - 15))
        gy = int(np.clip(self.gaze[1] * h, 15, h - 15))
        # Borders & HUD
        cv2.rectangle(img, (6, 6), (w - 6, h - 6), (0, 200, 0), 2)
        cv2.putText(img, f"GAZE: ({self.gaze[0]:.3f}, {self.gaze[1]:.3f})", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.putText(img, f"Q:{self.quality:.2f}  LAT:{self.lat_ms:.0f}ms", (18, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        # Gaze marker
        cv2.circle(img, (gx, gy), 22, (0, 255, 0), 3)
        cv2.drawMarker(img, (gx, gy), (255, 255, 255), cv2.MARKER_CROSS, 32, 3)
        # Corner markers
        cv2.circle(img, (20, 20), 10, (255, 0, 0), -1)
        cv2.circle(img, (w - 20, 20), 10, (0, 255, 0), -1)
        cv2.circle(img, (20, h - 20), 10, (0, 0, 255), -1)
        cv2.circle(img, (w - 20, h - 20), 10, (255, 255, 255), -1)
        # Calibration HUD
        if calib_target is not None:
            tx = int(np.clip(calib_target[0] * w, 30, w - 30))
            ty = int(np.clip(calib_target[1] * h, 30, h - 30))
            cv2.circle(img, (tx, ty), 20, (0, 165, 255), 2, cv2.LINE_AA)
            cv2.circle(img, (tx, ty), int(20 * dwell_progress), (0, 255, int(255 * (1 - dwell_progress))), -1, cv2.LINE_AA)
            draw_crosshair(img, tx, ty, size=24, color=(0, 165, 255), thick=2)
            cv2.putText(img, f"Calibrating {self.calib.idx+1}/{len(self.calib.points)}", (18, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Gaze UI — MVP", layout="wide")
st.title("👁️ Gaze UI — MVP (Engine-integrated)")

with st.sidebar:
    st.header("⚙️ Settings")
    d_w, d_h = get_default_screen_size()
    screen_w = st.number_input("Screen width (px)", 320, 9999, d_w, 10)
    screen_h = st.number_input("Screen height (px)", 240, 9999, d_h, 10)

    st.subheader("🖱️ Mouse Control")
    mc_enabled = st.checkbox("Enable Mouse Control", False)
    mc_dwell_enabled = st.checkbox("Enable Dwell Click", False)
    c1, c2 = st.columns(2)
    mc_dwell_time = c1.slider("Dwell Time (s)", 0.2, 2.0, 0.8, 0.1)
    mc_dwell_radius = c2.slider("Dwell Radius (px)", 10, 100, 30, 1)
    mc_alpha = st.slider("Mouse Smoothing (alpha)", 0.05, 0.95, 0.35, 0.05)

    with st.expander("🔧 Core Processing", expanded=False):
        qth = st.slider("Input Quality Threshold", 0.1, 0.9, 0.40, 0.01)
        hp_norm = st.checkbox("Head-Pose-Only Normalization", False)

    with st.expander("🎛️ Gaze Shaping & Sensitivity", expanded=False):
        gain = st.slider("Overall Sensitivity (gain)", 0.6, 2.0, 1.0, 0.01)
        g1, g2 = st.columns(2)
        gain_x = g1.slider("Gain X", 0.5, 2.0, 1.0, 0.01)
        gain_y = g2.slider("Gain Y", 0.5, 2.0, 1.0, 0.01)
        gamma = st.slider("Gamma", 0.6, 1.4, 1.0, 0.01, help="<1 boosts edges, >1 compresses edges")
        deadzone = st.slider("Deadzone", 0.0, 0.08, 0.0, 0.005)

    with st.expander("📊 Smoothing & Fallback", expanded=False):
        clampj = st.slider("Clamp Jump", 0.05, 0.6, 0.18, 0.01)
        s1, s2 = st.columns(2)
        smooth_fc = s1.slider("OneEuro fc", 1.0, 6.0, 2.0, 0.1)
        smooth_beta = s2.slider("OneEuro beta", 0.01, 0.1, 0.02, 0.005)
        st.caption("Fallback (when not calibrated)")
        f1, f2 = st.columns(2)
        fb_kx = f1.slider("Fallback kx", 0.5, 1.2, 0.8, 0.01)
        fb_ky = f2.slider("Fallback ky", 0.5, 1.2, 0.8, 0.01)

    st.subheader("💾 Calibration I/O")
    up = st.file_uploader("Load Calibration JSON", type=["json"], key="uploader")

    st.subheader("📷 Input Extractor")
    extractor_mode = st.selectbox("Extractor", ["Webcam/MediaPipe", "ESP32 (Orlosky)"])
    orlo_url = st.text_input("ESP32-CAM URL", STREAM_URL)
    if extractor_mode == "ESP32 (Orlosky)":
        st.info("โหมด ESP32 (Orlosky) รองรับการคาลิเบรตและ drift correction เหมือน webcam สามารถกด Calibrate ได้ตาม workflow ปกติ")

# WebRTC block (keep async_processing=False)
webrtc_ctx = webrtc_streamer(
    key="gaze-ui-mvp",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=GazeProcessor,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    },
    media_stream_constraints={
        "video": {"width": {"ideal": 960, "max": 1280}, "height": {"ideal": 540, "max": 720}, "frameRate": {"ideal": 15, "max": 30}, "facingMode": "user"},
        "audio": False,
    },
    async_processing=False,
    video_html_attrs={
        "style": {
            "width": "100%",
            "height": "400px",
            "maxHeight": "600px",
            "border": "2px solid #4CAF50",
            "borderRadius": "10px",
            "backgroundColor": "#000000",
            "objectFit": "cover",
        },
        "autoplay": True,
        "controls": False,
        "muted": True,
    },
)

if not webrtc_ctx.state.playing:
    st.info("🎥 Press **START** above to begin.")
    st.write("If you don't see video, check browser camera permission or try refresh.")
else:
    st.success("🎥 Video active. You should see HUD and a moving gaze marker.")
    if webrtc_ctx.video_processor:
        if extractor_mode == "ESP32 (Orlosky)":
            webrtc_ctx.video_processor.set_input_mode("orlosky", orlo_url)
        else:
            webrtc_ctx.video_processor.set_input_mode("webrtc")

    # Apply UI → processor
    if webrtc_ctx.video_processor:
        vp: GazeProcessor = webrtc_ctx.video_processor  # type: ignore
        vp.set_screen_size(screen_w, screen_h)
        vp.set_quality_thresh(qth)
        vp.set_head_pose_norm(hp_norm)
        vp.set_gain(gain)
        vp.set_gain_axes(gain_x, gain_y)
        vp.set_shape(gamma, deadzone)
        vp.set_clamp_jump(clampj)
        vp.set_smooth(smooth_fc, smooth_beta)
        vp.set_fallback_gain(fb_kx, fb_ky)
        vp.set_mouse_config(mc_enabled, mc_dwell_enabled, float(mc_dwell_time), int(mc_dwell_radius), float(mc_alpha))

    # Live metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    if webrtc_ctx.video_processor:
        vp: GazeProcessor = webrtc_ctx.video_processor  # type: ignore
        c1.metric("Quality", f"{vp.quality:.2f}")
        c2.metric("FPS", f"{vp.fps:.1f}")
        c3.metric("Latency", f"{vp.lat_ms:.0f} ms")
        c4.metric("Dropped", f"{vp.drop_rate*100:.1f}%")
        c5.metric("Frames", f"{vp.frame_count}")

    st.markdown("---")
    tab_cal, tab_track = st.tabs(["🧭 Calibrate", "🎯 Track & Results"])

    with tab_cal:
        left, right = st.columns([2, 1])
        with left:
            num_pt = st.selectbox("Points", [9, 12], 0)
            margin = st.slider("Margin (%)", 2, 20, 10, 1) / 100.0
            dwell_ms = st.slider("Dwell (ms)", 300, 1500, 800, 50)
            rnd = st.checkbox("Randomize order", False)

            if st.button("Start Calibration", type="primary") and webrtc_ctx.video_processor:
                pts = grid_points(9 if num_pt == 9 else 12, margin)
                vp: GazeProcessor = webrtc_ctx.video_processor  # type: ignore
                vp.start_calibration(pts, dwell_ms=dwell_ms, shuffle=rnd)
                st.success("Calibration started.")

            if st.button("Stop Calibration") and webrtc_ctx.video_processor:
                vp: GazeProcessor = webrtc_ctx.video_processor  # type: ignore
                vp.stop_calibration()
                st.info("Calibration stopped.")

        with right:
            if webrtc_ctx.video_processor:
                vp: GazeProcessor = webrtc_ctx.video_processor  # type: ignore
                # Visualize current target + trail on a virtual screen
                canvas = make_screen_canvas(
                    screen_w,
                    screen_h,
                    gaze_xy=vp.gaze,
                    max_canvas_w=480,
                    trail=vp.trail,
                    calib_target=(vp.calib.points[vp.calib.idx] if (vp.calib.active and vp.calib.points and vp.calib.idx < len(vp.calib.points)) else None),
                    calib_progress=((vp.calib.idx + (1 if vp.calib.active else 0), len(vp.calib.points) if vp.calib.points else 0) if vp.calib.active else None),
                    dwell_progress=(0.0 if not vp.calib.active else max(0.0, min(1.0, (time.time() - (vp.calib.last_step_at or time.time())) * 1000.0 / max(1.0, vp.calib.dwell_ms)))),
                )
                st.image(canvas, caption="Screen Visualization", use_container_width=True)

            # Download calibration report (if any)
            if webrtc_ctx.video_processor and webrtc_ctx.video_processor.calib.report:
                rpt = webrtc_ctx.video_processor.calib.report
                st.json(rpt)
                st.download_button("Download calibration JSON", data=json.dumps(rpt, indent=2), file_name="calibration_report.json", mime="application/json")

            # Load calibration report (WIP/placeholder — engine expects live fit)
            if up is not None and webrtc_ctx.video_processor:
                try:
                    data = json.loads(up.read())
                    st.success(f"Loaded JSON keys: {list(data.keys())[:6]} ...")
                    # In this MVP we keep live calibration only; storing W for reuse would require additional glue
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")

    with tab_track:
        if webrtc_ctx.video_processor:
            vp: GazeProcessor = webrtc_ctx.video_processor  # type: ignore
            canvas = make_screen_canvas(screen_w, screen_h, gaze_xy=vp.gaze, max_canvas_w=640, trail=vp.trail)
            st.image(canvas, caption="Screen Visualization", use_container_width=True)
        st.caption("Tip: enable mouse control in the sidebar to move the OS cursor with your gaze.")

st.markdown("---")
st.caption("If video stalls: ensure only one app uses your camera, check browser permissions, and keep async_processing=False.")