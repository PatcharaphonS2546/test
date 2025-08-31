
import time
import streamlit as st
from .state import AppState
from ..utils.screen import safe_screen_size

def sidebar(app: AppState):
    st.sidebar.subheader("🎯 Compensation (yaw/pitch)")
    st.sidebar.caption("ปรับค่าชดเชยการหมุนศีรษะ (yaw/pitch)")
    app.k_yaw   = float(st.sidebar.slider("k_yaw",   0.0, 1.0, app.k_yaw,   0.01))
    app.c_yaw   = float(st.sidebar.slider("c_yaw",   0.0, 0.5, app.c_yaw,   0.01))
    app.k_pitch = float(st.sidebar.slider("k_pitch", 0.0, 1.0, app.k_pitch, 0.01))
    app.c_pitch = float(st.sidebar.slider("c_pitch", 0.0, 0.5, app.c_pitch, 0.01))

    st.sidebar.subheader("🧮 Bias & Auto-Gain")
    c1, c2, c3 = st.sidebar.columns(3)
    if c1.button("Set Center Bias"):
        app.bias_x = 0.9*app.bias_x + 0.1*(0.5 - float(app.gx))
        app.bias_y = 0.9*app.bias_y + 0.1*(0.5 - float(app.gy))
        st.toast("Center bias updated")
    if c2.button("Reset Auto-Gain"):
        app.autogain_reset = True
    app.autogain_freeze = c3.toggle("Freeze Auto-Gain", value=app.autogain_freeze)

    st.sidebar.subheader("📈 Shaping (per-axis)")
    cols = st.sidebar.columns(2)
    app.gamma_x = float(cols[0].slider("Gamma X", 0.5, 2.0, app.gamma_x, 0.05))
    app.gamma_y = float(cols[1].slider("Gamma Y", 0.5, 2.0, app.gamma_y, 0.05))
    app.saccade_aware = st.sidebar.toggle("Saccade-aware smoothing", value=app.saccade_aware)

    st.sidebar.title("⚙️ Controls")
    app.mode = st.sidebar.selectbox("Mode", ["Webcam/MediaPipe"], index=0)
    app.mouse_enabled = st.sidebar.toggle("Enable Mouse Control (pointer only — no click)", value=app.mouse_enabled)

    st.sidebar.subheader("🖥️ Display / Screen")
    app.use_screen_override = st.sidebar.checkbox("Override screen size (px)", value=app.use_screen_override)
    colsw, colsh = st.sidebar.columns(2)
    app.screen_w = int(colsw.number_input("Width", min_value=320, max_value=10000, value=int(app.screen_w)))
    app.screen_h = int(colsh.number_input("Height", min_value=240, max_value=10000, value=int(app.screen_h)))
    sw_os, sh_os = safe_screen_size()
    st.sidebar.caption(f"OS reports: {sw_os}×{sh_os}px")

    st.sidebar.subheader("🎛 Shaping")
    app.gain = float(st.sidebar.slider("Gain", 0.5, 2.5, app.gain, 0.05))
    app.gamma = float(st.sidebar.slider("Gamma", 0.5, 2.0, app.gamma, 0.05))
    app.deadzone = float(st.sidebar.slider("Deadzone", 0.0, 0.1, app.deadzone, 0.005))

    st.sidebar.subheader("🪞 Mirror / Invert")
    app.mirror = st.sidebar.checkbox("Mirror webcam image", value=app.mirror)
    app.invert_x = st.sidebar.checkbox("Invert X (x→1−x)", value=app.invert_x)
    app.invert_y = st.sidebar.checkbox("Invert Y (y→1−y)", value=app.invert_y)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Optional: Degrees")
    app.deg_per_px = float(st.sidebar.number_input("deg per px (คูณเพื่อแปลงค่า px→deg)", min_value=0.0, max_value=1.0, value=float(app.deg_per_px), step=0.001, format="%.3f"))

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔊 Thai Soundboard (2×3)")
    app.soundboard_on = st.sidebar.toggle("Enable gaze soundboard overlay", value=app.soundboard_on)
    app.sound_dwell_ms = int(st.sidebar.slider("Dwell to speak (ms)", 1000, 5000, app.sound_dwell_ms, 250))
    app.sound_cooldown_ms = int(st.sidebar.slider("Cooldown (ms)", 500, 5000, app.sound_cooldown_ms, 250))
    for i in range(6):
        c1, c2 = st.sidebar.columns([1,2])
        app.sound_labels[i] = c1.text_input(f"Label {i+1}", app.sound_labels[i], key=f"sb_lbl_{i}")
        app.sound_files[i] = c2.text_input(f"File {i+1}", app.sound_files[i], key=f"sb_file_{i}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧭 Calibration (9 points)")
    app.countdown_secs = int(st.sidebar.slider("Countdown before start (s)", 0, 5, app.countdown_secs, 1))
    app.dwell_ms = st.sidebar.slider("Dwell per target (ms)", 400, 2000, app.dwell_ms, 50)
    if st.sidebar.button("Start Calibration"):
        grid = [
            (0.50, 0.50),
            (0.30, 0.50), (0.70, 0.50),
            (0.50, 0.30), (0.50, 0.70),
            (0.20, 0.20), (0.80, 0.20), (0.20, 0.80), (0.80, 0.80),
        ]
        app.targets = grid
        app.idx = 0
        app.calib_overlay = False
        app.countdown_active = True
        app.countdown_end = time.time() + max(0, app.countdown_secs)
        sw, sh = safe_screen_size()
        app.radius_norm = max(0.012, 40 / max(sw, sh))
        st.toast(f"เริ่มคาลิเบรทใน {app.countdown_secs} วินาที…")
