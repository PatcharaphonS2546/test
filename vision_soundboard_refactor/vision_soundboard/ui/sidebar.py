
import time
import streamlit as st
from .state import AppState
from ..utils.screen import safe_screen_size

def sidebar(app: AppState):
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 Calibration & Gaze State")
    # calib_qualities
    if hasattr(app, 'calib_qualities'):
        st.sidebar.write("Calibration Qualities:")
        app.calib_qualities = [
            st.sidebar.slider(f"Quality {i+1}", 0.0, 1.0, q, 0.01, key=f"calib_q_{i}")
            for i, q in enumerate(app.calib_qualities)
        ]
    # gaze_history
    if hasattr(app, 'gaze_history'):
        st.sidebar.write("Gaze History (last 5):")
        [st.sidebar.text(f"({pt[0]:.3f}, {pt[1]:.3f})") for pt in app.gaze_history[-5:]]
    # targets
    if hasattr(app, 'targets'):
        st.sidebar.write("Calibration Targets:")
        cols = st.sidebar.columns(2)
        app.targets = [
            (cols[0].number_input(f"Target X {i+1}", 0.0, 1.0, t[0], 0.01, key=f"target_x_{i}"),
             cols[1].number_input(f"Target Y {i+1}", 0.0, 1.0, t[1], 0.01, key=f"target_y_{i}"))
            for i, t in enumerate(app.targets)
        ]
    # radius_norm
    if hasattr(app, 'radius_norm'):
        app.set_param('radius_norm', st.sidebar.slider("Radius Norm", 0.005, 0.1, app.radius_norm, 0.001))
    # idx
    if hasattr(app, 'idx'):
        max_idx = len(app.targets)-1 if hasattr(app, 'targets') and len(app.targets) > 0 else 0
        safe_idx = min(app.idx, max_idx)
        app.set_param('idx', st.sidebar.number_input("Calibration Index", 0, max_idx, safe_idx))
    # countdown_active
    if hasattr(app, 'countdown_active'):
        app.set_param('countdown_active', st.sidebar.checkbox("Countdown Active", value=app.countdown_active))
    # countdown_end
    if hasattr(app, 'countdown_end'):
        app.set_param('countdown_end', st.sidebar.number_input("Countdown End (timestamp)", value=app.countdown_end))
    # calib_overlay
    if hasattr(app, 'calib_overlay'):
        app.set_param('calib_overlay', st.sidebar.checkbox("Calibration Overlay", value=app.calib_overlay))
    # dwell_ms
    if hasattr(app, 'dwell_ms'):
        app.set_param('dwell_ms', st.sidebar.slider("Dwell per target (ms)", 400, 2000, app.dwell_ms, 50))
    # countdown_secs
    if hasattr(app, 'countdown_secs'):
        app.set_param('countdown_secs', st.sidebar.slider("Countdown before start (s)", 0, 5, app.countdown_secs, 1))
    # screen_w, screen_h
    if hasattr(app, 'screen_w'):
        app.set_param('screen_w', st.sidebar.number_input("Screen Width", min_value=320, max_value=10000, value=int(app.screen_w)))
    if hasattr(app, 'screen_h'):
        app.set_param('screen_h', st.sidebar.number_input("Screen Height", min_value=240, max_value=10000, value=int(app.screen_h)))
    # deg_per_px
    if hasattr(app, 'deg_per_px'):
        app.set_param('deg_per_px', st.sidebar.number_input("deg per px", min_value=0.0, max_value=1.0, value=float(app.deg_per_px), step=0.001, format="%.3f"))
    # soundboard_on
    if hasattr(app, 'soundboard_on'):
        app.set_param('soundboard_on', st.sidebar.checkbox("Soundboard On", value=app.soundboard_on))
    # sound_dwell_ms
    if hasattr(app, 'sound_dwell_ms'):
        app.set_param('sound_dwell_ms', st.sidebar.slider("Sound Dwell (ms)", 1000, 5000, app.sound_dwell_ms, 250))
    # sound_cooldown_ms
    if hasattr(app, 'sound_cooldown_ms'):
        app.set_param('sound_cooldown_ms', st.sidebar.slider("Sound Cooldown (ms)", 500, 5000, app.sound_cooldown_ms, 250))
    # sound_labels
    if hasattr(app, 'sound_labels'):
        st.sidebar.write("Soundboard Labels:")
        sound_labels = [
            st.sidebar.text_input(f"Label {i+1}", lbl, key=f"sb_lbl2_{i}")
            for i, lbl in enumerate(app.sound_labels)
        ]
        app.set_param('sound_labels', sound_labels)
    # sound_files
    if hasattr(app, 'sound_files'):
        st.sidebar.write("Soundboard Files:")
        sound_files = [
            st.sidebar.text_input(f"File {i+1}", f, key=f"sb_file2_{i}")
            for i, f in enumerate(app.sound_files)
        ]
        app.set_param('sound_files', sound_files)
    # sound_icons
    if hasattr(app, 'sound_icons'):
        st.sidebar.write("Soundboard Icons:")
        sound_icons = [
            st.sidebar.text_input(f"Icon {i+1}", ic, key=f"sb_icon_{i}")
            for i, ic in enumerate(app.sound_icons)
        ]
        app.set_param('sound_icons', sound_icons)
    # sound_colors
    if hasattr(app, 'sound_colors'):
        st.sidebar.write("Soundboard Colors:")
        sound_colors = [
            (st.sidebar.number_input(f"Color R {i+1}", 0, 255, col[0], key=f"sb_col_r_{i}"),
             st.sidebar.number_input(f"Color G {i+1}", 0, 255, col[1], key=f"sb_col_g_{i}"),
             st.sidebar.number_input(f"Color B {i+1}", 0, 255, col[2], key=f"sb_col_b_{i}"))
            for i, col in enumerate(app.sound_colors)
        ]
        app.set_param('sound_colors', sound_colors)
    # _sound_current_idx
    if hasattr(app, '_sound_current_idx'):
        app.set_param('_sound_current_idx', st.sidebar.number_input("Current Soundboard Index", min_value=-1, max_value=5, value=app._sound_current_idx if app._sound_current_idx is not None else -1))
    # _sound_start
    if hasattr(app, '_sound_start'):
        app.set_param('_sound_start', st.sidebar.number_input("Soundboard Start Time", value=app._sound_start))
    # _sound_last_play
    if hasattr(app, '_sound_last_play'):
        st.sidebar.write("Soundboard Last Play Times:")
        app._sound_last_play = [
            st.sidebar.number_input(f"Last Play {i+1}", value=t, key=f"sb_lastplay_{i}")
            for i, t in enumerate(app._sound_last_play)
        ]
    st.sidebar.subheader("🎯 Compensation (yaw/pitch)")
    st.sidebar.caption("ปรับค่าชดเชยการหมุนศีรษะ (yaw/pitch)")
    app.set_param('k_yaw', float(st.sidebar.slider("k_yaw",   0.0, 1.0, app.k_yaw,   0.01)))
    app.set_param('c_yaw', float(st.sidebar.slider("c_yaw",   0.0, 0.5, app.c_yaw,   0.01)))
    app.set_param('k_pitch', float(st.sidebar.slider("k_pitch", 0.0, 1.0, app.k_pitch, 0.01)))
    app.set_param('c_pitch', float(st.sidebar.slider("c_pitch", 0.0, 0.5, app.c_pitch, 0.01)))

    st.sidebar.subheader("🧮 Bias & Auto-Gain")
    c1, c2, c3 = st.sidebar.columns(3)
    if c1.button("Set Center Bias"):
        app.set_param('bias_x', 0.9*app.bias_x + 0.1*(0.5 - float(app.gx)))
        app.set_param('bias_y', 0.9*app.bias_y + 0.1*(0.5 - float(app.gy)))
        st.toast("Center bias updated")
    if c2.button("Reset Auto-Gain"):
        app.set_param('autogain_reset', True)
    app.set_param('autogain_freeze', c3.toggle("Freeze Auto-Gain", value=app.autogain_freeze))

    st.sidebar.subheader("📈 Shaping (per-axis)")
    cols = st.sidebar.columns(2)
    app.set_param('gamma_x', float(cols[0].slider("Gamma X", 0.5, 2.0, app.gamma_x, 0.05)))
    app.set_param('gamma_y', float(cols[1].slider("Gamma Y", 0.5, 2.0, app.gamma_y, 0.05)))
    app.set_param('saccade_aware', st.sidebar.toggle("Saccade-aware smoothing", value=app.saccade_aware))

    st.sidebar.title("⚙️ Controls")
    app.set_param('mode', st.sidebar.selectbox("Mode", ["Webcam/MediaPipe"], index=0))
    app.set_param('mouse_enabled', st.sidebar.toggle("Enable Mouse Control (pointer only — no click)", value=app.mouse_enabled))

    st.sidebar.subheader("🖥️ Display / Screen")
    app.set_param('use_screen_override', st.sidebar.checkbox("Override screen size (px)", value=app.use_screen_override))
    colsw, colsh = st.sidebar.columns(2)
    app.set_param('screen_w', int(colsw.number_input("Width", min_value=320, max_value=10000, value=int(app.screen_w))))
    app.set_param('screen_h', int(colsh.number_input("Height", min_value=240, max_value=10000, value=int(app.screen_h))))
    sw_os, sh_os = safe_screen_size()
    st.sidebar.caption(f"OS reports: {sw_os}×{sh_os}px")

    st.sidebar.subheader("🎛 Shaping")
    app.set_param('gain', float(st.sidebar.slider("Gain", 0.5, 2.5, app.gain, 0.05)))
    app.set_param('gamma', float(st.sidebar.slider("Gamma", 0.5, 2.0, app.gamma, 0.05)))
    app.set_param('deadzone', float(st.sidebar.slider("Deadzone", 0.0, 0.1, app.deadzone, 0.005)))

    st.sidebar.subheader("🪞 Mirror / Invert")
    app.set_param('mirror', st.sidebar.checkbox("Mirror webcam image", value=app.mirror))
    app.set_param('invert_x', st.sidebar.checkbox("Invert X (x→1−x)", value=app.invert_x))
    app.set_param('invert_y', st.sidebar.checkbox("Invert Y (y→1−y)", value=app.invert_y))

    st.sidebar.markdown("---")
    st.sidebar.subheader("Optional: Degrees")
    app.set_param('deg_per_px', float(st.sidebar.number_input("deg per px (คูณเพื่อแปลงค่า px→deg)", min_value=0.0, max_value=1.0, value=float(app.deg_per_px), step=0.001, format="%.3f")))

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔊 Thai Soundboard (2×3)")
    app.set_param('soundboard_on', st.sidebar.toggle("Enable gaze soundboard overlay", value=app.soundboard_on))
    app.set_param('sound_dwell_ms', int(st.sidebar.slider("Dwell to speak (ms)", 1000, 5000, app.sound_dwell_ms, 250)))
    app.set_param('sound_cooldown_ms', int(st.sidebar.slider("Cooldown (ms)", 500, 5000, app.sound_cooldown_ms, 250)))
    # อัปเดต sound_labels/sound_files ผ่าน set_param ด้านบนแล้ว ไม่ต้องอัปเดตซ้ำ
    # หากต้องการให้แก้ไขเฉพาะ index ให้ใช้ set_param หลัง loop

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧭 Calibration (9 points)")
    app.set_param('countdown_secs', int(st.sidebar.slider("Countdown before start (s)", 0, 5, app.countdown_secs, 1, key="countdown_secs_slider")))
    app.set_param('dwell_ms', st.sidebar.slider("Dwell per target (ms)", 400, 2000, app.dwell_ms, 50, key="dwell_ms_slider"))
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
