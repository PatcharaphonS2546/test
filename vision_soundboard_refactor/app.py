
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from vision_soundboard.ui.state import AppState
from vision_soundboard.ui.sidebar import sidebar
from vision_soundboard.ui.processor import Processor
from vision_soundboard.ui.css import overlay_css
from vision_soundboard.metrics.recorder import MetricsRecorder

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

st.set_page_config(page_title="Vision Mouse", page_icon="👁️", layout="wide")
st.title("👁️ Vision Mouse — Eye-controlled pointer (Webcam)")
st.caption("Pointer-only (no click). If preview doesn't show, check camera permissions and install opencv-python / mediapipe.")

# Session state
if "APP" not in st.session_state:
    st.session_state["APP"] = AppState()
if "METRICS" not in st.session_state:
    st.session_state["METRICS"] = MetricsRecorder()
APP: AppState = st.session_state["APP"]
METRICS: MetricsRecorder = st.session_state["METRICS"]

# Sidebar controls
sidebar(APP)

# WebRTC
ctx = webrtc_streamer(
    key="vm-stream", mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video":{"width":{"min":640},"height":{"min":480},"frameRate":{"ideal":60,"min":30}}, "audio":False},
    async_processing=True, video_processor_factory=lambda: Processor(APP, METRICS),
)

overlay_css(APP.calib_overlay)

# Metrics strip
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Mode", APP.mode)
with c2: st.metric("Mouse", "ON" if APP.mouse_enabled else "OFF")
with c3: st.metric("FPS", f"{APP.ui_fps:.1f}")
with c4: st.metric("Latency (ms)", f"{APP.ui_lat:.1f}")
st.caption(f"📦 Samples collected: {len(METRICS.samples)}")

# Calibration report
if ctx and ctx.video_processor and ctx.video_processor.engine:
    eng = ctx.video_processor.engine
    rep = eng.get_report()
    st.subheader("Calibration Report")
    if not rep.get("model_ready", False):
        st.info("Model: **Uncalibrated** — pointer only (no click)")
    else:
        st.markdown(
            f"- RMSE(train): **{rep['rmse_px']:.0f}px** · RMSE(CV): **{rep['rmse_cv_px']:.0f}px** · Uniformity: **{rep['uniformity']:.2f}** · Points: **{rep['points']}**"
        )

# Charts
if pd is None:
    st.warning("Install pandas for tables/charts:  pip install pandas")
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
        st.caption("คำนวณจากตัวอย่างที่มี target จริงในจุดนั้น (n ≥ 1)")
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

    with cB:
        try:
            import matplotlib.pyplot as plt  # noqa: F401
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
        except Exception:
            st.warning("Install matplotlib to export PNGs:  pip install matplotlib")

st.markdown("""**Usage**
- Click **Start Calibration** for 9 points (with dwell ring)
- See **RMSE / CV / Uniformity / Points** after calibration
- **Metrics** and charts show **Latency / Jitter / MAE / RMSE** (global & per-point)
- **Export Webcam charts (PNG)** saves 6 PNGs under `exports/`
- Set **deg per px** in sidebar to include degree units in the summary chart
- Pointer only — no click
""")
