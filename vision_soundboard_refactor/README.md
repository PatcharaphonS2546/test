
# Vision Mouse (Refactor)
A modular refactor of the single-file Streamlit/WebRTC demo into a small package that is easier to develop and extend.

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py
```
> If `mediapipe` or `opencv-python` are missing, the app will still start in a degraded mode (pointer stays at center).

## Layout
- `app.py` — Streamlit entry point
- `vision_mouse/` — package with submodules
  - `engine/` — filters, feature extractor, gaze engine, mouse control, math, autogain
  - `metrics/` — metrics recorder + CSV/PNG export
  - `ui/` — webrtc processor, sidebar controls, CSS overlays
  - `utils/` — small helpers (screen size, geometry)
  - `audio/` — non-blocking audio playback

The refactor keeps the original behavior:
- 9-point calibration with countdown, dwell ring, and report (RMSE/CV/Uniformity)
- One-Euro smoothing with quality-aware tuning
- Auto-gain per axis, bias, gamma shaping
- Pointer-only mouse control (no click)
- Metrics recording, CSV export, and PNG chart export (matplotlib)
- Optional Thai 2x3 "gaze soundboard" overlay with dwell-to-speak

## Notes
- WebRTC works only in a real browser session (localhost). If the camera preview doesn't show, confirm browser permissions and install `opencv-python` and `mediapipe`.
- PNG export requires `matplotlib`.
