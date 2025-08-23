import av, cv2, streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

st.set_page_config(page_title="Camera Echo Test", layout="centered")

class Echo(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        cv2.putText(img, "ECHO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="echo",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=Echo,
    rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video":True,"audio":False},
    async_processing=False
)