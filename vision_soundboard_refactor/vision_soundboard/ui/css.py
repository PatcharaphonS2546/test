
import streamlit as st

def overlay_css(show_fullscreen_video: bool):
    if show_fullscreen_video:
        st.markdown(        """        <style>
          section[data-testid="stSidebar"] { display:none!important; }
          header, footer { display:none!important; }
          video { position:fixed!important; inset:0!important; width:100vw!important; height:100vh!important;
                  object-fit:cover!important; z-index:9999!important; }
        </style>
        """, unsafe_allow_html=True)
