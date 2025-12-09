import streamlit as st
from ui.auth import init_auth_session, render_auth_ui
from ui.upload import render_upload_ui
from ui.planner import render_planner_ui
from ui.layout import load_banner

st.set_page_config(page_title="SnapStyle", layout="wide")

def init_session():
    ss = st.session_state
    ss.setdefault("wardrobe_items", [])
    ss.setdefault("current_outfit_index", 0)
    ss.setdefault("recommended_outfits", [])  
    ss.setdefault("user_prompt", "")
    ss.setdefault("is_loading", False)

init_auth_session()
init_session()   
load_banner()

tab1, tab2, tab3 = st.tabs(["🏠 Home", "📸 Digitize Closet", "✨ Outfit Planner"])

with tab1:
    st.write("Welcome to SnapStyle!")

with tab2:
    render_auth_ui()
    render_upload_ui()

with tab3:
    render_planner_ui()
