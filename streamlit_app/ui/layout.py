import streamlit as st

def load_banner():
    st.markdown("""
        <div class="logo-container">
            <h1 class="app-title">🧥 SnapStyle - Smart Wardrobe Assistant</h1>
            <p class="app-subtitle">Your AI-powered guide to effortless dressing.</p>
        </div>
    """, unsafe_allow_html=True)

def spinner(msg):
    st.markdown(
        f"""
        <div style="padding:2rem;text-align:center">
            <div class="loader"></div>
            <p>{msg}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
