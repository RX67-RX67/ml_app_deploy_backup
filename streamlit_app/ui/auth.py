import streamlit as st
import requests
import os

AUTH_BASE_URL = os.getenv("AUTH_URL", "http://localhost:3000/api/auth")

def init_auth_session():
    """Initialize session state for auth."""
    ss = st.session_state
    ss.api = ss.get("api", requests.Session())
    ss.user = ss.get("user", None)
    ss.auth_msg = ss.get("auth_msg", "")

def _handle_auth_response(resp, default_error):
    data = resp.json()
    if resp.ok and data.get("user"):
        st.session_state.user = data["user"]
        st.session_state.auth_msg = f"Welcome, {data['user']['user_id']}!"
        return True
    st.session_state.auth_msg = data.get("error", default_error)
    return False

def auth_signup(user_id, password):
    try:
        resp = st.session_state.api.post(
            f"{AUTH_BASE_URL}/signup",
            json={"user_id": user_id, "password": password},
            timeout=8,
        )
        return _handle_auth_response(resp, "Signup failed.")
    except Exception as e:
        st.session_state.auth_msg = f"Signup failed: {e}"
        return False

def auth_login(user_id, password):
    try:
        resp = st.session_state.api.post(
            f"{AUTH_BASE_URL}/login",
            json={"user_id": user_id, "password": password},
            timeout=8,
        )
        return _handle_auth_response(resp, "Login failed.")
    except Exception as e:
        st.session_state.auth_msg = f"Login failed: {e}"
        return False

def auth_logout():
    try:
        st.session_state.api.post(f"{AUTH_BASE_URL}/logout", json={}, timeout=6)
    except:
        pass
    st.session_state.user = None
    st.session_state.auth_msg = "You’ve been logged out."

def render_auth_ui():
    """UI used inside Tab2 in app.py."""
    st.markdown("### Account")

    if st.session_state.user:
        st.success(f"Signed in as **{st.session_state.user.get('user_id', 'user')}**")
        if st.button("Log out"):
            auth_logout()
            st.rerun()
        return

    tab_signup, tab_login = st.tabs(["🆕 Sign up", "🔑 Log in"])

    with tab_signup:
        with st.form("signup_form"):
            uid = st.text_input("Create User ID")
            pw = st.text_input("Password", type="password")
            btn = st.form_submit_button("Create account")
            if btn:
                if uid and pw:
                    if auth_signup(uid.lower(), pw): st.rerun()
                else:
                    st.error("Please enter a user ID and password.")

    with tab_login:
        with st.form("login_form"):
            uid = st.text_input("User ID")
            pw = st.text_input("Password", type="password")
            btn = st.form_submit_button("Log in")
            if btn:
                if uid and pw:
                    if auth_login(uid.lower(), pw): st.rerun()
                else:
                    st.error("Please enter all fields.")

    if st.session_state.auth_msg:
        st.caption(st.session_state.auth_msg)
