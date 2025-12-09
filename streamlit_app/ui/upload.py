import streamlit as st
import requests
from datetime import datetime
import random

BACKEND_URL = "http://snapstyle-backend:8000"

def render_upload_ui():

    st.markdown("## 📸 Digitize Your Closet (Layer 1)")

    # ---------------- FORM ---------------- #
    with st.form("item_upload_form"):
        uploaded_file = st.file_uploader("Upload clothing item image", type=["png", "jpg", "jpeg"])
        item_name = st.text_input("Item Name (optional)")
        submitted = st.form_submit_button("⬆️ Upload & Analyze")

    # ---------------- SUBMIT ---------------- #
    if submitted:
        if uploaded_file is None:
            st.error("Please upload an image.")
            return

        st.session_state.uploaded_file = uploaded_file
        st.session_state.item_name = item_name
        st.session_state.is_loading = True
        st.rerun()

    # ---------------- PROCESSING ---------------- #
    if st.session_state.get("is_loading"):
        st.info("Processing image with YOLO + CLIP embedding...")

        try:
            file = st.session_state.uploaded_file

            resp = requests.post(
                f"{BACKEND_URL}/yolo/embed",
                files={"file": (file.name, file.getvalue(), file.type)}
            )

            st.session_state.is_loading = False

            if resp.status_code != 200:
                st.error(resp.text)
                return

            data = resp.json()
            st.session_state.detected_items = data["items"]

            # ---------------- STORE ITEMS ---------------- #
            for item in data["items"]:
                st.session_state.wardrobe_items.append({
                    "item_id": item["item_id"],  # ⭐ 必须加入
                    "name": st.session_state.item_name or item["category"],
                    "category": item["category"],
                    "color": "Unknown",
                    "texture": "Unknown",
                    "uploaded_at": datetime.now().isoformat(),
                    "style_score": round(random.uniform(75, 95), 1),
                    "embedding_path": item["embedding_path"],
                    "crop_path": item["crop_path"],
                })

            st.success("Items added to closet! 🎉")
            st.rerun()

        except Exception as e:
            st.session_state.is_loading = False
            st.error(f"Error: {e}")

    # ---------------- DISPLAY YOLO CROPS ---------------- #
    if "detected_items" in st.session_state:
        st.markdown("### ✂️ YOLO Cropped Items:")
        for item in st.session_state.detected_items:
            st.image(item["crop_path"], caption=item["category"])
