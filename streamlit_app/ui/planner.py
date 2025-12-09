import streamlit as st
import requests

BACKEND_URL = "http://snapstyle-backend:8000"


def display_outfit_card(outfit):
    st.subheader(f"✨ Outfit {outfit['outfit_id']} — Score: {outfit['score']:.3f}")

    cols = st.columns(3)
    order = ["tops", "bottoms", "shoes"]

    for i, cat in enumerate(order):
        if cat in outfit["items"]:
            item = outfit["items"][cat]
            with cols[i]:
                st.image(item["crop_path"], width=180)
                st.caption(f"{cat} — {item['item_id']}")


def render_planner_ui():
    st.markdown("## Outfit Planner — Ranked Results")

    prompt = st.text_input("Describe your style (optional):")

    if st.button("Generate Outfits"):
        if "wardrobe_items" not in st.session_state or len(st.session_state.wardrobe_items) == 0:
            st.error("Upload items first!")
            return

        anchor = st.session_state.wardrobe_items[0]
        anchor_id = anchor["item_id"]

        resp = requests.post(
            f"{BACKEND_URL}/outfit/generate",
            json={"anchor_id": anchor_id, "prompt": prompt}
        )

        if resp.status_code != 200:
            st.error(resp.text)
            return

        # ⭐ FIX: backend returns top_outfits, not outfits
        st.session_state.recommended_outfits = resp.json()["top_outfits"]

    # ---------- DISPLAY ----------
    if st.session_state.get("recommended_outfits"):
        st.markdown("## 🔥 Top 3 Best-Matched Outfits")

        for outfit in st.session_state.recommended_outfits:
            display_outfit_card(outfit)
            st.markdown("---")
