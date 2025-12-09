import streamlit as st
import requests

BACKEND_URL = "http://snapstyle-backend:8000"


# -----------------------------------
# UI: Display one outfit card
# -----------------------------------
def display_outfit_card(outfit):
    st.subheader(f"✨ Outfit ID: {outfit['outfit_id']}  — Score: {outfit.get('score', 0):.3f}")

    cols = st.columns(3)

    ordered = ["tops", "bottoms", "shoes"]

    for i, cat in enumerate(ordered):
        if cat in outfit["items"]:
            info = outfit["items"][cat]

            with cols[i]:
                st.markdown(f"### {cat.capitalize()}")
                st.image(info["crop_path"], width=180)
                st.markdown(f"**Item ID:** `{info['item_id']}`")
    st.markdown("---")


# -----------------------------------
# Main Planner UI
# -----------------------------------
def render_planner_ui():
    st.markdown("## 👗 AI Outfit Planner")

    prompt = st.text_input("Describe your look (optional, not used yet):")

    if st.button("Generate Outfits"):
        if "wardrobe_items" not in st.session_state or len(st.session_state.wardrobe_items) == 0:
            st.error("Please upload at least one clothing item first.")
            return

        anchor = st.session_state.wardrobe_items[0]
        anchor_id = anchor["item_id"]

        resp = requests.post(
            f"{BACKEND_URL}/outfit/generate",
            json={"anchor_id": anchor_id}
        )

        if resp.status_code != 200:
            st.error(resp.text)
            return

        outfits = resp.json()["outfits"]

        # -----------------------------------------
        # ⭐ Sort by score (descending)
        # -----------------------------------------
        outfits_sorted = sorted(outfits, key=lambda x: x.get("score", 0), reverse=True)

        # -----------------------------------------
        # ⭐ Take top 3 outfits
        # -----------------------------------------
        st.session_state.recommended_outfits = outfits_sorted[:3]

    # ---------------- DISPLAY ---------------- #
    if st.session_state.get("recommended_outfits"):
        st.markdown("## 🔥 Top 3 Best-Matched Outfits")
        for outfit in st.session_state.recommended_outfits:
            display_outfit_card(outfit)
