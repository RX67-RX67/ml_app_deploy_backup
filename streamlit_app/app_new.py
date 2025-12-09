import os
import sys
import streamlit as st
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.snapstyle_pipeline import SnapStylePipeline


# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="SnapStyle - Smart Wardrobe Assistant",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🧥",
)


# -------------------------
# GLOBAL PIPELINE (cached)
# -------------------------
@st.cache_resource
def load_pipeline():
    return SnapStylePipeline()

pipeline = load_pipeline()


# -------------------------
# UI BANNER
# -------------------------
def load_banner():
    st.markdown(
        """
        <div style='text-align:center;margin-top:20px;'>
            <h1 style='font-size:2.5rem;font-weight:700;'>🧥 SnapStyle - Smart Wardrobe Assistant</h1>
            <p style='color:#6b7280;'>Your AI-powered guide to effortless dressing.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------
# DIGITIZE CLOSET TAB
# -------------------------
def digitize_closet_tab():
    st.header("📸 Digitize Your Closet")

    with st.form("upload_form"):
        uploaded_file = st.file_uploader(
            "Upload clothing image:",
            type=["jpg", "jpeg", "png"]
        )
        submitted = st.form_submit_button("⬆️ Upload & Digitize")

    if submitted and uploaded_file:
        st.info("Running YOLO detection + CLIP Embedding ...")

        # --- pipeline 本身已经会自动把 UploadedFile 保存到 /tmp ---
        with st.spinner("Analyzing image ..."):
            digitized_items = pipeline.digitize_image(uploaded_file)

        st.success(f"Digitized {len(digitized_items)} items!")
        st.json(digitized_items)


# -------------------------
# OUTFIT PLANNER TAB
# -------------------------
def outfit_planner_tab():
    st.header("✨ Outfit Planner")

    metadata = pipeline.metadata_dict

    if len(metadata) == 0:
        st.warning("You must first digitize some clothing items before generating outfits.")
        return

    item_ids = list(metadata.keys())
    anchor_id = st.selectbox("Choose an anchor item:", item_ids)

    anchor_meta = metadata[anchor_id]
    st.write("Selected Item Category:", anchor_meta["category"])

    crop_path = anchor_meta.get("crop_path")
    if crop_path and os.path.exists(crop_path):
        st.image(crop_path, caption="Anchor Item", width=250)

    prompt = st.text_input("Describe the look you want:", "formal interview look")

    if st.button("🚀 Generate Outfits"):
        with st.spinner("Finding compatible items ..."):
            ranked_outfits = pipeline.recommend_outfits(
                anchor_id=anchor_id,
                prompt_text=prompt,
                ann_top_k=5
            )

        st.success("Outfits generated!")

        top_score, top_outfit = ranked_outfits[0]
        st.subheader("🏆 Best Outfit")
        st.write(f"Score: **{top_score:.3f}**")
        st.json(top_outfit)


# -------------------------
# MAIN APP
# -------------------------
def main():

    load_banner()

    tab_home, tab_digitize, tab_outfit = st.tabs(
        ["🏠 Home", "📸 Digitize Closet", "✨ Outfit Planner"]
    )

    # -------------------------
    # HOME TAB
    # -------------------------
    with tab_home:
        st.write("Welcome to SnapStyle — your AI wardrobe companion!")

        st.write("---")
        st.subheader("🧹 Data Management")

        if st.button("❌ Clear ALL digitized data (reset system)"):
            import shutil

            HF_DATA_ROOT = "/data/snapstyle"

            folders = [
                f"{HF_DATA_ROOT}/embeddings",
                f"{HF_DATA_ROOT}/crops",
                f"{HF_DATA_ROOT}/faiss",
                "/tmp/snapstyle/uploads"
            ]

            for folder in folders:
                if os.path.exists(folder):
                    shutil.rmtree(folder)

            # recreate dirs
            os.makedirs(f"{HF_DATA_ROOT}/embeddings", exist_ok=True)
            os.makedirs(f"{HF_DATA_ROOT}/crops", exist_ok=True)
            os.makedirs(f"{HF_DATA_ROOT}/faiss", exist_ok=True)
            os.makedirs("/tmp/snapstyle/uploads", exist_ok=True)

            # clear cache
            load_pipeline.clear()

            st.success("All data cleared! Please refresh the page.")

    # -------------------------
    # DIGITIZE TAB
    # -------------------------
    with tab_digitize:
        digitize_closet_tab()

    # -------------------------
    # OUTFIT PLANNER TAB
    # -------------------------
    with tab_outfit:
        outfit_planner_tab()


if __name__ == "__main__":
    main()
