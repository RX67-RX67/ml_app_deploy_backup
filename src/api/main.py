from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import os
import json
import numpy as np

from pydantic import BaseModel

# --- Internal modules ---
from .inference import process_uploaded_image
from .schemas import YoloEmbedResponse

# ⭐ FIXED imports — must match your actual file names
from src.models.ann.ann_search_service import ANNSearchService
from src.models.ann.outfit_generator import generate_outfits
from src.models.ann.index_builder_loader import IndexBuilderLoader


ann_engine = None
# ================================
# FastAPI App
# ================================
app = FastAPI(
    title="SnapStyle Backend",
    version="3.0.0",
)

# ================================
# Enable CORS
# ================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================
# Health Check
# ================================
@app.get("/health")
async def health():
    """
    Frontend calls this to check backend status.
    """
    global ann_engine
    return {
        "status": "ok",
        "ann_ready": ann_engine is not None
    }


# ================================
# File paths
# ================================
META_PATH = "data/user_embeddings/metadata.json"
CROP_DIR = "data/user_crops"
INDEX_DIR = "src/models/ann/faiss"

os.makedirs(CROP_DIR, exist_ok=True)


# ================================
# AUTO REBUILD FAISS INDEX
# ================================
def rebuild_faiss_indexes():
    print("\n🔄 [FAISS] Rebuilding FAISS Indexes...")

    # ---- metadata.json exists? ----
    if not os.path.exists(META_PATH):
        print("❌ metadata.json not found")
        return None, None

    # ---- load metadata ----
    try:
        with open(META_PATH, "r") as f:
            metadata_list = json.load(f)
    except Exception as e:
        print(f"❌ Cannot read metadata.json: {e}")
        return None, None

    # ---- empty metadata ----
    if len(metadata_list) == 0:
        print("❌ metadata.json is EMPTY → no embeddings to index")
        return None, None

    metadata_dict = {item["item_id"]: item for item in metadata_list}

    index_paths = {
        "tops": f"{INDEX_DIR}/tops.index",
        "bottoms": f"{INDEX_DIR}/bottoms.index",
        "shoes": f"{INDEX_DIR}/shoes.index",
    }

    loader = IndexBuilderLoader(index_paths=index_paths, dim=512)
    loader.ingest_items_from_metadata(metadata_dict)
    loader.save_all()

    print("✅ [FAISS] Rebuild completed.\n")
    return loader.indexes, loader.id_maps



# ================================
# Startup Event → Auto rebuild
# ================================
@app.on_event("startup")
def startup_event():
    global ann_engine
    indexes, id_maps = rebuild_faiss_indexes()

    if indexes is None:
        ann_engine = None
        print("⚠ ANN engine DISABLED — FAISS not ready.")
        return

    ann_engine = ANNSearchService(indexes=indexes, id_maps=id_maps)
    print("🚀 ANN Engine Ready on Startup!")


# ================================
# YOLO + CLIP Embedding API
# ================================
@app.post("/yolo/embed", response_model=YoloEmbedResponse)
async def yolo_embed_endpoint(file: UploadFile = File(...)):
    """
    Upload image → YOLO detect → CLIP embed → metadata.json update → auto FAISS rebuild.
    """
    global ann_engine

    try:
        result = process_uploaded_image(file)

        # Convert crop_path to URL
        for item in result["items"]:
            filename = os.path.basename(item["crop_path"])
            item["crop_path"] = f"http://localhost:8000/yolo/crop/{filename}"

        # --------------------------------------
        # ⭐ Auto-rebuild FAISS after new upload
        # --------------------------------------
        indexes, id_maps = rebuild_faiss_indexes()

        if indexes is not None:
            ann_engine = ANNSearchService(indexes=indexes, id_maps=id_maps)
            print("✨ ANN Engine Refreshed After Upload!")

        return result

    except Exception as e:
        raise HTTPException(500, str(e))


# Serve cropped user clothing images
@app.get("/yolo/crop/{filename}")
async def get_crop_image(filename: str):
    path = os.path.join(CROP_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, f"{filename} not found")
    return FileResponse(path)


# ================================
# ANN Search
# ================================
@app.post("/ann/search")
async def ann_search(embedding: list):
    global ann_engine

    if ann_engine is None:
        raise HTTPException(500, "FAISS index not ready")

    try:
        emb = np.array(embedding, dtype="float32")
        return ann_engine.search_multiple_categories(
            anchor_embedding=emb,
            categories=["tops", "bottoms", "shoes"],
            k_per_category=10
        )

    except Exception as e:
        raise HTTPException(500, str(e))


# ================================
# Outfit Generator API
# ================================
class OutfitRequest(BaseModel):
    anchor_id: str


@app.post("/outfit/generate")
async def outfit_generate(req: OutfitRequest):
    global ann_engine

    if ann_engine is None:
        raise HTTPException(500, "FAISS index not ready")

    try:
        # Load metadata.json
        with open(META_PATH, "r") as f:
            metadata_list = json.load(f)
        metadata_dict = {item["item_id"]: item for item in metadata_list}

        if req.anchor_id not in metadata_dict:
            raise HTTPException(404, f"anchor '{req.anchor_id}' not found")

        # Load anchor embedding
        emb_path = metadata_dict[req.anchor_id]["embedding_path"]
        anchor_emb = np.load(emb_path).astype("float32")

        # ANN search
        ann_results = ann_engine.search_multiple_categories(
            anchor_embedding=anchor_emb,
            categories=["tops", "bottoms", "shoes"],
            k_per_category=10
        )

        # Outfit generation
        outfits = generate_outfits(req.anchor_id, metadata_dict, ann_results)

        # Add images for frontend display
        enhanced = []
        for outfit in outfits:
            new_outfit = {
                "outfit_id": outfit["outfit_id"],
                "items": {}
            }

            for cat, item_id in outfit["items"].items():
                meta = metadata_dict[item_id]
                new_outfit["items"][cat] = {
                    "item_id": item_id,
                    "category": meta["category"],
                    "crop_path": f"http://localhost:8000/yolo/crop/{os.path.basename(meta['crop_path'])}"
                }

            enhanced.append(new_outfit)

        return {"anchor_id": req.anchor_id, "outfits": enhanced}

    except Exception as e:
        raise HTTPException(500, str(e))
