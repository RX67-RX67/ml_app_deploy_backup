from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import os
import json
import numpy as np

from pydantic import BaseModel

# ---- internal modules ----
from .inference import process_uploaded_image
from .schemas import YoloEmbedResponse

from src.models.ann.ann_search_service import ANNSearchService
from src.models.ann.index_builder_loader import IndexBuilderLoader
from src.models.ann.outfit_generator import generate_outfits
from src.models.ann.post_search_reranking import OutfitReranker
from src.models.ann.text_embedding import CLIPTextEmbeddingExtractor



# ================================
# App
# ================================
app = FastAPI(title="SnapStyle Backend", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)


# ================================
# Paths
# ================================
META_PATH = "data/user_embeddings/metadata.json"
CROP_DIR = "data/user_crops"
INDEX_DIR = "src/models/ann/faiss"

os.makedirs(CROP_DIR, exist_ok=True)


# ================================
# Health Check
# ================================
@app.get("/health")
async def health():
    global ann_engine
    return {"status": "ok", "ann_ready": ann_engine is not None}


# ================================
# Auto FAISS Rebuild
# ================================
def rebuild_faiss_indexes():
    print("\n🔄 [FAISS] Rebuilding FAISS Indexes...")

    if not os.path.exists(META_PATH):
        print("❌ metadata.json does not exist")
        return None, None

    with open(META_PATH, "r") as f:
        metadata_list = json.load(f)

    if len(metadata_list) == 0:
        print("❌ metadata.json is empty")
        return None, None

    metadata_dict = {m["item_id"]: m for m in metadata_list}

    index_paths = {
        "tops": f"{INDEX_DIR}/tops.index",
        "bottoms": f"{INDEX_DIR}/bottoms.index",
        "shoes": f"{INDEX_DIR}/shoes.index",
    }

    loader = IndexBuilderLoader(index_paths=index_paths, dim=512)
    loader.ingest_items_from_metadata(metadata_dict)
    loader.save_all()

    print("✅ [FAISS] Rebuild complete.\n")
    return loader.indexes, loader.id_maps


# ================================
# Startup → rebuild FAISS
# ================================
@app.on_event("startup")
def startup():
    global ann_engine
    indexes, id_maps = rebuild_faiss_indexes()

    if indexes is None:
        ann_engine = None
        print("⚠ ANN disabled")
    else:
        ann_engine = ANNSearchService(indexes=indexes, id_maps=id_maps)
        print("🚀 ANN Engine Ready")


# ================================
# YOLO + CLIP Embedding
# ================================
@app.post("/yolo/embed", response_model=YoloEmbedResponse)
async def yolo_embed_endpoint(file: UploadFile = File(...)):
    global ann_engine

    result = process_uploaded_image(file)  # updates metadata.json inside

    # fix crop URLs
    for item in result["items"]:
        name = os.path.basename(item["crop_path"])
        item["crop_path"] = f"http://localhost:8000/yolo/crop/{name}"

    # rebuild FAISS after new upload
    indexes, id_maps = rebuild_faiss_indexes()
    if indexes is not None:
        ann_engine = ANNSearchService(indexes=indexes, id_maps=id_maps)
        print("✨ ANN refreshed after upload")

    return result


@app.get("/yolo/crop/{filename}")
async def get_crop(filename: str):
    path = os.path.join(CROP_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, f"{filename} not found")
    return FileResponse(path)


# ================================
# Outfit Generator
# ================================
class OutfitRequest(BaseModel):
    anchor_id: str
    prompt: str | None = None  # 用户输入 prompt


@app.post("/outfit/generate")
async def outfit_generate(req: OutfitRequest):
    global ann_engine

    if ann_engine is None:
        raise HTTPException(500, "FAISS index not ready")

    # load metadata
    with open(META_PATH, "r") as f:
        meta_list = json.load(f)
    metadata = {m["item_id"]: m for m in meta_list}

    if req.anchor_id not in metadata:
        raise HTTPException(404, f"anchor_id `{req.anchor_id}` not found")
  
    anchor_cat = metadata[req.anchor_id]["category"].lower()

    if anchor_cat not in ["top", "bottom", "shoes"]:
        
        fallback = next(
            (item_id for item_id, m in metadata.items()
             if m["category"].lower() in ["top", "bottom", "shoes"]),
            None
        )

        if fallback is None:
            raise HTTPException(400, "You must upload at least one top/bottom/shoes before generating outfits.")

        print(f"⚠️ Invalid anchor `{req.anchor_id}` (category `{anchor_cat}`) → switching to `{fallback}`")

      
        req.anchor_id = fallback



    # load anchor embedding
    emb = np.load(metadata[req.anchor_id]["embedding_path"]).astype("float32")

    # ANN search
    ann_results = ann_engine.search_multiple_categories(
        anchor_embedding=emb,
        categories=["tops", "bottoms", "shoes"],
        k_per_category=10
    )

    # generate raw outfits
    outfits = generate_outfits(req.anchor_id, metadata, ann_results)

    # ================================
    # ⭐ Prompt Embedding (optional)
    # ================================
    prompt_emb = None
    if req.prompt:
        print(f"🔮 Generating prompt embedding: {req.prompt}")
        text_model = CLIPTextEmbeddingExtractor("./local_clip")
        prompt_emb = text_model.encode(req.prompt)

    # ================================
    # ⭐ Rerank outfits
    # ================================
    reranker = OutfitReranker(metadata)
    ranked = reranker.rerank(outfits, prompt_emb=prompt_emb)

    # take top 3
    best = ranked[:3]

    # ================================
    # return in frontend friendly format
    # ================================
    result = []
    for score, outfit in best:
        formatted = {
            "outfit_id": outfit["outfit_id"],
            "score": score,
            "items": {}
        }
        for cat, item_id in outfit["items"].items():
            m = metadata[item_id]
            formatted["items"][cat] = {
                "item_id": item_id,
                "category": m["category"],
                "crop_path": f"http://localhost:8000/yolo/crop/{os.path.basename(m['crop_path'])}"
            }
        result.append(formatted)

    return {"anchor_id": req.anchor_id, "top_outfits": result}
