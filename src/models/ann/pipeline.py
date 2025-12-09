import os
import json
import numpy as np
import argparse
import shutil

from INDEX_BUILDER_LOADER import IndexBuilderLoader
from ANN_SEARCH_SERVICE import ANNSearchService
from OUTFIT_GENERATOR import generate_outfits
from POST_SEARCH_RERANKING import OutfitReranker

# ---------------------------------------------
# Step 1. Load upstream JSON
# ---------------------------------------------
def load_items(json_path):
   
    with open(json_path, "r") as f:
        metadata_list = json.load(f)

    metadata_dict = {m["item_id"]: m for m in metadata_list}

    embedding_dim = metadata_list[0]["embedding_dim"]
    return metadata_dict, embedding_dim

# ------------------------------
# Build FAISS index FROM .NPY FILES
# ------------------------------
def build_indexes(metadata_dict, dim):

    index_paths = {
        "tops": "faiss/tops.index",
        "bottoms": "faiss/bottoms.index",
        "shoes": "faiss/shoes.index",
    }

    # Clear FAISS directory
    faiss_dir = "faiss"
    if os.path.exists(faiss_dir):
        shutil.rmtree(faiss_dir)
    os.makedirs(faiss_dir, exist_ok=True)

    # Initialize builder
    builder = IndexBuilderLoader(index_paths=index_paths, dim=dim)

    print("[Pipeline] Ingesting embeddings from .npy files ...")

    builder.ingest_items_from_metadata(metadata_dict)
    builder.save_all()

    return builder.indexes, builder.id_maps

# ---------------------------------------------
# Step 3. Run ANN search
# ---------------------------------------------
def run_search(anchor_emb, indexes, id_maps, categories, top_k, threshold):

    ann = ANNSearchService(indexes=indexes, id_maps=id_maps)

    return ann.search_multiple_categories(
        anchor_embedding=anchor_emb,
        categories=categories,
        k_per_category=top_k,
        similarity_threshold=threshold,
    )

def load_prompt_embedding(prompt: str):
    """
    Convert prompt → slug → load corresponding .npy file using fixed relative path ../../../
    """
    slug = prompt.lower().strip().replace(" ", "_")

    # 固定查找路径：../../../data/prompt_embeddings/
    base_dir = os.path.join("../../../", "data", "prompt_embeddings")

    npy_path = os.path.join(base_dir, f"{slug}.npy")

    if not os.path.exists(npy_path):
        raise FileNotFoundError(
            f"[Error] No precomputed embedding for prompt '{prompt}'. "
            f"Expected file: {npy_path}"
        )

    print(f"[Pipeline] Loaded prompt embedding: {npy_path}")
    return np.load(npy_path).astype("float32")

# ---------------------------------------------
# Main pipeline
# ---------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ANN Index Pipeline")
    parser.add_argument("--json", type=str, required=True)
    parser.add_argument("--anchor_id", type=str, required=True, help="Item ID of the anchor clothing")
    parser.add_argument("--top_k", type=int, default=5, help="Number of nearest items to return per category")
    parser.add_argument("--threshold", type=float, default=0.0, help="Minimum similarity required to return a match")
    parser.add_argument("--prompt", type=str, default="", help="User text prompt for outfit generation")
    args = parser.parse_args()

    metadata_dict, embedding_dim = load_items(args.json)

    # 2) Load anchor embedding
    embedding_rel = metadata_dict[args.anchor_id]["embedding_path"]
    embedding_abs = os.path.join("../../../", embedding_rel)

    anchor_emb = np.load(embedding_abs).astype("float32")
    anchor_cat = metadata_dict[args.anchor_id]["category"]
    
    # 3) Build FAISS index
    indexes, id_maps = build_indexes(metadata_dict, embedding_dim)

    # 4) Decide search categories
    if anchor_cat == "top":
        cats = ["bottoms", "shoes"]
    elif anchor_cat == "bottom":
        cats = ["tops", "shoes"]
    else:
        cats = ["tops", "bottoms"]

    # 5) ANN search
    ann_service = ANNSearchService(indexes=indexes, id_maps=id_maps)
    ann_results = ann_service.search_multiple_categories(anchor_emb, cats, args.top_k, args.threshold)

    print("\nANN RESULTS:")
    print(json.dumps(ann_results, indent=2))
    
    # 6) Generate all outfit combinations
    outfits = generate_outfits(
        anchor_id=args.anchor_id,
        metadata_dict=metadata_dict,
        ann_results=ann_results,
    )

    print(f"\n[Pipeline] Generated {len(outfits)} outfit candidates.")

    # 7) Load prompt embedding from precomputed .npy file
    if args.prompt.strip():
        try:
            prompt_emb = load_prompt_embedding(args.prompt)
        except Exception as e:
            print(f"[Warning] Could not load embedding for prompt '{args.prompt}'. Error: {e}")
            prompt_emb = None
    else:
        prompt_emb = None
  
    # 8) Rerank
    if prompt_emb is not None:
        reranker = OutfitReranker(metadata_dict=metadata_dict)
        ranked = reranker.rerank(outfits, prompt_emb)

        print("\n🎯 Top ranked outfit:")
        print(ranked[0])
    else:
        print("\nNo prompt given → skipping reranking")

if __name__ == "__main__":
    main()
