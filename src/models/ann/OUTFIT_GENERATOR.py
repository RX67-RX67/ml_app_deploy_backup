import uuid
import numpy as np
import os

CATEGORY_NORMALIZE = {
    "top": "tops",
    "bottom": "bottoms",
    "shoe": "shoes",
}

OUTFIT_STRUCTURE = {
    "tops": ["tops", "bottoms", "shoes"],
    "bottoms": ["tops", "bottoms", "shoes"],
    "shoes": ["tops", "bottoms", "shoes"]
}

def normalize(cat):
    return CATEGORY_NORMALIZE.get(cat, cat)

def generate_outfits(anchor_id, metadata_dict, ann_results):
    print("\n==================== generate_outfits DEBUG START ====================\n")

    # --------------------------------------------------
    # 1. anchor item 
    # --------------------------------------------------
    anchor_item = metadata_dict.get(anchor_id)
    print("[DEBUG] Anchor item =", anchor_item)

    if anchor_item is None:
        raise ValueError(f"Anchor id {anchor_id} not found in metadata")

    anchor_cat = normalize(anchor_item["category"])
    print("[DEBUG] Normalized anchor category =", anchor_cat)

    required_cats = OUTFIT_STRUCTURE[anchor_cat]
    print("[DEBUG] Outfit requires categories =", required_cats)

    # --------------------------------------------------
    # load all embeddings
    # --------------------------------------------------
    embeddings = {}

    for item_id, meta in metadata_dict.items():

        if "embedding_path" not in meta:
            continue

        emb_path = meta["embedding_path"]

        if not os.path.exists(emb_path):
            emb_path = os.path.join("../../../", emb_path)

        # load .npy
        try:
            emb = np.load(emb_path).astype("float32")
            embeddings[item_id] = emb
        except Exception as e:
            print(f"[ERROR] Failed loading embedding for {item_id}: {e}")

    # --------------------------------------------------
    # search ANN candidates for each required category
    # --------------------------------------------------
    category_candidates = {}

    for cat in required_cats:
        if cat == anchor_cat:
            continue  # anchor 

        ann_list = ann_results.get(cat, {}).get("results", [])

        valid_ids = [
            r["item_id"]
            for r in ann_list
            if r["item_id"] in embeddings
        ]

        print(f"[DEBUG] ANN candidates for category '{cat}': {valid_ids}")
        category_candidates[cat] = valid_ids

    # --------------------------------------------------
    # check if any category has no candidates
    # --------------------------------------------------
    from itertools import product

    other_cats = [cat for cat in required_cats if cat != anchor_cat]
    candidate_lists = [category_candidates[cat] for cat in other_cats]

    print("[DEBUG] Candidate lists =", candidate_lists)

    if any(len(lst) == 0 for lst in candidate_lists):
        print("[WARNING] Some category has NO candidates → NO OUTFIT GENERATED")
        print("\n==================== generate_outfits DEBUG END ====================\n")
        return []

    # --------------------------------------------------
    # generate all outfit combinations
    # --------------------------------------------------
    outfits = []

    for combo in product(*candidate_lists):
        outfit = {
            "outfit_id": uuid.uuid4().hex,
            "items": {}
        }

        # anchor item
        outfit["items"][anchor_cat] = anchor_id

        # other categories combine
        for cat, item_id in zip(other_cats, combo):
            outfit["items"][cat] = item_id

        outfits.append(outfit)

    print(f"[DEBUG] Generated {len(outfits)} outfits")
    print("\n==================== generate_outfits DEBUG END ====================\n")

    return outfits