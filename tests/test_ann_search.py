"""
Test ANN Search Service
"""

import os
import sys
import numpy as np
import faiss

# -------------------------------------------------------------------
# Ensure project root in sys.path
# -------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f">>> PROJECT ROOT = {PROJECT_ROOT}")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(f">>> sys.path = {sys.path[:3]}")

# -------------------------------------------------------------------
# Import target module
# -------------------------------------------------------------------
from src.ann_search.ANN_SEARCH_SERVICE import ANNSearchService


def test_ann_search():
    print("\n[TEST] Running ANN Search Service Test...\n")

    dim = 512  # CLIP embedding dimension

    # ----------------------------------------------------------
    # 1. Create FAISS indexes for each category
    # ----------------------------------------------------------
    index_tops = faiss.IndexFlatL2(dim)
    index_bottoms = faiss.IndexFlatL2(dim)
    index_shoes = faiss.IndexFlatL2(dim)

    # ----------------------------------------------------------
    # 2. Fake 512-d vectors (normalized)
    # ----------------------------------------------------------
    np.random.seed(42)

    emb_top = np.random.rand(dim).astype("float32")
    emb_top /= np.linalg.norm(emb_top)

    emb_bottom = np.random.rand(dim).astype("float32")
    emb_bottom /= np.linalg.norm(emb_bottom)

    emb_shoe = np.random.rand(dim).astype("float32")
    emb_shoe /= np.linalg.norm(emb_shoe)

    # Add to FAISS (as 2D arrays)
    index_tops.add(emb_top.reshape(1, -1))
    index_bottoms.add(emb_bottom.reshape(1, -1))
    index_shoes.add(emb_shoe.reshape(1, -1))

    indexes = {
        "tops": index_tops,
        "bottoms": index_bottoms,
        "shoes": index_shoes,
    }

    # ----------------------------------------------------------
    # 3. Build id_maps
    # (FAISS assigns vector ID = insertion order)
    # ----------------------------------------------------------
    id_maps = {
        "tops": {0: "top_001"},
        "bottoms": {0: "bottom_001"},
        "shoes": {0: "shoe_001"},
    }

    # ----------------------------------------------------------
    # 4. Initialize ANN Service
    # ----------------------------------------------------------
    ann = ANNSearchService(indexes=indexes, id_maps=id_maps)

    # ----------------------------------------------------------
    # 5. Use "top" embedding as anchor to search bottoms + shoes
    # ----------------------------------------------------------
    categories = ["bottoms", "shoes"]

    ann_results = ann.search_multiple_categories(
        anchor_embedding=emb_top,
        categories=categories,
        k_per_category=1
    )

    print("\nANN SEARCH RESULTS:")
    print(ann_results)

    # ----------------------------------------------------------
    # 6. Assertions
    # ----------------------------------------------------------
    assert "bottoms" in ann_results
    assert "shoes" in ann_results

    assert ann_results["bottoms"]["results"][0]["item_id"] == "bottom_001"
    assert ann_results["shoes"]["results"][0]["item_id"] == "shoe_001"

    print("\n[TEST PASSED] ANN Search works correctly!\n")


if __name__ == "__main__":
    test_ann_search()
