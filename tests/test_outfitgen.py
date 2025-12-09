"""
Test OutfitGenerator Module
----------------------------------
This test simulates:
- fake metadata.json
- fake .npy embedding files
- fake ANN search results

No YOLO / CLIP / FAISS needed.
"""

import os
import sys
import numpy as np

# -----------------------------------------------------------------------------
# Add PROJECT ROOT so "src" can be imported
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(">>> PROJECT ROOT =", PROJECT_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(">>> sys.path =", sys.path[:3])

# -----------------------------------------------------------------------------
# Import module under test
# -----------------------------------------------------------------------------
from src.outfit_generator.OUTFIT_GENERATOR import OutfitGenerator


def test_outfit_generator():
    print("\n[TEST] Running OutfitGenerator test...\n")

    # -----------------------------------------------------------------------------
    # 1. Create temp directory for fake embeddings
    # -----------------------------------------------------------------------------
    tmp_dir = os.path.join(PROJECT_ROOT, "tests", "tmp_embeddings")
    os.makedirs(tmp_dir, exist_ok=True)

    # Fake embedding paths
    emb_top = os.path.join("tests/tmp_embeddings", "top.npy")
    emb_bottom = os.path.join("tests/tmp_embeddings", "bottom.npy")
    emb_shoes = os.path.join("tests/tmp_embeddings", "shoes.npy")

    # Save fake embedding files
    np.save(os.path.join(PROJECT_ROOT, emb_top), np.random.rand(512).astype("float32"))
    np.save(os.path.join(PROJECT_ROOT, emb_bottom), np.random.rand(512).astype("float32"))
    np.save(os.path.join(PROJECT_ROOT, emb_shoes), np.random.rand(512).astype("float32"))

    # -----------------------------------------------------------------------------
    # 2. Fake metadata dict (simulating metadata.json)
    # -----------------------------------------------------------------------------
    metadata = {
        "item_top": {
            "item_id": "item_top",
            "category": "top",
            "embedding_path": emb_top,   # relative path!
        },
        "item_bottom": {
            "item_id": "item_bottom",
            "category": "bottom",
            "embedding_path": emb_bottom,
        },
        "item_shoes": {
            "item_id": "item_shoes",
            "category": "shoe",
            "embedding_path": emb_shoes,
        }
    }

    # -----------------------------------------------------------------------------
    # 3. Create fake ANN results returned from ANNSearchService
    # -----------------------------------------------------------------------------
    ann_results = {
        "bottoms": {
            "results": [
                {"item_id": "item_bottom", "similarity": 0.9},
            ]
        },
        "shoes": {
            "results": [
                {"item_id": "item_shoes", "similarity": 0.88},
            ]
        }
    }

    # -----------------------------------------------------------------------------
    # 4. Initialize generator
    # -----------------------------------------------------------------------------
    generator = OutfitGenerator(metadata_dict=metadata)

    # -----------------------------------------------------------------------------
    # 5. Generate outfits
    # -----------------------------------------------------------------------------
    outfits = generator.generate(anchor_id="item_top", ann_results=ann_results)

    print("\nGenerated outfits:")
    for outfit in outfits:
        print(outfit)

    # -----------------------------------------------------------------------------
    # 6. Assertions
    # -----------------------------------------------------------------------------
    assert len(outfits) == 1, "Expected 1 outfit combination"
    assert outfits[0]["items"]["tops"] == "item_top"
    assert outfits[0]["items"]["bottoms"] == "item_bottom"
    assert outfits[0]["items"]["shoes"] == "item_shoes"

    print("\n[TEST PASSED] OutfitGenerator works correctly!\n")


if __name__ == "__main__":
    test_outfit_generator()
