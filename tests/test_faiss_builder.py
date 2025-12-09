"""
Unit test for IndexBuilderLoader (final cleaned version)
"""

import os
import sys
import numpy as np

# -------------------------------------------------------------------
# Ensure project root added to sys.path
# -------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f">>> PROJECT ROOT = {PROJECT_ROOT}")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print(f">>> sys.path = {sys.path[:3]}")

# -------------------------------------------------------------------
# Import module under test
# -------------------------------------------------------------------
from src.index_builder.INDEX_BUILDER_LOADER import IndexBuilderLoader


def test_index_builder_loader():
    """
    Test FAISS ingestion + save_all functionality
    """

    print("\n[TEST] Running IndexBuilderLoader test...\n")

    # -------------------------------
    # 1. Prepare temp embedding files
    # -------------------------------
    tmp_dir = os.path.join(PROJECT_ROOT, "tests/tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    emb_paths = []
    for i in range(3):
        tmp_emb_path = os.path.join(tmp_dir, f"emb_{i}.npy")
        np.save(tmp_emb_path, np.random.rand(512).astype("float32"))
        emb_paths.append(tmp_emb_path)

    # -------------------------------
    # 2. Fake metadata
    # -------------------------------
    metadata = {
        "item_1": {
            "item_id": "item_1",
            "category": "top",
            "embedding_path": emb_paths[0]  # absolute path
        },
        "item_2": {
            "item_id": "item_2",
            "category": "bottom",
            "embedding_path": emb_paths[1]
        },
        "item_3": {
            "item_id": "item_3",
            "category": "shoe",
            "embedding_path": emb_paths[2]
        },
    }

    # -------------------------------
    # 3. Prepare FAISS directories
    # -------------------------------
    faiss_dir = os.path.join(PROJECT_ROOT, "faiss")
    if os.path.exists(faiss_dir):
        # clean old files
        for f in os.listdir(faiss_dir):
            os.remove(os.path.join(faiss_dir, f))
    else:
        os.makedirs(faiss_dir, exist_ok=True)

    index_paths = {
        "tops": os.path.join(faiss_dir, "tops.index"),
        "bottoms": os.path.join(faiss_dir, "bottoms.index"),
        "shoes": os.path.join(faiss_dir, "shoes.index"),
    }

    # -------------------------------
    # 4. Build and ingest
    # -------------------------------
    builder = IndexBuilderLoader(
        index_paths=index_paths,
        dim=512,
        index_factory_string="Flat",
    )

    builder.ingest_items_from_metadata(metadata)
    builder.save_all()

    # -------------------------------
    # 5. Assertions
    # -------------------------------
    assert os.path.exists(index_paths["tops"])
    assert os.path.exists(index_paths["bottoms"])
    assert os.path.exists(index_paths["shoes"])

    # Check id_maps count
    assert len(builder.id_maps["tops"]) == 1
    assert len(builder.id_maps["bottoms"]) == 1
    assert len(builder.id_maps["shoes"]) == 1

    print("[TEST PASSED] IndexBuilderLoader works correctly.\n")


if __name__ == "__main__":
    test_index_builder_loader()
