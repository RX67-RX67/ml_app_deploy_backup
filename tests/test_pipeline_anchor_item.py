"""
Full Pipeline Integration Test:
Anchor Item → ANN Search → OutfitGeneration → TextEmbedding → OutfitReranking
"""

import os
import sys
import numpy as np
import shutil

# -------------------------------------
# Add project root to sys.path
# -------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(">>> PROJECT ROOT =", PROJECT_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -------------------------------------
# Import pipeline modules
# -------------------------------------
from src.index_builder.INDEX_BUILDER_LOADER import IndexBuilderLoader
from src.ann_search.ANN_SEARCH_SERVICE import ANNSearchService
from src.outfit_generator.OUTFIT_GENERATOR import OutfitGenerator
from src.post_search_ranker.POST_SEARCH_RERANKING import OutfitReranker
from src.text_embedding.SIMPLECLIP import SimpleCLIPTextEncoder


def test_full_pipeline():
    print("\n====== FULL PIPELINE TEST START ======\n")

    # ----------------------------------------------------
    # 1. Create fake metadata with embeddings
    # ----------------------------------------------------
    os.makedirs("tests/tmp_emb", exist_ok=True)

    CATEGORY_LIST = ["top", "bottom", "shoe", "bottom", "top"]
    fake_metadata = {}

    for i, cat in enumerate(CATEGORY_LIST):
        emb = np.random.rand(512).astype("float32")
        emb_path = f"tests/tmp_emb/emb_{i}.npy"
        np.save(emb_path, emb)

        fake_metadata[f"item_{i}"] = {
            "item_id": f"item_{i}",
            "category": cat,
            "embedding_path": emb_path,
        }

    # ----------------------------------------------------
    # 2. Pick an anchor item (NOT text!)
    # ----------------------------------------------------
    anchor_id = "item_0"
    anchor_item = fake_metadata[anchor_id]
    anchor_cat = anchor_item["category"]

    anchor_emb = np.load(anchor_item["embedding_path"]).astype("float32")

    print(f">>> Anchor Item = {anchor_id}, category = {anchor_cat}")

    if anchor_cat == "top":
        search_categories = ["bottoms", "shoes"]
    elif anchor_cat == "bottom":
        search_categories = ["tops", "shoes"]
    else:
        search_categories = ["tops", "bottoms"]

    print(">>> Search categories:", search_categories)

    # ----------------------------------------------------
    # 3. Build FAISS index from metadata
    # ----------------------------------------------------
    if os.path.exists("faiss"):
        shutil.rmtree("faiss")
    os.makedirs("faiss", exist_ok=True)

    index_paths = {
        "tops": "faiss/tops.index",
        "bottoms": "faiss/bottoms.index",
        "shoes": "faiss/shoes.index",
    }

    builder = IndexBuilderLoader(index_paths=index_paths, dim=512)
    builder.ingest_items_from_metadata(fake_metadata)
    builder.save_all()

    # ----------------------------------------------------
    # 4. ANN search from anchor embedding
    # ----------------------------------------------------
    ann = ANNSearchService(builder.indexes, builder.id_maps)

    ann_results = ann.search_multiple_categories(
        anchor_embedding=anchor_emb,
        categories=search_categories,
        k_per_category=3,
    )

    print("\nANN Results:")
    print(ann_results)

    # Basic structure check
    for cat in search_categories:
        assert cat in ann_results

    # ----------------------------------------------------
    # 5. Generate outfits using ANN results
    # ----------------------------------------------------
    generator = OutfitGenerator(fake_metadata)
    outfits = generator.generate(anchor_id, ann_results)

    print("\nGenerated Outfits:")
    for o in outfits:
        print(o)

    if len(outfits) == 0:
        print("\n[WARNING] No outfits generated → test ends here.\n")
        return

    # ----------------------------------------------------
    # 6. NEW STEP: Text embedding for reranking
    # ----------------------------------------------------
    encoder = SimpleCLIPTextEncoder(device="cpu")
    prompt_text = "a clean formal business outfit"
    prompt_emb = encoder.encode(prompt_text)

    print("\nPrompt Embedding Norm:", np.linalg.norm(prompt_emb))
    assert prompt_emb.shape == (512,)

    # ----------------------------------------------------
    # 7. Rerank outfits with text embedding
    # ----------------------------------------------------
    reranker = OutfitReranker(fake_metadata)

    ranked = reranker.rerank(outfits, prompt_emb)

    print("\nRanked Outfits:")
    for score, outfit in ranked:
        print(f"score={score:.4f}, items={outfit}")

    # Check sorting descending
    scores = [s for s, _ in ranked]
    assert scores == sorted(scores, reverse=True)

    print("\n====== FULL PIPELINE TEST PASSED ======\n")


if __name__ == "__main__":
    test_full_pipeline()
