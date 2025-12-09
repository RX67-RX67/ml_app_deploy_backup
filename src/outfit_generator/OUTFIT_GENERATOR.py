import uuid
import numpy as np
import os
from itertools import product


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


class OutfitGenerator:
    """
    Generate outfit combinations using:
    - anchor item
    - metadata_dict (contains category + embedding_path)
    - ANN search results from ANNSearchService
    """

    def __init__(self, metadata_dict, project_root=None):
        """
        Args:
            metadata_dict: dict[item_id] -> metadata entry
            project_root: absolute path to project root (optional)
                          if None → auto-detect
        """
        self.metadata_dict = metadata_dict

        # ---------------------------------------------
        # Detect project root (same logic as test_yolo)
        # ---------------------------------------------
        if project_root is None:
            # Assume src/ is inside your project directory
            self.project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
        else:
            self.project_root = project_root

        self.embeddings = {}

        # Load all embeddings ONCE
        self._load_embeddings()

    # -------------------------------------------------------
    def _resolve_embedding_path(self, emb_path):
        """
        Convert a metadata relative path to an absolute path
        using project_root.
        """
        # If already absolute → return
        if os.path.isabs(emb_path):
            return emb_path

        candidate = os.path.join(self.project_root, emb_path)

        if os.path.exists(candidate):
            return candidate

        raise FileNotFoundError(
            f"[OutfitGenerator] Embedding file not found:\n"
            f"  metadata path: {emb_path}\n"
            f"  resolved: {candidate}\n"
            f"  project_root: {self.project_root}"
        )

    # -------------------------------------------------------
    def _load_embeddings(self):
        """Load embedding vectors from metadata.json (project-root aware)"""

        print("[OutfitGenerator] Loading embeddings with project_root =", self.project_root)

        for item_id, meta in self.metadata_dict.items():
            emb_path = meta.get("embedding_path")
            if emb_path is None:
                continue

            try:
                abs_path = self._resolve_embedding_path(emb_path)
                emb = np.load(abs_path).astype("float32")
                self.embeddings[item_id] = emb

            except Exception as e:
                print(f"[ERROR] Failed to load embedding for {item_id}: {e}")

        print(f"[OutfitGenerator] Loaded {len(self.embeddings)} embeddings.")

    # -------------------------------------------------------
    @staticmethod
    def normalize_category(cat):
        return CATEGORY_NORMALIZE.get(cat.lower().strip(), cat)

    # -------------------------------------------------------
    def generate(self, anchor_id, ann_results):
        """
        Args:
            anchor_id: item_id of the anchor item
            ann_results: ANN output from ANNSearchService.search_multiple_categories()

        Returns:
            List of outfit dicts
        """

        print("\n===== OUTFIT GENERATION START =====\n")

        # ---- Validate anchor ----
        anchor_item = self.metadata_dict.get(anchor_id)
        if anchor_item is None:
            raise ValueError(f"[OutfitGenerator] anchor_id={anchor_id} not found")

        raw_cat = anchor_item["category"]
        anchor_cat = self.normalize_category(raw_cat)
        print(f"[DEBUG] Anchor category → {anchor_cat}")

        required_cats = OUTFIT_STRUCTURE[anchor_cat]
        print(f"[DEBUG] Outfit structure requires → {required_cats}")

        # ---- Collect ANN candidates ----
        category_candidates = {}

        for cat in required_cats:
            if cat == anchor_cat:
                continue

            ann_list = ann_results.get(cat, {}).get("results", [])

            valid_ids = [
                result["item_id"]
                for result in ann_list
                if result["item_id"] in self.embeddings
            ]

            print(f"[DEBUG] Candidates for '{cat}' → {valid_ids}")
            category_candidates[cat] = valid_ids

        # ---- Check for empty categories ----
        other_cats = [c for c in required_cats if c != anchor_cat]
        candidate_lists = [category_candidates[c] for c in other_cats]

        if any(len(lst) == 0 for lst in candidate_lists):
            print("[WARNING] Some category has NO candidates → no outfit generated.")
            return []

        # ---- Generate outfit combinations ----
        outfits = []

        for combo in product(*candidate_lists):
            outfit = {"outfit_id": uuid.uuid4().hex, "items": {}}

            outfit["items"][anchor_cat] = anchor_id

            for cat, item_id in zip(other_cats, combo):
                outfit["items"][cat] = item_id

            outfits.append(outfit)

        print(f"[OutfitGenerator] Generated {len(outfits)} outfits.")
        print("\n===== OUTFIT GENERATION END =====\n")

        return outfits
