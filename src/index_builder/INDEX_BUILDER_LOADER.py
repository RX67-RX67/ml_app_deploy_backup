import os
import faiss
import json
import numpy as np
from typing import Dict


class IndexBuilderLoader:
    """
    FAISS Index Builder & Loader
    --------------------------------

    Responsibilities:
    - Load existing FAISS indexes OR create new ones
    - Ingest embeddings from metadata and build vector indexes
    - Store id_maps (FAISS vector_id → item_id)
    - Save indexes and id_maps to disk

    NOT responsible for:
    - Deciding embedding path structure
    - Running YOLO / CLIP / text encoding
    """

    def __init__(self, index_paths: Dict[str, str], dim: int, index_factory_string="Flat"):
        """
        Args:
            index_paths: dict mapping category → faiss index path
                Example:
                    {
                        "tops": "faiss/tops.index",
                        "bottoms": "faiss/bottoms.index",
                        "shoes": "faiss/shoes.index"
                    }

            dim: embedding dimension
            index_factory_string: FAISS index type (default "Flat")
        """

        self.index_paths = index_paths
        self.dim = dim
        self.factory = index_factory_string

        self.indexes = {}
        self.id_maps = {}

        self._load_or_create_indexes()
        self._load_or_create_id_maps()

    # ------------------------------------------------------------------
    # Build FAISS index
    # ------------------------------------------------------------------
    def _build_new_index(self):
        print(f"[IndexBuilder] Creating new FAISS index ({self.factory}, dim={self.dim})")
        return faiss.index_factory(self.dim, self.factory)

    # ------------------------------------------------------------------
    # Load or create id_map
    # ------------------------------------------------------------------
    def _load_or_create_id_maps(self):
        idmap_path = "faiss/id_maps.json"

        if os.path.exists(idmap_path):
            print("[IndexLoader] Loading id_maps from disk...")
            with open(idmap_path, "r") as f:
                self.id_maps = json.load(f)

            # Ensure FAISS vector_id keys are int
            for cat in self.id_maps:
                self.id_maps[cat] = {int(k): v for k, v in self.id_maps[cat].items()}

        else:
            print("[IndexLoader] No id_maps found → using empty maps")
            self.id_maps = {cat: {} for cat in self.index_paths}

    # ------------------------------------------------------------------
    # Load or create indexes
    # ------------------------------------------------------------------
    def _load_or_create_indexes(self):
        for category, path in self.index_paths.items():

            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)

            if os.path.exists(path):
                print(f"[IndexLoader] Loading FAISS index for {category}: {path}")
                idx = faiss.read_index(path)

                if idx.d != self.dim:
                    raise ValueError(
                        f"Dimension mismatch for {category}: index={idx.d}, expected={self.dim}"
                    )

                self.indexes[category] = idx

            else:
                print(f"[IndexLoader] No existing index for {category} → creating new index")
                self.indexes[category] = self._build_new_index()

    # ------------------------------------------------------------------
    # Save FAISS indexes + id_maps
    # ------------------------------------------------------------------
    def save_all(self):
        for category, idx in self.indexes.items():
            path = self.index_paths[category]
            faiss.write_index(idx, path)
            print(f"[IndexBuilder] Saved index → {path}")

        idmap_path = "faiss/id_maps.json"
        with open(idmap_path, "w") as f:
            json.dump(self.id_maps, f, indent=2)
        print(f"[IndexBuilder] Saved id_maps → {idmap_path}")

    # ------------------------------------------------------------------
    # Ingest embeddings from metadata.json format
    # ------------------------------------------------------------------
    def ingest_items_from_metadata(self, metadata_dict: Dict):
        """
        metadata_dict format:
        {
            "item_123": {
                "category": "top",
                "embedding_path": "/absolute/path/to/embedding.npy"
            }
        }
        """
        print("[Ingest] Starting metadata ingestion...")

        category_map = {
            "top": "tops",
            "tops": "tops",
            "bottom": "bottoms",
            "bottoms": "bottoms",
            "shoe": "shoes",
            "shoes": "shoes",
        }

        # Reset id_maps
        self.id_maps = {cat: {} for cat in self.index_paths}

        for item_id, meta in metadata_dict.items():

            # Determine category
            raw_cat = meta["category"].lower().strip()
            cat = category_map.get(raw_cat)

            if cat not in self.indexes:
                print(f"[Ingest] WARNING: skipping item '{item_id}' → invalid category '{raw_cat}'")
                continue

            # Resolve embedding path
            emb_path = meta["embedding_path"]

            # Allow relative paths (relative to project root)
            if not os.path.isabs(emb_path):
                emb_path = os.path.abspath(emb_path)

            if not os.path.exists(emb_path):
                print(f"[Ingest] WARNING: embedding path missing → {emb_path}")
                continue

            # Load embedding
            try:
                emb = np.load(emb_path).astype("float32").reshape(1, -1)
            except Exception as e:
                print(f"[Ingest] ERROR loading embedding for {item_id}: {e}")
                continue

            # Normalize for FAISS cosine sim
            faiss.normalize_L2(emb)

            idx = self.indexes[cat]
            vid = idx.ntotal
            idx.add(emb)

            # Save reverse mapping
            self.id_maps[cat][vid] = item_id

        print("[Ingest] Completed embedding ingestion.")
