"""
Index Builder & Loader Module
-----------------------------

This module is responsible for:
1. Building FAISS indexes for each category (tops/bottoms/shoes)
2. Persisting FAISS indexes to disk (Docker volume)
3. Loading existing indexes from disk
4. Rebuilding indexes when vector deletions accumulate
5. Validating dimension compatibility

This module does NOT insert vectors (EmbeddingManager does that).
"""

import os
import faiss
import json
import numpy as np
from typing import Dict, List


class IndexBuilderLoader:
    """
    Builds, loads, saves, and rebuilds FAISS indexes for each category.
    Also loads/saves id_maps so that index and metadata remain aligned.
    """

    def __init__(self, index_paths: Dict[str, str], dim: int, index_factory_string: str = "Flat"):
        """
        Args:
            index_paths: dict mapping category -> index path
                {
                    "tops": "/data/faiss/tops.index",
                    "bottoms": "/data/faiss/bottoms.index",
                    "shoes": "/data/faiss/shoes.index"
                }

            dim: embedding dimension (CLIP output dimension)
            index_factory_string: FAISS index type (e.g., "Flat", "IVF64,HNSW32")
        """

        self.index_paths = index_paths
        self.dim = dim
        self.factory = index_factory_string

        self.indexes = {}
        self.id_maps = {}  

        self._load_or_create_indexes()
        self._load_or_create_id_maps()

    # -------------------------------------------------------------
    # Build new FAISS index
    # -------------------------------------------------------------
    def _build_new_index(self):
        print(f"[IndexBuilder] Building new FAISS index: {self.factory}, dim={self.dim}")
        index = faiss.index_factory(self.dim, self.factory)
        return index

    # -------------------------------------------------------------
    # Load saved id_maps or create empty structure
    # -------------------------------------------------------------
    def _load_or_create_id_maps(self):
        idmap_path = "faiss/id_maps.json"

        if os.path.exists(idmap_path):
            print("[IndexLoader] Loading id_maps from disk")
            with open(idmap_path, "r") as f:
                self.id_maps = json.load(f)

            # Convert keys back to int (JSON stores keys as strings)
            for cat in self.id_maps:
                self.id_maps[cat] = {int(k): v for k, v in self.id_maps[cat].items()}

        else:
            print("[IndexLoader] No id_maps found → creating empty map")
            self.id_maps = {cat: {} for cat in self.index_paths}
    
    # -------------------------------------------------------------
    # Load existing FAISS indexes or create new ones
    # -------------------------------------------------------------
    def _load_or_create_indexes(self):
        for category, path in self.index_paths.items():

            if os.path.exists(path):
                print(f"[IndexLoader] Loading FAISS index for {category} from {path}")
                idx = faiss.read_index(path)

                # Dimension check
                if idx.d != self.dim:
                    raise ValueError(
                        f"[IndexLoader] Dimension mismatch for {category}: "
                        f"index_dim={idx.d}, expected={self.dim}"
                    )

                self.indexes[category] = idx

            else:
                print(f"[IndexLoader] No index found for {category} → creating new index")
                new_idx = self._build_new_index()
                self.indexes[category] = new_idx

    # -------------------------------------------------------------
    # Save indexes and id_maps
    # -------------------------------------------------------------
    def save_all(self):
        base_dir = os.path.dirname(list(self.index_paths.values())[0])
        os.makedirs(base_dir, exist_ok=True)

        for category, index in self.indexes.items():
            path = self.index_paths[category]
            faiss.write_index(index, path)
            print(f"[IndexLoader] Saved {category} index → {path}")

        # save id_maps
        idmap_path = "faiss/id_maps.json"
        with open(idmap_path, "w") as f:
            json.dump(self.id_maps, f, indent=2)
        print(f"[IndexLoader] Saved id_maps → {idmap_path}")

    # -------------------------------------------------------------
    # Ingest items (ONLY when building index)
    # -------------------------------------------------------------
    def ingest_items(self, items_json):
        """
        items_json: list of dicts
        [
            {"item_id": "abc123", "category": "tops", "embedding": [...]},
            ...
        ]
        """

        category_map = {
            "top": "tops",
            "tops": "tops",
            "bottom": "bottoms",
            "bottoms": "bottoms",
            "shoe": "shoes",
            "shoes": "shoes",
        }

        # regenerate id_maps
        self.id_maps = {cat: {} for cat in self.index_paths}

        for item in items_json:
            raw_cat = item["category"].lower().strip()
            cat = category_map.get(raw_cat)

            if cat not in self.indexes:
                print(f"[IngestItems] WARNING: skipping item_id={item['item_id']} "
                      f"because category '{raw_cat}' is not defined.")
                continue

            emb = np.array(item["embedding"], dtype="float32").reshape(1, -1)
            faiss.normalize_L2(emb)

            index = self.indexes[cat]
            vid = index.ntotal
            index.add(emb)

            self.id_maps[cat][vid] = item["item_id"]

        print("[IngestItems] Successfully ingested items.")
    
    # -------------------------------------------------------------
    # New ingestion method for new metadata.json structure
    # -------------------------------------------------------------
    def ingest_items_from_metadata(self, metadata_dict):
        """
        metadata_dict: dict[item_id] -> metadata entry
        """
        print("[IngestItems] Ingesting embeddings from metadata.json (.npy files)")

        category_map = {
            "top": "tops", "tops": "tops",
            "bottom": "bottoms", "bottoms": "bottoms",
            "shoe": "shoes", "shoes": "shoes",
        }

        # Reset id_maps
        self.id_maps = {cat: {} for cat in self.index_paths}

        for item_id, meta in metadata_dict.items():

            raw_cat = meta["category"].lower().strip()
            cat = category_map.get(raw_cat)

            if cat not in self.indexes:
                print(f"[IngestItems] WARNING: skipping {item_id}, category '{raw_cat}' not recognized.")
                continue

            # Fix embedding path relative to pipeline location
            emb_path = meta["embedding_path"]
            emb_path = os.path.join("../../../", emb_path)

            if not os.path.exists(emb_path):
                print(f"[IngestItems] WARNING: embedding file missing → {emb_path}")
                continue

            # Load embedding
            try:
                emb = np.load(emb_path).astype("float32").reshape(1, -1)
            except Exception as e:
                print(f"[IngestItems] ERROR loading embedding for {item_id}: {e}")
                continue

            # Normalize
            faiss.normalize_L2(emb)

            # Add to FAISS index
            idx = self.indexes[cat]
            vid = idx.ntotal
            idx.add(emb)

            self.id_maps[cat][vid] = item_id

        print("[IngestItems] Successfully ingested all vectors from metadata.json")
