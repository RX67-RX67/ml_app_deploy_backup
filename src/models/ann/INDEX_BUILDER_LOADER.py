import os
import faiss
import json
import numpy as np


class IndexBuilderLoader:
    """
    Builds/Loads FAISS indexes for tops / bottoms / shoes
    using metadata.json (embedding_path -> .npy file)
    """

    def __init__(self, index_paths: dict, dim: int, index_factory_string="Flat"):
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
        return faiss.index_factory(self.dim, self.factory)

    # -------------------------------------------------------------
    # Load id_maps.json
    # -------------------------------------------------------------
    def _load_or_create_id_maps(self):
        idmap_path = "src/models/ann/faiss/id_maps.json"

        if os.path.exists(idmap_path):
            print("[IndexLoader] Loading id_maps.json")
            with open(idmap_path, "r") as f:
                self.id_maps = json.load(f)

            # convert keys to int
            for cat in self.id_maps:
                self.id_maps[cat] = {int(k): v for k, v in self.id_maps[cat].items()}

        else:
            print("[IndexLoader] No id_maps found → creating empty structure")
            self.id_maps = {c: {} for c in self.index_paths}

    # -------------------------------------------------------------
    # Load existing FAISS indexes or create new ones
    # -------------------------------------------------------------
    def _load_or_create_indexes(self):
        for cat, path in self.index_paths.items():
            if os.path.exists(path):
                print(f"[IndexLoader] Loading FAISS index → {path}")
                idx = faiss.read_index(path)

                if idx.d != self.dim:
                    raise ValueError(f"Dim mismatch in {cat}: Index={idx.d}, Expected={self.dim}")

                self.indexes[cat] = idx
            else:
                print(f"[IndexLoader] No index found for {cat} → creating new index")
                self.indexes[cat] = self._build_new_index()

    # -------------------------------------------------------------
    # Save FAISS indexes and id_maps
    # -------------------------------------------------------------
    def save_all(self):
        for cat, idx in self.indexes.items():
            path = self.index_paths[cat]
            os.makedirs(os.path.dirname(path), exist_ok=True)
            faiss.write_index(idx, path)
            print(f"[IndexSaver] Saved {cat}.index → {path}")

        idmap_path = "src/models/ann/faiss/id_maps.json"
        with open(idmap_path, "w") as f:
            json.dump(self.id_maps, f, indent=2)
        print(f"[IndexSaver] Saved id_maps.json")

    # -------------------------------------------------------------
    # ⭐ Correct ingestion logic for new metadata.json
    # -------------------------------------------------------------
    def ingest_items_from_metadata(self, metadata_dict: dict):
        """
        metadata_dict = { item_id → { metadata } }
        """

        print("\n[IngestItems] START ingesting embeddings from metadata.json")

        category_map = {
            "top": "tops",
            "tops": "tops",
            "bottom": "bottoms",
            "bottoms": "bottoms",
            "shoes": "shoes",
            "shoe": "shoes",
        }

        # reset id_maps
        self.id_maps = {c: {} for c in self.index_paths}

        # iterate metadata entries
        for item_id, meta in metadata_dict.items():

            raw_cat = meta["category"].lower().strip()
            cat = category_map.get(raw_cat)

            if cat not in self.indexes:
                print(f"[IngestItems] SKIP {item_id}: Category '{raw_cat}' invalid.")
                continue

            emb_path = meta["embedding_path"]

            # ---- FIXED: correct embedding path inside docker ---- #
            emb_path = os.path.join("/app", emb_path)

            if not os.path.exists(emb_path):
                print(f"[IngestItems] WARNING: Embedding file missing → {emb_path}")
                continue

            # load embedding
            try:
                emb = np.load(emb_path).astype("float32").reshape(1, -1)
            except Exception as e:
                print(f"[IngestItems] ERROR loading embedding {emb_path}: {e}")
                continue

            # normalize
            faiss.normalize_L2(emb)

            idx = self.indexes[cat]
            vid = idx.ntotal
            idx.add(emb)

            self.id_maps[cat][vid] = item_id

        print("[IngestItems] DONE ingesting all embeddings.\n")
