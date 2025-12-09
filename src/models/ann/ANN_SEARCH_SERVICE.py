"""
ANN Search Service Module
-------------------------

This module is responsible for:
1. Performing ANN search over category-specific FAISS indexes
   (e.g. bottoms.index, shoes.index)
2. Translating FAISS vector IDs back to business-level item IDs
3. Optionally attaching metadata from a database client
4. Providing a clean, JSON-serializable result structure for the
   upper-layer outfit combination and reranking modules.

This module does NOT:
- run YOLO or CLIP
- insert new embeddings into indexes (EmbeddingManager does that)
- build or load FAISS indexes from disk (IndexBuilderLoader does that)
"""

from typing import Dict, List, Optional, Iterable
import numpy as np
import faiss


class ANNSearchService:
    """
    ANN Search Service for retrieving compatible clothing items
    given an anchor CLIP embedding.
    """

    def __init__(
        self,
        indexes: Dict[str, faiss.Index],
        id_maps: Dict[str, Dict[int, str]],
        db_client=None,
        normalize: bool = True,
    ):
        """
        Args:
            indexes:
                A dict mapping category -> FAISS index.
                Example:
                    {
                        "tops": faiss.IndexFlatL2(...),
                        "bottoms": faiss.IndexFlatL2(...),
                        "shoes": faiss.IndexFlatL2(...)
                    }

            id_maps:
                A dict mapping category -> {vector_id: item_id}.
                Example:
                    {
                        "tops":    {0: "top_001", 1: "top_002"},
                        "bottoms": {0: "bottom_010"},
                        "shoes":   {0: "shoe_003"}
                    }

            db_client (optional):
                External DB client with method:
                    - get_item_metadata(item_id) -> dict
                If provided, metadata will be attached in search results.

            normalize:
                Whether to L2-normalize the anchor embedding before search.
                For CLIP embeddings (cosine-like similarity), normalization
                is usually recommended.
        """

        self.indexes = indexes
        self.id_maps = id_maps
        self.db = db_client
        self.normalize = normalize

        # Basic consistency check (optional)
        self._check_dimensions_consistency()

    # ------------------------------------------------------------------
    # Internal helper: sanity check that all indexes share the same dim
    # ------------------------------------------------------------------
    def _check_dimensions_consistency(self):
        dims = set()
        for cat, idx in self.indexes.items():
            dims.add(idx.d)
        if len(dims) > 1:
            raise ValueError(
                f"[ANNSearchService] Inconsistent index dimensions detected: {dims}"
            )

    # ------------------------------------------------------------------
    # Public API: search a single category
    # ------------------------------------------------------------------
    def search_category(self, anchor_embedding: np.ndarray, category: str, k: int = 10, similarity_threshold: float = 0.0) -> Dict:
        """
        Perform ANN search in a single category.
        """

        if category not in self.indexes:
            raise ValueError(f"[ANNSearchService] Invalid category: {category}")

        index = self.indexes[category]

        # Prepare anchor vector
        vec = np.array(anchor_embedding, dtype="float32")
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)

        if self.normalize:
            faiss.normalize_L2(vec)

        # ANN search
        distances, vector_ids = index.search(vec, k)

        results = []

        for dist, vid in zip(distances[0], vector_ids[0]):
            vid = int(vid)
            item_id = self.id_maps[category].get(vid)
            if item_id is None:
                continue
            
            # change distance to similarity for better interpretability
            similarity = 1.0 / (1.0 + float(dist))

            # apply similarity threshold
            if similarity < similarity_threshold:
                continue

            item_data = {
                "item_id": item_id,
                "distance": float(dist),
                "similarity": similarity,   
            }

            results.append(item_data)

        return {
            "category": category,
            "results": results,
        }


    # ------------------------------------------------------------------
    # Public API: search multiple categories (e.g. bottoms + shoes)
    # ------------------------------------------------------------------
    def search_multiple_categories(
        self,
        anchor_embedding: np.ndarray,
        categories: Iterable[str],
        k_per_category: int = 10,
        similarity_threshold: float = 0.0,
    ) -> Dict[str, Dict]:
        """
        Perform ANN search for several categories at once
        (e.g. bottoms and shoes for one anchor top).

        Args:
            anchor_embedding: CLIP embedding of the anchor item.
            categories: iterable of categories to search in.
            k_per_category: top-k items to return per category.

        Returns:
            {
                "bottoms": {
                    "category": "bottoms",
                    "results": [...]
                },
                "shoes": {  
                    "category": "shoes",
                    "results": [...]
                }
            }
        """
        out = {}
        for cat in categories:
            out[cat] = self.search_category(
                anchor_embedding=anchor_embedding,
                category=cat,
                k=k_per_category,
                similarity_threshold=similarity_threshold,  
            )
        return out

   