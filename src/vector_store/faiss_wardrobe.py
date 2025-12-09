"""
FAISS Wardrobe Vector Store
Manages FAISS indices for clothing item embeddings and similarity search.

Integration Points:
- Input: Embeddings from CLIP/Meta model (512-dim or 768-dim vectors)
- Output: Similar items for outfit generation (Layer 2)
"""

import os
import json
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class FAISSWardrobe:
    """
    FAISS-based vector store for wardrobe items.

    Partitions items by category type (tops, bottoms, shoes, etc.) for efficient
    similarity search during outfit generation.

    Example Usage:
        # Build indices (one-time setup)
        wardrobe = FAISSWardrobe()
        wardrobe.build_indices_from_embeddings(
            embeddings_file='data/processed/embeddings.npy',
            metadata_file='data/processed/metadata.csv'
        )

        # Search for compatible items
        shirt_embedding = get_embedding('my_shirt.jpg')
        compatible_bottoms = wardrobe.search_similar(
            query_embedding=shirt_embedding,
            category='bottoms',
            k=20,
            filters={'formality': 'formal'}
        )
    """

    def __init__(self, index_dir='models/faiss'):
        """
        Initialize FAISS wardrobe.

        Args:
            index_dir: Directory where FAISS indices are stored
        """
        self.index_dir = Path(index_dir)
        self.indices = {}
        self.id_mappings = {}
        self.metadata = None
        self.dimension = None

        # Load existing indices if available
        if self.index_dir.exists():
            self._load_indices()

    def build_indices_from_embeddings(
        self,
        embeddings_file: str,
        metadata_file: str,
        dimension: int = 512,
        index_type: str = "Flat"
    ):
        """
        Build FAISS indices from precomputed embeddings.

        Args:
            embeddings_file: Path to .npy file with shape (N, dimension)
            metadata_file: Path to metadata CSV with category info
            dimension: Embedding dimension (512 for CLIP, 768 for some models)
            index_type: FAISS index type ('Flat', 'IVF', 'HNSW')

        TODO: This method needs embeddings from teammate's CLIP extractor!
        Embeddings should be saved as: np.save('embeddings.npy', embeddings_array)
        """
        print(f"Building FAISS indices from {embeddings_file}...")

        # Load embeddings and metadata
        embeddings = np.load(embeddings_file).astype('float32')
        self.metadata = pd.read_csv(metadata_file)
        self.dimension = dimension

        assert embeddings.shape[0] == len(self.metadata), \
            f"Embeddings ({embeddings.shape[0]}) and metadata ({len(self.metadata)}) count mismatch!"

        assert embeddings.shape[1] == dimension, \
            f"Expected {dimension}-dim embeddings, got {embeddings.shape[1]}-dim"

        # Partition by category type
        categories = self.metadata['category_type'].unique()

        for category in categories:
            print(f"Building index for category: {category}")

            # Get embeddings for this category
            mask = (self.metadata['category_type'] == category).values
            category_embeddings = embeddings[mask]
            category_ids = self.metadata[mask]['item_id'].tolist()

            # Create FAISS index
            if index_type == "Flat":
                index = faiss.IndexFlatL2(dimension)
            elif index_type == "IVF":
                # For larger datasets, use IVF index
                nlist = min(100, len(category_embeddings) // 10)
                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                index.train(category_embeddings)
            elif index_type == "HNSW":
                # HNSW for faster search
                index = faiss.IndexHNSWFlat(dimension, 32)
            else:
                raise ValueError(f"Unknown index type: {index_type}")

            # Add embeddings to index
            index.add(category_embeddings)

            # Store index and ID mapping
            self.indices[category] = index
            self.id_mappings[category] = category_ids

            print(f"Added {len(category_ids)} items to {category} index")

        # Save indices
        self.save_indices()

        print(f"\nBuilt {len(self.indices)} FAISS indices")
        print(f"Saved to {self.index_dir}")

    def search_similar(
        self,
        query_embedding: np.ndarray,
        category: str,
        k: int = 10,
        filters: Optional[Dict[str, str]] = None
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search for k most similar items in a category.

        Args:
            query_embedding: Query vector (shape: (dimension,))
            category: Category to search ('tops', 'bottoms', 'shoes', etc.)
            k: Number of results to return
            filters: Optional filters like {'formality': 'formal', 'color': 'blue'}

        Returns:
            List of (item_id, similarity_score, metadata_dict) tuples

        Example:
            shirt_emb = get_embedding('my_shirt.jpg')
            bottoms = wardrobe.search_similar(
                query_embedding=shirt_emb,
                category='bottoms',
                k=20,
                filters={'formality': 'formal'}
            )
        """
        if category not in self.indices:
            raise ValueError(
                f"Category '{category}' not found. Available: {list(self.indices.keys())}")

        # Reshape query
        query = query_embedding.reshape(1, -1).astype('float32')

        # Get extra results for filtering
        search_k = k * 3 if filters else k

        # Search FAISS index
        distances, indices = self.indices[category].search(query, search_k)

        # Map indices to item IDs
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue

            item_id = self.id_mappings[category][idx]

            # Get metadata
            item_meta = self.metadata[self.metadata['item_id']
                                      == item_id].iloc[0].to_dict()

            # Apply filters
            if filters:
                if not self._matches_filters(item_meta, filters):
                    continue

            # Convert L2 distance to similarity score (higher is better)
            similarity = 1 / (1 + distance)

            results.append((item_id, float(similarity), item_meta))

            if len(results) >= k:
                break

        return results

    def add_item(
        self,
        item_id: str,
        embedding: np.ndarray,
        category: str,
        metadata_dict: Dict
    ):
        """
        Add a new item to the index (e.g., user uploads a new clothing item).

        Args:
            item_id: Unique item identifier
            embedding: Item embedding vector
            category: Category type (tops/bottoms/shoes/etc.)
            metadata_dict: Item metadata (color, formality, etc.)

        Example:
            new_embedding = extract_embedding('user_shirt.jpg')
            wardrobe.add_item(
                item_id='user_item_001',
                embedding=new_embedding,
                category='tops',
                metadata_dict={'color': 'blue', 'formality': 'casual', ...}
            )
        """
        if category not in self.indices:
            print(f"Creating new index for category: {category}")
            self.indices[category] = faiss.IndexFlatL2(self.dimension)
            self.id_mappings[category] = []

        # Add to FAISS
        embedding_reshaped = embedding.reshape(1, -1).astype('float32')
        self.indices[category].add(embedding_reshaped)

        # Update ID mapping
        self.id_mappings[category].append(item_id)

        # Update metadata
        new_row = pd.DataFrame([metadata_dict])
        self.metadata = pd.concat([self.metadata, new_row], ignore_index=True)

        print(f"Added item {item_id} to {category} index")

    def save_indices(self):
        """Save all FAISS indices and mappings to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)

        for category, index in self.indices.items():
            index_path = self.index_dir / f"{category}.index"
            faiss.write_index(index, str(index_path))

        # Save ID mappings
        mappings_path = self.index_dir / "id_mappings.json"
        with open(mappings_path, 'w') as f:
            json.dump(self.id_mappings, f)

        # Save metadata
        if self.metadata is not None:
            metadata_path = self.index_dir / "metadata.csv"
            self.metadata.to_csv(metadata_path, index=False)

        print(f"Saved indices to {self.index_dir}")

    def _load_indices(self):
        """Load existing FAISS indices from disk."""
        # Load indices
        for index_file in self.index_dir.glob("*.index"):
            category = index_file.stem
            self.indices[category] = faiss.read_index(str(index_file))

        # Load ID mappings
        mappings_path = self.index_dir / "id_mappings.json"
        if mappings_path.exists():
            with open(mappings_path, 'r') as f:
                self.id_mappings = json.load(f)

        # Load metadata
        metadata_path = self.index_dir / "metadata.csv"
        if metadata_path.exists():
            self.metadata = pd.read_csv(metadata_path)

        if self.indices:
            # Infer dimension from first index
            first_index = next(iter(self.indices.values()))
            self.dimension = first_index.d

            print(f"Loaded {len(self.indices)} indices from {self.index_dir}")

    def _matches_filters(self, item_meta: Dict, filters: Dict[str, str]) -> bool:
        """Check if item matches all filters."""
        for key, value in filters.items():
            if item_meta.get(key) != value:
                return False
        return True

    def get_index_stats(self) -> Dict:
        """Get statistics about the indices."""
        stats = {}
        for category, index in self.indices.items():
            stats[category] = {
                'num_items': index.ntotal,
                'dimension': index.d
            }
        return stats


# ============================================================================
# INTEGRATION GUIDE FOR TEAMMATE
# ============================================================================
"""
To integrate CLIP embeddings:

1. Your teammate should create: src/embeddings/clip_extractor.py

Example structure:

    from transformers import CLIPModel, CLIPProcessor
    import torch
    from PIL import Image
    import numpy as np

    class CLIPEmbeddingExtractor:
        def __init__(self, model_name="openai/clip-vit-base-patch32"):
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)

        def extract(self, image_path):
            image = Image.open(image_path)
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                embeddings = self.model.get_image_features(**inputs)
            return embeddings[0].numpy()  # Shape: (512,)

2. Process all images and save embeddings:

    extractor = CLIPEmbeddingExtractor()
    metadata = pd.read_csv('data/processed/metadata.csv')

    embeddings = []
    for img_path in metadata['image_path']:
        emb = extractor.extract(img_path)
        embeddings.append(emb)

    # Save
    np.save('data/processed/embeddings.npy', np.array(embeddings))

3. Then you (FAISS owner) can run:

    wardrobe = FAISSWardrobe()
    wardrobe.build_indices_from_embeddings(
        embeddings_file='data/processed/embeddings.npy',
        metadata_file='data/processed/metadata.csv'
    )
"""
