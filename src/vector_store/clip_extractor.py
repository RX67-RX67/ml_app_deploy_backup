# src/embeddings/clip_extractor.py

from pathlib import Path
from typing import List, Union, Optional

import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor


class CLIPEmbeddingExtractor:
    """
    Wrapper around a CLIP image encoder.

    Default model: openai/clip-vit-base-patch32 → 512-dim image embeddings.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        """
        Args:
            model_name: Hugging Face model ID for CLIP.
            device: "cuda", "cpu", or None (auto-detect).
            normalize: L2-normalize embeddings to unit length.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = normalize

        # Load model + processor
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @staticmethod
    def _load_image(path: Union[str, Path]) -> Image.Image:
        """Load an image from disk and ensure RGB format."""
        img = Image.open(path).convert("RGB")
        return img

    def embed_single(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Compute embedding for a single image.

        Returns:
            np.ndarray with shape (512,) and dtype float32.
        """
        img = self._load_image(image_path)

        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)  # (1, 512)

        if self.normalize:
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)

        emb = feats[0].cpu().numpy().astype("float32")
        return emb

    def embed_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Compute embeddings for a list of image paths.

        Args:
            image_paths: List of paths.
            batch_size: Number of images per forward pass.

        Returns:
            np.ndarray with shape (N, 512) and dtype float32.
        """
        all_embeddings = []
        n = len(image_paths)

        for start in range(0, n, batch_size):
            batch_paths = image_paths[start : start + batch_size]
            images = [self._load_image(p) for p in batch_paths]

            inputs = self.processor(images=images, return_tensors="pt").to(self.device)

            with torch.no_grad():
                feats = self.model.get_image_features(**inputs)  # (B, 512)

            if self.normalize:
                feats = torch.nn.functional.normalize(feats, p=2, dim=-1)

            all_embeddings.append(feats.cpu().numpy().astype("float32"))

        if not all_embeddings:
            return np.empty((0, 512), dtype="float32")

        return np.concatenate(all_embeddings, axis=0)
