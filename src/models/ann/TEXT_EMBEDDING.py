# TEXT_EMBEDDING.py
import torch
import numpy as np
from transformers import CLIPTokenizer, CLIPModel

class CLIPTextEmbeddingExtractor:
    """
    Extract CLIP text embeddings using a local model directory.
    """

    def __init__(self,
                 model_path: str = "./local_clip",   # local model folder
                 device: str = None,
                 normalize: bool = True):

        device = "cpu"

        self.device = device
        self.normalize = normalize

        print(f"[CLIPTextEncoder] Loading CLIP model from {model_path} on {device} ...")

        # Load tokenizer & model from local directory
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path)
        self.model = CLIPModel.from_pretrained(model_path)
        self.model = self.model.to(device)
        self.model.eval()

        print("[CLIPTextEncoder] Loaded successfully.")

    def encode(self, text: str) -> np.ndarray:
        """
        Encode text → 512-d CLIP embedding
        """
        inputs = self.tokenizer([text], padding=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            emb = self.model.get_text_features(**inputs)

        emb = emb.cpu().numpy().astype("float32")[0]

        # normalize to unit vector
        if self.normalize:
            emb = emb / (np.linalg.norm(emb) + 1e-8)

        return emb
