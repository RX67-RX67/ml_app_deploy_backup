import clip
import torch
import numpy as np

class SimpleCLIPTextEncoder:
    def __init__(self, device="cpu"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)

    def encode(self, text: str) -> np.ndarray:
        text_tokens = clip.tokenize([text]).to(self.device)
        
        with torch.no_grad():
            emb = self.model.encode_text(text_tokens)

        emb = emb.cpu().numpy().astype("float32")[0]
        emb /= (np.linalg.norm(emb) + 1e-8)
        return emb
