import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizer


class SimpleCLIPTextEncoder:
    """
    HuggingFace CLIP text encoder with lazy loading
    """

    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

     
        self.model = None
        self.tokenizer = None

    # -------------------------------------------------------
    # Lazy load 
    # -------------------------------------------------------
    def _load_model(self):
        if self.model is None:
            print("[SimpleCLIP] Loading tokenizer + CLIP model...")

            self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()

            print("[SimpleCLIP] CLIP model loaded.")

    # -------------------------------------------------------
    # Encoding function
    # -------------------------------------------------------
    def encode(self, text: str) -> np.ndarray:

        self._load_model()

        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)

        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)

        return feats[0].cpu().numpy().astype("float32")
