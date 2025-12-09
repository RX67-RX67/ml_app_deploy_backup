import os
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class YOLOv8FeatureExtractor:
    """
    Use a forward hook on the YOLOv8 backbone to obtain a fixed-length embedding.
    The hook captures the backbone output (before the detection head) for rich feature vectors.
    """

    def __init__(self,
                 model_path: str = "models/trained/best.pt",
                 device: str = "cpu",
                 imgsz: int = 640):
        self.model_path = model_path
        self.device = device
        self.imgsz = imgsz
        self._feat: Optional[torch.Tensor] = None

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found at: {model_path}")

        print(f"Loading YOLO model (feature extractor): {model_path}...")

        try:
            self.yolo = YOLO(model_path)
            self.model = self.yolo.model.to(device)
            self.model.eval()
            print(f"✓ Feature extractor successfully loaded on {device}")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            raise

        # Find a suitable backbone Conv2d layer for feature extraction
        target_layer = None
        target_channels = 0

        # Scan the model to find a good backbone layer
        # (usually a C2f or a large Conv with >=256 channels)
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                # Prefer layers with large output channels (backbone features usually >=256)
                if module.out_channels >= 256 and module.out_channels > target_channels:
                    # Exclude small 1×1 convs used in detection head
                    if module.kernel_size[0] > 1 or module.out_channels >= 256:
                        target_layer = module
                        target_channels = module.out_channels

        if target_layer is None:
            # Fallback: pick any Conv2d layer with >=128 channels
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv2d) and module.out_channels >= 128:
                    target_layer = module
                    target_channels = module.out_channels
                    break

        if target_layer is None:
            raise RuntimeError("Could not find a suitable Conv2d layer for embeddings.")

        print(f"✓ Hook registered on feature layer, output dimension: {target_channels}")

        # Register forward hook
        def hook_forward(_, __, output):
            self._feat = output

        target_layer.register_forward_hook(hook_forward)
        self._embedding_dim = target_channels

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        return img

    def extract_embedding(self, image_path: str) -> np.ndarray:
        """
        Returns a 1D numpy embedding array of shape (C,).
        """
        img = self._load_image(image_path)

        tensor = torch.from_numpy(img).float().to(self.device)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0

        self._feat = None
        with torch.no_grad():
            _ = self.model(tensor)

        if self._feat is None:
            raise RuntimeError("Forward hook failed to capture feature map.")

        fmap = self._feat  # (1, C, H, W)
        fmap = fmap.mean(dim=(2, 3))  # global average pooling → (1, C)
        emb = fmap.squeeze(0).cpu().numpy().astype(np.float32)
        return emb


if __name__ == "__main__":

    extractor = YOLOv8FeatureExtractor(
        model_path="models/trained/best.pt",
        device="cpu"
    )
    print(f"Embedding dimension: {extractor.embedding_dim}")
