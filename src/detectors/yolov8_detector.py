import os
import uuid
from typing import List, Dict

import cv2
from ultralytics import YOLO


class YoloV8Detector:
    """
    YOLOv8 detector with L A Z Y  loading.
    Model loads ONLY when first used, avoiding Streamlit double-execution freeze.
    """

    def __init__(
        self,
        model_path: str = "models/trained/best.pt",
        device: str = "cpu",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
    ):

        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

        # DO NOT LOAD MODEL HERE ❌ (Lazy loading)
        self.yolo = None
        self.class_names = {}

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found at: {model_path}")

        print("[YoloV8Detector] Initialized (model not yet loaded).")

    # ----------------------------------------------------
    # L A Z Y   L O A D I N G  
    # ----------------------------------------------------
    def _load_model(self):
        """Load YOLO model only when first used."""
        if self.yolo is None:
            print(f"[YoloV8Detector] Loading YOLO model: {self.model_path} ...")

            try:
                self.yolo = YOLO(self.model_path)
                print(f"[YoloV8Detector] ✓ YOLOv8 loaded on device={self.device}")
            except Exception as e:
                print(f"[YoloV8Detector] ✗ Failed to load YOLO model: {e}")
                raise

            # Move to device
            try:
                self.yolo.to(self.device)
            except AttributeError:  # older ultralytics fallback
                self.yolo.model.to(self.device)

            # Class names
            try:
                self.class_names = self.yolo.model.names
            except AttributeError:
                self.class_names = {}

    # ----------------------------------------------------
    # YOLO detection only
    # ----------------------------------------------------
    def detect(self, image_path: str) -> List[Dict]:

        # <-- ensure model is loaded
        self._load_model()  

        results = self.yolo.predict(
            image_path,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        if not results:
            return []

        r = results[0]
        dets = []
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item()) if box.cls is not None else -1
            conf = float(box.conf[0].item()) if box.conf is not None else 0.0
            cls_name = self.class_names.get(cls_id, str(cls_id))

            dets.append({
                "bbox": [x1, y1, x2, y2],
                "cls_id": cls_id,
                "cls_name": cls_name,
                "conf": conf,
            })

        return dets

    # ----------------------------------------------------
    # Clothing refinement
    # ----------------------------------------------------
    def refine_clothing_category(self, bbox, img_h):
        x1, y1, x2, y2 = bbox
        yc = (y1 + y2) / 2

        return "top" if yc < img_h * 0.55 else "bottom"

    # ----------------------------------------------------
    # Detect + Crop + Assign category
    # ----------------------------------------------------
    def detect_and_crop(
        self,
        image_path: str,
        output_dir: str = "data/user_crops",
    ) -> List[Dict]:

        # <-- ensure model is loaded
        self._load_model()

        os.makedirs(output_dir, exist_ok=True)

        img = cv2.imread(image_path)
        if img is None:
            return []

        dets = self.detect(image_path)
        out = []

        h, w = img.shape[:2]

        for d in dets:
            x1, y1, x2, y2 = d["bbox"]

            # Clamp coordinates
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            cls_name = d["cls_name"].lower()

            # ------------------------------
            # 1. YOLO → category mapping
            # ------------------------------
            if any(x in cls_name for x in ["shoe", "boot", "sandal"]):
                category = "shoes"

            elif any(x in cls_name for x in ["bag", "handbag", "backpack"]):
                category = "bag"

            elif "accessory" in cls_name or cls_name in ["belt", "hat", "scarf"]:
                category = "accessory"

            elif any(x in cls_name for x in ["cloth", "clothing", "apparel"]):
                category = self.refine_clothing_category(d["bbox"], h)

            else:
                category = "unknown"

            # ------------------------------
            # save crop
            # ------------------------------
            fname = f"{category}_{uuid.uuid4().hex}.jpg"
            save_path = os.path.join(output_dir, fname)
            cv2.imwrite(save_path, crop)

            out.append({
                "crop_path": save_path,
                "bbox": [x1, y1, x2, y2],
                "class_name": d["cls_name"],
                "conf": d["conf"],
                "category": category,
                "img_h": h,
            })

        return out
