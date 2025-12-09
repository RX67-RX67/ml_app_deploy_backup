# src/pipeline/snapstyle_pipeline.py

import os
import json
import uuid
from typing import Dict, List, Optional

import numpy as np

from src.detectors.yolov8_detector import YoloV8Detector
from src.vector_store.clip_extractor import CLIPEmbeddingExtractor
from src.text_embedding.SIMPLECLIP import SimpleCLIPTextEncoder
from src.index_builder.INDEX_BUILDER_LOADER import IndexBuilderLoader
from src.ann_search.ANN_SEARCH_SERVICE import ANNSearchService
from src.outfit_generator.OUTFIT_GENERATOR import OutfitGenerator
from src.post_search_ranker.POST_SEARCH_RERANKING import OutfitReranker


class SnapStylePipeline:
    """
    HuggingFace-friendly pipeline:
    All writable paths moved to /data and /tmp.
    """

    def __init__(
        self,
        device: str = "cpu",

        # HF 可写路径（persistent）
        data_root: str = "/data/snapstyle",

        # YOLO 模型
        yolo_model_path: str = "/data/best.pt",
    ):

        self.device = device

        # --------------------------
        # 目录结构（HF 兼容）
        # --------------------------
        self.DATA_ROOT = data_root
        self.CROP_DIR = f"{self.DATA_ROOT}/crops"
        self.EMB_DIR = f"{self.DATA_ROOT}/embeddings"
        self.FAISS_DIR = f"{self.DATA_ROOT}/faiss"
        self.METADATA_PATH = f"{self.DATA_ROOT}/metadata.json"

        self.TMP_UPLOAD = "/tmp/snapstyle/uploads"

        # 创建所有 HF 允许写入的目录
        for d in [
            self.DATA_ROOT,
            self.CROP_DIR,
            self.EMB_DIR,
            self.FAISS_DIR,
            self.TMP_UPLOAD,
        ]:
            os.makedirs(d, exist_ok=True)

        # YOLO 模型路径
        if not os.path.exists(yolo_model_path):
            raise FileNotFoundError(
                f"YOLO model not found at {yolo_model_path}\n"
                "➡️ 请到 HuggingFace Space 的 Files 页上传 best.pt 到 /data/"
            )

        self.yolo_model_path = yolo_model_path

        # models
        self.detector = YoloV8Detector(
            model_path=self.yolo_model_path, device=device
        )
        self.image_encoder = CLIPEmbeddingExtractor(device=device)
        self.text_encoder = SimpleCLIPTextEncoder(device=device)

        # metadata
        self.metadata_dict: Dict[str, Dict] = {}
        self._load_metadata()

        # ann
        self.indexes = None
        self.id_maps = None
        self.ann_service = None

        # outfit tools
        self.outfit_generator = OutfitGenerator(self.metadata_dict)
        self.reranker = OutfitReranker(self.metadata_dict)

    # =========================================================
    # Metadata
    # =========================================================
    def _load_metadata(self):
        if not os.path.exists(self.METADATA_PATH):
            self.metadata_dict = {}
            return

        with open(self.METADATA_PATH, "r") as f:
            data = json.load(f)

        self.metadata_dict = {
            m["item_id"]: m for m in data
        } if isinstance(data, list) else data

    def _save_metadata(self):
        os.makedirs(self.DATA_ROOT, exist_ok=True)
        with open(self.METADATA_PATH, "w") as f:
            json.dump(list(self.metadata_dict.values()), f, indent=4)

    # =========================================================
    # Save uploaded input image
    # =========================================================
    def _save_uploaded_image(self, uploaded_file) -> str:
        """Save uploaded image into /tmp/"""
        filename = f"{uuid.uuid4().hex}_{uploaded_file.name}"
        save_path = os.path.join(self.TMP_UPLOAD, filename)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())

        return save_path

    # =========================================================
    # Digitization
    # =========================================================
    def digitize_image(self, image_input) -> List[Dict]:
        """YOLO + CLIP embedding"""

        # get real path
        if hasattr(image_input, "read"):
            image_path = self._save_uploaded_image(image_input)
        else:
            image_path = image_input

        # 1. yolo detect
        detections = self.detector.detect_and_crop(
            image_path=image_path,
            output_dir=self.CROP_DIR,
        )

        if not detections:
            return []

        new_items = []

        for det in detections:
            crop_path = det["crop_path"]
            category = det["category"]
            item_id = uuid.uuid4().hex

            # 2. embedding
            emb = np.array(
                self.image_encoder.embed_single(crop_path),
                dtype="float32"
            )
            emb_file = f"{category}_{item_id}.npy"
            emb_path = os.path.join(self.EMB_DIR, emb_file)
            np.save(emb_path, emb)

            # record metadata
            meta = {
                "item_id": item_id,
                "category": category,
                "crop_path": crop_path,           # absolute path OK
                "embedding_path": emb_path,       # absolute path OK
                "embedding_dim": int(emb.shape[-1]),
                "bbox": det["bbox"],
                "confidence": det["conf"],
                "yolo_class": det["class_name"],
            }

            self.metadata_dict[item_id] = meta
            new_items.append(meta)

        self._save_metadata()

        # Reset FAISS
        self.indexes = None
        self.id_maps = None
        self.ann_service = None

        self.outfit_generator.metadata_dict = self.metadata_dict
        self.outfit_generator._load_embeddings()
        self.reranker.meta = self.metadata_dict

        return new_items

    # =========================================================
    # ANN Index
    # =========================================================
    def _ensure_indexes(self):
        if self.ann_service is not None:
            return

        if not self.metadata_dict:
            raise RuntimeError("No items digitized yet")

        # deduce dim
        first_item = next(iter(self.metadata_dict.values()))
        emb = np.load(first_item["embedding_path"]).astype("float32")
        dim = emb.shape[-1]

        index_paths = {
            "tops": f"{self.FAISS_DIR}/tops.index",
            "bottoms": f"{self.FAISS_DIR}/bottoms.index",
            "shoes": f"{self.FAISS_DIR}/shoes.index",
        }

        builder = IndexBuilderLoader(
            index_paths=index_paths,
            dim=dim,
            index_factory_string="Flat",
        )
        builder.ingest_items_from_metadata(self.metadata_dict)
        builder.save_all()

        self.indexes = builder.indexes
        self.id_maps = builder.id_maps
        self.ann_service = ANNSearchService(
            indexes=self.indexes,
            id_maps=self.id_maps,
        )

    # =========================================================
    # Load embedding
    # =========================================================
    def _load_item_embedding(self, item_id):
        path = self.metadata_dict[item_id]["embedding_path"]
        return np.load(path).astype("float32")

    # =========================================================
    # For Outfit Planning
    # =========================================================
    def recommend_outfits(self, anchor_id, prompt_text=None, ann_top_k=5):
        self._ensure_indexes()

        anchor_emb = self._load_item_embedding(anchor_id)
        anchor_cat = self.metadata_dict[anchor_id]["category"]

        if anchor_cat == "top":
            cats = ["bottoms", "shoes"]
        elif anchor_cat == "bottom":
            cats = ["tops", "shoes"]
        else:
            cats = ["tops", "bottoms"]

        ann_results = self.ann_service.search_multiple_categories(
            anchor_embedding=anchor_emb,
            categories=cats,
            k_per_category=ann_top_k,
        )

        outfits = self.outfit_generator.generate(anchor_id, ann_results)
        ranked = self.reranker.rerank(
            outfits,
            prompt_emb=self.text_encoder.encode(prompt_text)
            if prompt_text else None,
        )
        return ranked
