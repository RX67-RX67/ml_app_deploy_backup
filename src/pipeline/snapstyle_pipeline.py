# src/pipeline/snapstyle_pipeline.py

import os
import json
import uuid
from typing import Dict, List, Optional

import numpy as np

# ---- 业务模块 ----
from src.detectors.yolov8_detector import YoloV8Detector
from src.vector_store.clip_extractor import CLIPEmbeddingExtractor
from src.text_embedding.SIMPLECLIP import SimpleCLIPTextEncoder
from src.index_builder.INDEX_BUILDER_LOADER import IndexBuilderLoader
from src.ann_search.ANN_SEARCH_SERVICE import ANNSearchService
from src.outfit_generator.OUTFIT_GENERATOR import OutfitGenerator
from src.post_search_ranker.POST_SEARCH_RERANKING import OutfitReranker


class SnapStylePipeline:
    """
    SnapStyle end-to-end pipeline:

    Layer 1: Digitize closet (YOLO + CLIP image embeddings + metadata.json)
    Layer 2: Build / load FAISS indexes + ANN search
    Layer 3: Outfit generation (组合 top / bottom / shoes)
    Layer 4: Reranking with outfit coherence + text prompt embedding
    """

    # ---------------------------------------------------------
    # 初始化
    # ---------------------------------------------------------
    def __init__(
        self,
        device: str = "cpu",
        yolo_model_path: str = "models/trained/best.pt",
        metadata_path: str = "data/user_embeddings/metadata.json",
        crop_dir: str = "data/user_crops",
        emb_dir: str = "data/user_embeddings",
        index_dir: str = "faiss",
    ):
        # 工程根目录（…/ml_app_deploy_backup）
        self.PROJECT_ROOT = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )

        # 路径（都用相对路径字符串保存到 metadata 中）
        self.yolo_model_path = yolo_model_path
        self.metadata_rel = metadata_path
        self.crop_dir_rel = crop_dir
        self.emb_dir_rel = emb_dir
        self.index_dir_rel = index_dir

        # 运行时用的绝对路径
        self.metadata_path = self._resolve_path(self.metadata_rel)
        self.crop_dir = self._resolve_path(self.crop_dir_rel)
        self.emb_dir = self._resolve_path(self.emb_dir_rel)
        self.index_dir = self._resolve_path(self.index_dir_rel)

        os.makedirs(self.crop_dir, exist_ok=True)
        os.makedirs(self.emb_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)

        # 设备
        self.device = device

        # ---- Lazy 模型（真正的模型在各自类里 lazy-load）----
        self.detector = YoloV8Detector(
            model_path=self.yolo_model_path,
            device=self.device,
        )
        self.image_encoder = CLIPEmbeddingExtractor(
            device=self.device,
            normalize=True,
        )
        self.text_encoder = SimpleCLIPTextEncoder(
            device=self.device,
        )

        # ---- Metadata & Index ----
        self.metadata_dict: Dict[str, Dict] = {}
        self._load_metadata()

        self.indexes: Optional[Dict[str, "faiss.Index"]] = None
        self.id_maps: Optional[Dict[str, Dict[int, str]]] = None
        self.ann_service: Optional[ANNSearchService] = None

        # outfit 生成 & rerank（依赖 metadata）
        self.outfit_generator = OutfitGenerator(self.metadata_dict)
        self.reranker = OutfitReranker(self.metadata_dict)

    # =========================================================
    # 工具函数
    # =========================================================
    def _resolve_path(self, path: str) -> str:
        """把相对路径变成基于 PROJECT_ROOT 的绝对路径；绝对路径原样返回。"""
        if os.path.isabs(path):
            return path
        abs_path = os.path.join(self.PROJECT_ROOT, path)
        return abs_path

    def _rel_to_project(self, abs_path: str) -> str:
        """将绝对路径转成相对于 project_root 的相对路径，用于写入 metadata。"""
        return os.path.relpath(abs_path, self.PROJECT_ROOT)

    # ---------------------------------------------------------
    # Metadata 读写
    # ---------------------------------------------------------
    def _load_metadata(self):
        """从 metadata.json 读取到 self.metadata_dict（item_id -> meta）。"""
        if not os.path.exists(self.metadata_path):
            self.metadata_dict = {}
            return

        with open(self.metadata_path, "r") as f:
            data = json.load(f)

        # 磁盘上是 list，内存里转为 dict
        if isinstance(data, list):
            self.metadata_dict = {m["item_id"]: m for m in data}
        elif isinstance(data, dict):
            # 如果某次被写成 dict，也兼容一下
            self.metadata_dict = data
        else:
            raise ValueError("metadata.json format not supported")

    def _save_metadata(self):
        """把 self.metadata_dict 写回 metadata.json（以 list 形式）。"""
        data_list = list(self.metadata_dict.values())
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        with open(self.metadata_path, "w") as f:
            json.dump(data_list, f, indent=4)

    # =========================================================
    # Layer 1: Digitize Closet
    # =========================================================
    def _prepare_image_path(self, image_input) -> str:
        """
        支持两种输入：
        1. 字符串 path
        2. Streamlit UploadedFile / file-like 对象
        """
        # 字符串路径
        if isinstance(image_input, str):
            return self._resolve_path(image_input)

        # Streamlit UploadedFile: 有 .name 和 .read()
        if hasattr(image_input, "read"):
            os.makedirs(self._resolve_path("data/uploads"), exist_ok=True)
            filename = f"user_upload_{uuid.uuid4().hex}.png"
            save_path = self._resolve_path(os.path.join("data/uploads", filename))
            with open(save_path, "wb") as f:
                f.write(image_input.read())
            return save_path

        raise TypeError("image_input must be a filepath string or a file-like object")

    def digitize_image(self, image_input) -> List[Dict]:
        """
        Digitize 一张图片：
          1) YOLO 检测并裁剪
          2) CLIP 提取 embedding
          3) 写入 .npy + metadata.json

        返回：新增的 metadata 条目列表（每条是一个 dict）
        """
        image_path = self._prepare_image_path(image_input)

        # 1. YOLO detect + crop
        detections = self.detector.detect_and_crop(
            image_path=image_path,
            output_dir=self.crop_dir,
        )

        if not detections:
            print("[SnapStylePipeline] No clothing detected.")
            return []

        # 2. CLIP embedding + 保存 .npy + metadata
        new_items = []

        for det in detections:
            crop_abs = det["crop_path"]
            category = det["category"]
            item_id = uuid.uuid4().hex

            # 2.1 计算 embedding
            emb = self.image_encoder.embed_single(crop_abs)

            # 2.2 保存 embedding .npy
            emb_filename = f"{category}_{item_id}.npy"
            emb_abs = os.path.join(self.emb_dir, emb_filename)
            np.save(emb_abs, emb)

            # 相对路径写入 metadata
            emb_rel = self._rel_to_project(emb_abs)
            crop_rel = self._rel_to_project(crop_abs)

            meta_entry = {
                "item_id": item_id,
                "category": category,
                "crop_path": crop_rel,
                "embedding_path": emb_rel,
                "embedding_type": "CLIP",
                "embedding_dim": int(emb.shape[-1]),
                "bbox": det["bbox"],
                "confidence": det["conf"],
                "yolo_class": det["class_name"],
            }

            self.metadata_dict[item_id] = meta_entry
            new_items.append(meta_entry)

        # 3. 更新 metadata.json
        self._save_metadata()

        # 4. 由于新增了 item，FAISS index 需要重建：简单做法是清空缓存，让下一次搜索时重建
        self.indexes = None
        self.id_maps = None
        self.ann_service = None

        # 同时更新 outfit_generator & reranker 内部的 metadata 引用
        self.outfit_generator.metadata_dict = self.metadata_dict
        self.outfit_generator._load_embeddings()  # 重新载入一次 embedding
        self.reranker.meta = self.metadata_dict

        print(f"[SnapStylePipeline] Digitized {len(new_items)} items.")
        return new_items

    # =========================================================
    # Layer 2: ANN Index & Search
    # =========================================================
    def _ensure_indexes(self):
        """保证 FAISS indexes 和 ANNSearchService 已经就绪。"""
        if self.ann_service is not None and self.indexes is not None:
            return

        if not self.metadata_dict:
            raise RuntimeError("No metadata found, please digitize some images first.")

        # 1. 推断 embedding 维度（读任意一个 .npy）
        any_item = next(iter(self.metadata_dict.values()))
        emb_rel = any_item["embedding_path"]
        emb_abs = self._resolve_path(emb_rel)
        emb = np.load(emb_abs).astype("float32")
        dim = int(emb.shape[-1])

        # 2. 设置 index 路径（相对路径交给 IndexBuilderLoader 处理）
        index_paths = {
            "tops": os.path.join(self.index_dir_rel, "tops.index"),
            "bottoms": os.path.join(self.index_dir_rel, "bottoms.index"),
            "shoes": os.path.join(self.index_dir_rel, "shoes.index"),
        }

        # 3. 构建/加载 index
        builder = IndexBuilderLoader(
            index_paths=index_paths,
            dim=dim,
            index_factory_string="Flat",
        )

        # 用 metadata 中的 .npy 文件填充 index
        builder.ingest_items_from_metadata(self.metadata_dict)
        builder.save_all()

        self.indexes = builder.indexes
        self.id_maps = builder.id_maps
        self.ann_service = ANNSearchService(
            indexes=self.indexes,
            id_maps=self.id_maps,
        )

    def _load_item_embedding(self, item_id: str) -> np.ndarray:
        """从 metadata 中加载某个 item 的 embedding."""
        meta = self.metadata_dict[item_id]
        emb_rel = meta["embedding_path"]
        emb_abs = self._resolve_path(emb_rel)
        emb = np.load(emb_abs).astype("float32")
        return emb

    def _decide_search_categories(self, anchor_cat: str) -> List[str]:
        """
        根据 anchor category 决定 ANN 搜索的目标类别集合。
        """
        anchor_cat = anchor_cat.lower().strip()
        if anchor_cat in ["top", "tops"]:
            return ["bottoms", "shoes"]
        elif anchor_cat in ["bottom", "bottoms"]:
            return ["tops", "shoes"]
        elif anchor_cat in ["shoe", "shoes"]:
            return ["tops", "bottoms"]
        else:
            # 默认：top + bottom + shoes 全搜
            return ["tops", "bottoms", "shoes"]

    def ann_search_for_anchor(
        self,
        anchor_id: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> Dict:
        """
        对某个 anchor item 做 ANN 搜索，返回 ann_results 结构，
        供 OutfitGenerator 使用。
        """
        if anchor_id not in self.metadata_dict:
            raise ValueError(f"anchor_id={anchor_id} not found in metadata")

        self._ensure_indexes()

        anchor_meta = self.metadata_dict[anchor_id]
        anchor_cat = anchor_meta["category"]
        anchor_emb = self._load_item_embedding(anchor_id)

        cats = self._decide_search_categories(anchor_cat)

        ann_results = self.ann_service.search_multiple_categories(
            anchor_embedding=anchor_emb,
            categories=cats,
            k_per_category=top_k,
            similarity_threshold=similarity_threshold,
        )

        return ann_results

    # =========================================================
    # Layer 3: Outfit Generation
    # =========================================================
    def generate_outfits_for_anchor(
        self,
        anchor_id: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> List[Dict]:
        """
        完整执行：ANN 搜索 → Outfit 组合（不含 rerank）。
        """
        ann_results = self.ann_search_for_anchor(
            anchor_id=anchor_id,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )
        outfits = self.outfit_generator.generate(
            anchor_id=anchor_id,
            ann_results=ann_results,
        )
        return outfits

    # =========================================================
    # Layer 4: Rerank with Text Prompt
    # =========================================================
    def rerank_outfits(
        self,
        outfits: List[Dict],
        prompt_text: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> List:
        """
        对生成的 outfits 做重排序。
        返回 [(score, outfit), ...]，按 score 降序。
        """
        if not outfits:
            return []

        if prompt_text:
            prompt_emb = self.text_encoder.encode(prompt_text)
        else:
            prompt_emb = None

        ranked = self.reranker.rerank(outfits, prompt_emb=prompt_emb)

        if top_n is not None:
            ranked = ranked[:top_n]

        return ranked

    # =========================================================
    # 高层封装：一键推荐
    # =========================================================
    def recommend_outfits(
        self,
        anchor_id: str,
        prompt_text: Optional[str] = None,
        ann_top_k: int = 5,
        similarity_threshold: float = 0.0,
        max_outfits: int = 20,
    ) -> List:
        """
        一次性完成：
          1) ANN 搜索
          2) Outfit 生成
          3) 基于 prompt 的 rerank

        返回 [(score, outfit_dict), ...] 按 score 排序。
        """
        outfits = self.generate_outfits_for_anchor(
            anchor_id=anchor_id,
            top_k=ann_top_k,
            similarity_threshold=similarity_threshold,
        )

        ranked = self.rerank_outfits(
            outfits=outfits,
            prompt_text=prompt_text,
            top_n=max_outfits,
        )
        return ranked

    # =========================================================
    # 便捷工具：列出已有衣物
    # =========================================================
    def list_items(self, category: Optional[str] = None) -> List[Dict]:
        """
        用于前端展示：返回当前 metadata 中的所有 items。
        category 可选过滤，例如 "top" / "bottom" / "shoe"。
        """
        items = list(self.metadata_dict.values())
        if category:
            cat = category.lower().strip()
            items = [
                m
                for m in items
                if m["category"].lower().strip().startswith(cat)
            ]
        return items
