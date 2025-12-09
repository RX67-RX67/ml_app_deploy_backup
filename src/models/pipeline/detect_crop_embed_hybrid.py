"""
Hybrid YOLO + CLIP Pipeline
Combines YOLO detection/classification with CLIP semantic embeddings.

Pipeline:
1. YOLO detects and crops clothing items
2. YOLO classifies items (top/bottom/shoes/etc)
3. CLIP extracts semantic embeddings from cropped items
4. Save embeddings for FAISS similarity search

Author: Integration of Xun's YOLO + Lance's CLIP
"""

import os
import argparse
import uuid
import numpy as np
import json

from src.models.detectors.yolov8_detector import YoloV8Detector
from src.vector_store.clip_extractor import CLIPEmbeddingExtractor

SAVE_DIR = "data/user_embeddings"
CROP_DIR = "data/user_crops"
UPLOAD_DIR = "data/uploads"
META_PATH = "data/user_embeddings/metadata.json"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_images_from_folder(folder):
    """Get all image files from a folder."""
    allowed = [".jpg", ".jpeg", ".png", ".webp"]
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if any(f.lower().endswith(ext) for ext in allowed):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def embed_user_image_hybrid(
    image_path: str,
    yolo_model_path: str = "models/trained/best.pt",
    device: str = "cpu",
    use_clip: bool = True
):
    """
    Process ONE IMAGE using hybrid YOLO + CLIP pipeline.

    Args:
        image_path: Path to user image
        yolo_model_path: Path to YOLO model
        device: 'cpu' or 'cuda'
        use_clip: If True, use CLIP embeddings; if False, use YOLO embeddings

    Returns:
        List of detected items with embeddings
    """

    # Initialize models
    detector = YoloV8Detector(model_path=yolo_model_path, device=device)

    if use_clip:
        extractor = CLIPEmbeddingExtractor(device=device)
        embedding_type = "CLIP"
        embedding_dim = 512
    else:
        from src.models.embeddings.yolov8_feature_extractor import YOLOv8FeatureExtractor
        extractor = YOLOv8FeatureExtractor(model_path=yolo_model_path, device=device)
        embedding_type = "YOLO"
        embedding_dim = extractor.embedding_dim

    print(f"\n🔍 Processing image: {image_path}")
    print(f"   Embedding type: {embedding_type} ({embedding_dim}-dim)")

    # Step 1: YOLO detection & cropping
    detections = detector.detect_and_crop(
        image_path,
        output_dir=CROP_DIR
    )

    if not detections:
        print("⚠ No clothing detected.")
        return []

    print(f"   ✓ Detected {len(detections)} items")

    results = []

    # Step 2: Extract embeddings from cropped items
    for i, det in enumerate(detections, 1):
        crop_path = det["crop_path"]
        category = det["category"]

        # Extract embedding
        if use_clip:
            emb = extractor.embed_single(crop_path)
        else:
            emb = extractor.extract_embedding(crop_path)

        # Generate unique item ID
        item_id = uuid.uuid4().hex

        # Save embedding
        emb_save = os.path.join(
            SAVE_DIR,
            f"{category}_{item_id}.npy"
        )
        np.save(emb_save, emb)

        print(f"   {i}/{len(detections)} ✓ {category:<10} | {embedding_type} embedding saved")

        # Create metadata entry
        results.append({
            "item_id": item_id,
            "category": category,
            "crop_path": crop_path,
            "embedding_path": emb_save,
            "embedding_type": embedding_type,
            "embedding_dim": embedding_dim,
            "bbox": det["bbox"],
            "confidence": det["conf"],
            "yolo_class": det["class_name"]
        })

    print(f"🎉 Finished processing → {len(results)} items embedded with {embedding_type}")
    return results


def save_metadata(new_items):
    """Append new items to metadata.json."""
    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            old = json.load(f)
    else:
        old = []

    old.extend(new_items)

    with open(META_PATH, "w") as f:
        json.dump(old, f, indent=4)

    print(f"📝 metadata.json updated → {len(old)} total items.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid YOLO + CLIP pipeline for clothing detection and embedding"
    )

    parser.add_argument("--img", nargs="*", help="One or multiple image paths")
    parser.add_argument("--folder", help="Process all images in folder")
    parser.add_argument("--model", type=str, default="models/trained/best.pt",
                       help="YOLO model path")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device: 'cpu' or 'cuda'")
    parser.add_argument("--embedding", type=str, default="clip",
                       choices=["clip", "yolo"],
                       help="Embedding type: 'clip' (better similarity) or 'yolo' (faster)")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # Determine image sources
    if args.img:
        images = args.img
        print("📌 Mode: individual images (--img)")
    elif args.folder:
        images = get_images_from_folder(args.folder)
        print(f"📌 Mode: folder processing (--folder): {args.folder}")
    else:
        images = get_images_from_folder(UPLOAD_DIR)
        print(f"📌 Mode: auto-processing default folder: {UPLOAD_DIR}")

    if len(images) == 0:
        print("❌ ERROR: No images found.")
        exit()

    print(f"📸 Total images to process: {len(images)}")
    print(f"🔧 Embedding mode: {args.embedding.upper()}")
    print(f"{'='*60}\n")

    use_clip = (args.embedding == "clip")

    for img in images:
        items = embed_user_image_hybrid(
            image_path=img,
            yolo_model_path=args.model,
            device=args.device,
            use_clip=use_clip
        )
        save_metadata(items)

    print(f"\n{'='*60}")
    print("✅ PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Embeddings saved to: {SAVE_DIR}")
    print(f"Metadata saved to: {META_PATH}")
