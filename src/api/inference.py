from datetime import datetime
import uuid
import os
import shutil

from fastapi import UploadFile
from src.models.pipeline.detect_crop_embed_hybrid import embed_user_image_hybrid

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def process_uploaded_image(file: UploadFile):
    
    # Save uploaded file
    file_id = uuid.uuid4().hex
    extension = file.filename.split(".")[-1]
    saved_path = os.path.join(UPLOAD_DIR, f"{file_id}.{extension}")

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"Saved uploaded image → {saved_path}")

    # Run YOLO + CLIP hybrid embedding pipeline
    results = embed_user_image_hybrid(saved_path)

    if not results:
        return {
            "timestamp": datetime.now(),
            "num_items": 0,
            "items": [],
            "message": "No clothing items detected."
        }

    # Format API response
    formatted_items = []
    for item in results:
        formatted_items.append({
            "item_id": item["item_id"],
            "category": item["category"],
            "crop_path": item["crop_path"],
            "embedding_path": item["embedding_path"],
        })

    return {
        "timestamp": datetime.now(),
        "num_items": len(formatted_items),
        "items": formatted_items,
        "message": "YOLO + CLIP hybrid embedding completed."
    }
