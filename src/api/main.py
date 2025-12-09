from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .inference import process_uploaded_image
from .schemas import YoloEmbedResponse
from fastapi.responses import FileResponse
import os

app = FastAPI(
    title="SnapStyle - YOLO Detection & Embedding API",
    description="Detect clothes, crop items, extract embeddings using YOLO+CLIP pipeline.",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok", "service": "YOLO embedding API"}

@app.post("/yolo/embed", response_model=YoloEmbedResponse)
async def yolo_embed_endpoint(file: UploadFile = File(...)):
    try:
        result = process_uploaded_image(file)

        # Convert local crop_path → public URL
        for item in result["items"]:
            filename = os.path.basename(item["crop_path"])
            item["crop_path"] = f"http://localhost:8000/yolo/crop/{filename}"

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ⭐ YOU MUST PUT THIS BACK
CROP_DIR = "data/user_crops"

@app.get("/yolo/crop/{filename}")
async def get_crop_image(filename: str):
    file_path = os.path.join(CROP_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"{filename} not found")

    return FileResponse(file_path)
