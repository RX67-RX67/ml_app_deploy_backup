import os
from ultralytics import YOLO

def train_yolo():

    model = YOLO("yolov8n.pt")

    model.train(
        data="models/trained/fashion3.yaml",  
        project="runs/train",
        name="deepfashion_yolo",
        epochs=50,
        imgsz=640,
        batch=16,
        device="cpu",   
        pretrained=True
    )

    run_dir = "runs/train/deepfashion_yolo"
    src = os.path.join(run_dir, "weights", "best.pt")

    dst_dir = "models/trained"
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "best_custom.pt")

    if os.path.exists(src):
        os.replace(src, dst)
        print(f" Saved trained custom model → {dst}")
    else:
        print("⚠️ ERROR: Could not find best.pt — did training complete?")



if __name__ == "__main__":
    train_yolo()
