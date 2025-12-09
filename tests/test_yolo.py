import os
import sys

# ----------------------------------------------------
# 让 Python 知道 "src" 在哪里
# ----------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

print(">>> PROJECT ROOT =", ROOT_DIR)
print(">>> sys.path =", sys.path[:3])   # debug

# ----------------------------------------------------
# 现在可以 import 你自己的模块
# ----------------------------------------------------
from src.detectors.yolov8_detector import YoloV8Detector


def test_yolo_detector():
    detector = YoloV8Detector(
        model_path="models/trained/best.pt",
        device="cpu"
    )

    img = "data/test_sample/test_sample.PNG"

    dets = detector.detect_and_crop(img, output_dir="data/test_tmp_crops")

    print("DETECTIONS =", dets)
    assert len(dets) > 0, "YOLO 没检测到衣服"


if __name__ == "__main__":
    test_yolo_detector()
    print("\n✅ TEST PASSED!")
