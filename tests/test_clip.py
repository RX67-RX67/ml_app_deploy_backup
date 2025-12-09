"""
Test: CLIP Embedding Extractor
"""

import os
import sys

# ---------------------------------------------------------
# 1) 让 Python 能正确定位到 src/
# ---------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(">>> PROJECT ROOT =", PROJECT_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(">>> sys.path =", sys.path)

# ---------------------------------------------------------
# 2) Import 模块（如果路径没问题会成功）
# ---------------------------------------------------------
from src.vector_store.clip_extractor import CLIPEmbeddingExtractor


def test_clip_embedding():
    print("\n[TEST] Running CLIP Embedding Test...")

    clip = CLIPEmbeddingExtractor(device="cpu")

    # 你必须保证有一个 crop 图片
    test_crop_path = "data/test_tmp_crops/bottom_98fdcd7376e84479a5ce27788566f0e6.jpg"

    if not os.path.exists(test_crop_path):
        raise FileNotFoundError(
            f"❌ 测试失败：找不到测试图片 {test_crop_path}\n"
            f"请先运行 YOLO 测试生成 crop 文件，再测试 CLIP"
        )

    emb = clip.embed_single(test_crop_path)

    print("EMBEDDING SHAPE =", emb.shape)

    assert emb.shape == (512,), "❌ CLIP embedding shape 不正确，应为 (512,)"
    print("✅ CLIP embedding test passed!")


if __name__ == "__main__":
    test_clip_embedding()
