import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.snapstyle_pipeline import SnapStylePipeline

pipe = SnapStylePipeline(device="cpu")

# 1. 数字化一张图片
pipe.digitize_image("data/test_sample/test_sample.PNG")

# 2. 选一个 anchor
items = pipe.list_items()
anchor_id = items[0]["item_id"]

# 3. 一键生成 + rerank
ranked = pipe.recommend_outfits(
    anchor_id=anchor_id,
    prompt_text="formal interview outfit",
    ann_top_k=5,
    max_outfits=5,
)

for score, outfit in ranked:
    print(score, outfit)
