import os
import numpy as np
import argparse
import clip
import torch

# -------------------------------------------------
# Simple CLIP Text Encoder (OpenAI official version)
# -------------------------------------------------
class SimpleCLIPTextEncoder:
    def __init__(self, device="cpu"):
        self.device = device

        print(f"[CLIP] Loading ViT-B/32 model on {device} ...")
        model, _ = clip.load("ViT-B/32", device="cpu")  # load CPU first for safety
        self.model = model.to(device)
        self.model.eval()
        print("[CLIP] Model loaded.")

    def encode(self, text: str) -> np.ndarray:
        tokens = clip.tokenize([text])  # always CPU
        tokens = tokens.to(self.device)

        with torch.no_grad():
            emb = self.model.encode_text(tokens)

        emb = emb.cpu().numpy().astype("float32")[0]

        # normalize
        emb /= (np.linalg.norm(emb) + 1e-8)

        return emb


# -------------------------------------------------
# Helper: convert prompt → slug filename
# -------------------------------------------------
def slugify(text: str) -> str:
    slug = text.lower().strip()
    # basic safe conversion
    for ch in [" ", "/", "\\", ",", ".", "!", "?", ":", ";", "'", '"', "(", ")", "[", "]"]:
        slug = slug.replace(ch, "_")
    return slug


# -------------------------------------------------
# Main script
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate CLIP text embedding for a prompt.")
    parser.add_argument("--prompt", type=str, required=True, help="User text prompt")
    args = parser.parse_args()

    encoder = SimpleCLIPTextEncoder(device="cpu")

    # generate embedding
    emb = encoder.encode(args.prompt)

    # build save path
    slug = slugify(args.prompt)

    # relative folder: ../../../data/prompt_embeddings/
    save_dir = os.path.join("../../../", "data", "prompt_embeddings")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{slug}.npy")

    # save npy
    np.save(save_path, emb)

    print(f"\n[SAVED] Prompt embedding saved to:\n  {save_path}")
    print(f"[INFO] Embedding shape: {emb.shape}")


if __name__ == "__main__":
    main()

