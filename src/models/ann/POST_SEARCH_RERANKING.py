import numpy as np
import os

class OutfitReranker:

    def __init__(self, metadata_dict, w1=1.0, w2=1.0, w3=1.0, w_prompt=2.0):
        self.meta = metadata_dict
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w_prompt = w_prompt

    def cosine(self, a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def get_emb(self, item_id):
        emb_path = self.meta[item_id]["embedding_path"]

        # ⭐ Docker 环境内正确的路径
        full_path = os.path.join("/app", emb_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Embedding not found: {full_path}")

        emb = np.load(full_path).astype("float32")

        # normalize
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb

    def score_outfit(self, outfit, prompt_emb=None):

        top_id    = outfit["items"]["tops"]
        bottom_id = outfit["items"]["bottoms"]
        shoes_id  = outfit["items"]["shoes"]

        top_emb    = self.get_emb(top_id)
        bottom_emb = self.get_emb(bottom_id)
        shoes_emb  = self.get_emb(shoes_id)

        # internal coherence
        s1 = self.cosine(top_emb, bottom_emb)
        s2 = self.cosine(top_emb, shoes_emb)
        s3 = self.cosine(bottom_emb, shoes_emb)

        score = self.w1*s1 + self.w2*s2 + self.w3*s3

        # optional — prompt alignment
        if prompt_emb is not None:
            sp = (
                self.cosine(top_emb, prompt_emb) +
                self.cosine(bottom_emb, prompt_emb) +
                self.cosine(shoes_emb, prompt_emb)
            )
            score += self.w_prompt * sp

        return score

    def rerank(self, outfits, prompt_emb=None):
        scored = []
        for o in outfits:
            s = self.score_outfit(o, prompt_emb)
            scored.append((s, o))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored
