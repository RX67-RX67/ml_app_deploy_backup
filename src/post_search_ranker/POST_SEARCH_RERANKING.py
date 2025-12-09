import numpy as np
import os


class OutfitReranker:

    def __init__(self, metadata_dict, w1=1.0, w2=1.0, w3=1.0, w_prompt=2.0):
        self.meta = metadata_dict
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w_prompt = w_prompt

        # Auto-detect project root (src/post_search_ranker/ -> go up 2 levels)
        self.PROJECT_ROOT = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )

    # -------------------------------------------------------
    def cosine(self, a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    # -------------------------------------------------------
    def _resolve_path(self, path):
        """Return correct full path regardless of relative or absolute."""
        if os.path.isabs(path):
            return path

        # If file exists as is, use it
        if os.path.exists(path):
            return path

        # Otherwise assume relative to project root
        return os.path.join(self.PROJECT_ROOT, path)

    # -------------------------------------------------------
    def get_emb(self, item_id):
        emb_path = self.meta[item_id]["embedding_path"]

        full_path = self._resolve_path(emb_path)

        emb = np.load(full_path).astype("float32").reshape(-1)

        # normalize
        emb = emb / (np.linalg.norm(emb) + 1e-8)

        return emb

    # -------------------------------------------------------
    def score_outfit(self, outfit, prompt_emb=None):
        top_id = outfit["items"]["tops"]
        bottom_id = outfit["items"]["bottoms"]
        shoes_id = outfit["items"]["shoes"]

        top_emb = self.get_emb(top_id)
        bottom_emb = self.get_emb(bottom_id)
        shoes_emb = self.get_emb(shoes_id)

        # internal coherence
        s1 = self.cosine(top_emb, bottom_emb)
        s2 = self.cosine(top_emb, shoes_emb)
        s3 = self.cosine(bottom_emb, shoes_emb)

        score = self.w1 * s1 + self.w2 * s2 + self.w3 * s3

        # prompt alignment
        if prompt_emb is not None:
            prompt_emb = prompt_emb.reshape(-1)
            prompt_emb = prompt_emb / (np.linalg.norm(prompt_emb) + 1e-8)

            sp = (
                self.cosine(top_emb, prompt_emb)
                + self.cosine(bottom_emb, prompt_emb)
                + self.cosine(shoes_emb, prompt_emb)
            )
            score += self.w_prompt * sp

        return score

    # -------------------------------------------------------
    def rerank(self, outfits, prompt_emb=None):
        scored = []
        for o in outfits:
            s = self.score_outfit(o, prompt_emb)
            scored.append((s, o))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored
