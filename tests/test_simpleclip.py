"""
Test SimpleCLIPTextEncoder module
"""

import os
import sys
import numpy as np

# -------------------------------------------------------------------
# Ensure project root added to sys.path
# -------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f">>> PROJECT ROOT = {PROJECT_ROOT}")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print(f">>> sys.path[:3] = {sys.path[:3]}")

# -------------------------------------------------------------------
# Import module under test
# -------------------------------------------------------------------
from src.text_embedding.SIMPLECLIP import SimpleCLIPTextEncoder


def test_text_encoder():
    print("\n[TEST] Running SimpleCLIPTextEncoder unit test...\n")

    encoder = SimpleCLIPTextEncoder(device="cpu")

    text = "a black leather jacket"
    emb = encoder.encode(text)

    print("Embedding shape:", emb.shape)
    print("Embedding dtype:", emb.dtype)
    print("Embedding norm:", np.linalg.norm(emb))

    # ------------------ Assertions ------------------
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (512,)
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-3  # should be normalized

    print("\n[TEST PASSED] Text encoder working correctly!\n")


if __name__ == "__main__":
    test_text_encoder()
