"""
Simple embeddings builder using original CLIP extractor.
Workaround for torch version compatibility issues.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import original CLIP extractor
sys.path.insert(0, str(Path(__file__).parent))
from clip_extractor import CLIPEmbeddingExtractor


def build_embeddings(
    metadata_file: str,
    output_file: str,
    image_column: str = "image_path",
    image_root: str = ".",
    batch_size: int = 32,
):
    """Build CLIP embeddings from metadata."""

    metadata_path = Path(metadata_file)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    output_path = Path(output_file)

    # Load metadata
    df = pd.read_csv(metadata_path)
    if image_column not in df.columns:
        raise KeyError(f"Column '{image_column}' not found. Available: {list(df.columns)}")

    root = Path(image_root)
    image_paths = [root / p for p in df[image_column].astype(str).tolist()]
    total_images = len(image_paths)

    print(f"\n{'='*60}")
    print(f"CLIP EMBEDDING EXTRACTION")
    print(f"{'='*60}")
    print(f"Total images: {total_images:,}")
    print(f"Output file: {output_path}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*60}\n")

    # Initialize extractor
    print("Loading CLIP model...")
    extractor = CLIPEmbeddingExtractor()
    print("✓ Model loaded\n")

    # Process in batches with progress bar
    all_embeddings = []

    for start in tqdm(range(0, total_images, batch_size), desc="Extracting embeddings"):
        batch_paths = image_paths[start : start + batch_size]

        try:
            batch_embeddings = extractor.embed_batch(batch_paths, batch_size=batch_size)
            all_embeddings.append(batch_embeddings)
        except Exception as e:
            print(f"\n⚠️  Error processing batch {start}-{start+batch_size}: {e}")
            # Try one by one
            batch_embs = []
            for path in batch_paths:
                try:
                    emb = extractor.embed_single(path)
                    if emb is not None:
                        batch_embs.append(emb)
                except Exception as e2:
                    print(f"  Failed: {path}")
            if batch_embs:
                all_embeddings.append(np.array(batch_embs))

    # Combine all embeddings
    if not all_embeddings:
        raise RuntimeError("No embeddings generated!")

    final_embeddings = np.concatenate(all_embeddings, axis=0)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, final_embeddings)

    # Summary
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Output file: {output_path}")
    print(f"Embeddings shape: {final_embeddings.shape}")
    print(f"Dtype: {final_embeddings.dtype}")
    print(f"{'='*60}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Build CLIP embeddings")
    parser.add_argument("--metadata-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--image-column", type=str, default="image_path")
    parser.add_argument("--image-root", type=str, default=".")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_embeddings(
        metadata_file=args.metadata_file,
        output_file=args.output_file,
        image_column=args.image_column,
        image_root=args.image_root,
        batch_size=args.batch_size,
    )
