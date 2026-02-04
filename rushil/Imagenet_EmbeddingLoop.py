#!/usr/bin/env python3
"""
MS COCO Object-Level Embedding Pipeline

- Streams MS COCO from Hugging Face
- Crops individual objects using bounding boxes
- Embeds object crops with CLIP (ViT-B/32)
- Saves embeddings to disk
- Saves metadata (CSV + JSON) linking embeddings to source objects
"""

import os
import json
import uuid
from tqdm import tqdm

import torch
import pandas as pd
from PIL import Image

from datasets import load_dataset
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------

EMBED_DIR = "embeddings"
META_DIR = "metadata"

CSV_PATH = os.path.join(META_DIR, "INET_embeddings.csv")
JSON_PATH = os.path.join(META_DIR, "INET_embeddings.json")

MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
DATASET_NAME = "imagenet-1k"
DATASET_SPLIT = "train"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Setup
# -----------------------------

def setup_dirs():
    os.makedirs(EMBED_DIR, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)


def crop_object(image: Image.Image, bbox):
    """
    Crop a bounding box from an image.

    COCO bbox format: [x, y, width, height]
    """
    x, y, w, h = bbox
    return image.crop((x, y, x + w, y + h))


# -----------------------------
# Main
# -----------------------------

def main():
    print(f"Using device: {DEVICE}")
    setup_dirs()

    # Load model
    print("Loading CLIP model...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    # Stream dataset
    print("Streaming MS COCO dataset...")
    dataset = load_dataset(
        DATASET_NAME,
        split=DATASET_SPLIT,
        streaming=True
    )

    
    metadata_rows = []
    metadata_json = []

    # Process dataset
    for idx, sample in enumerate(tqdm(dataset)):
        try:
            image: Image.Image = sample["image"].convert("RGB")
            label = int(sample["label"])
        except Exception:
            continue

        with torch.no_grad():
            embedding = model.encode(
                image,
                convert_to_tensor=True,
                normalize_embeddings=True
            )

        vector_id = str(uuid.uuid4())
        vector_path = os.path.join(EMBED_DIR, f"{vector_id}.pt")
        torch.save(embedding.cpu(), vector_path)

        row = {
            "vector_id": vector_id,
            "vector_path": vector_path,
            "label": label,
            "sample_idx": idx
        }

        metadata_rows.append(row)
        metadata_json.append(row)

    # Save metadata
    print("Saving metadata...")
    df = pd.DataFrame(metadata_rows)
    df.to_csv(CSV_PATH, index=False)

    with open(JSON_PATH, "w") as f:
        json.dump(metadata_json, f, indent=2)

    print("Done.")
    print(f"Embeddings saved to: {EMBED_DIR}/")
    print(f"Metadata saved to: {META_DIR}/")


if __name__ == "__main__":
    main()
