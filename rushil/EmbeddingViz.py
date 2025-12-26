import os
import random
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# CONFIG
# -----------------------------
EMBEDDING_DIR = "embeddings"
METADATA_PATH = "metadata/embeddings.csv"
MAX_SAMPLES = 10000        # IMPORTANT: keep <= 10k
PERPLEXITY = 30
RANDOM_SEED = 42

# -----------------------------
# LOAD METADATA
# -----------------------------
df = pd.read_csv(METADATA_PATH)
print(df.columns)

# Optional: subsample
if len(df) > MAX_SAMPLES:
    df = df.sample(MAX_SAMPLES, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"Using {len(df)} samples for t-SNE")

# -----------------------------
# LOAD EMBEDDINGS
# -----------------------------
embeddings = []
labels = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    emb_path = row["vector_path"]
    emb = torch.load(emb_path, map_location="cpu")

    embeddings.append(emb.numpy())
    labels.append(row["label"])

X = np.vstack(embeddings)

# -----------------------------
# ENCODE LABELS
# -----------------------------
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# -----------------------------
# t-SNE
# -----------------------------
print("Running t-SNE...")
tsne = TSNE(
    n_components=2,
    perplexity=PERPLEXITY,
    random_state=RANDOM_SEED,
    init="pca",
    learning_rate="auto"
)

X_2d = tsne.fit_transform(X)

# -----------------------------
# PLOT
# -----------------------------
plt.figure(figsize=(12, 10))
scatter = plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=y,
    cmap="tab20",
    s=5,
    alpha=0.7
)

plt.title("t-SNE of CLIP Object Embeddings (MS COCO)")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")

# Legend (optional – can get crowded)
handles, _ = scatter.legend_elements()
plt.legend(
    handles,
    label_encoder.classes_,
    title="Category",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize="small"
)

plt.tight_layout()
plt.savefig("tsne_plot.png", dpi=300)
print("Saved plot to tsne_plot.png")

