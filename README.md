# CMPSC189 - Visual Layer

**Team Members:**  
- Alec Song  
- Rushil Gupta
- Kushagra Kanaujia
- Saeed Arellano
- Bhavya Ranjan

## Overview

This project explores how *structured label noise* affects model performance on large-scale vision datasets.  
The `noise_injection.py` script is responsible for generating **noisy label variants** of a cleaned ImageNet dataset using the image embeddings produced by Rushil’s embedding model. These variants are later consumed by the training pipeline to evaluate robustness under different noise patterns.

## What `noise_injection.py` Does

`noise_injection.py` takes in:

- A `.npy` embedding file (from the embedding model)  
- A clean labels CSV file (from the cleaned dataset)  
- Hyperparameters specifying the **noise type** and **severity**

Using these inputs, it corrupts labels in embedding space with the following methods:

- **Nearest-neighbor noise (`nearest_neighbor`):**  
  Randomly select points and flip each label to the label of its nearest neighbor in embedding space.

- **Cluster-based noise (`cluster`):**  
  Build small clusters around selected seeds (k-NN in embedding space) and flip entire clusters to the label of the nearest external example.

- **Boundary nearest-neighbor noise (`boundary_nearest`):**  
  First detect **boundary points** whose neighbors contain multiple labels (hard / ambiguous examples), then flip only these points to their nearest neighbor’s label.

- **Boundary cluster noise (`boundary_cluster`):**  
  Use boundary points as seeds, form clusters around them in embedding space, and flip entire clusters located near the boundary between classes.

The script outputs:

1. A **noisy labels CSV** that can directly replace the clean labels in the training pipeline.  
2. A **log CSV** detailing which examples were changed, their original/new labels, and cluster/boundary metadata.

## How to Use

Basic usage examples:

Nearest-neighbor noise (random seeds):

```bash
python noise_injection.py \
  --embeddings imagenet_embeddings.npy \
  --labels imagenet_clean_labels.csv \
  --output-labels labels_noise_nn_05.csv \
  --log-file labels_noise_nn_05_log.csv \
  --mode nearest_neighbor \
  --noise-level 0.05 \
  --cluster-size 1 \
  --metric cosine \
  --random-seed 42
