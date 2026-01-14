import os
import subprocess

# 1. Generate dummy images
from PIL import Image
os.makedirs("dummy_data", exist_ok=True)
for i in range(1, 5):
    img = Image.new("RGB", (224, 224), color=(i*50, i*50, i*50))
    img.save(f"dummy_data/img{i}.jpg")

# 2. Generate CSVs
train_csv = """image_path,label
dummy_data/img1.jpg,0
dummy_data/img2.jpg,1
"""
val_csv = """image_path,label
dummy_data/img3.jpg,0
dummy_data/img4.jpg,1
"""

with open("dummy_train.csv", "w") as f:
    f.write(train_csv)

with open("dummy_val.csv", "w") as f:
    f.write(val_csv)

# 3. Generate minimal YAML config
config_yaml = """seed: 42

dataset:
  train_csv: dummy_train.csv
  val_csv: dummy_val.csv
  num_classes: 2

model:
  name: resnet18

training:
  batch_size: 2
  lr: 0.001
  epochs: 1

wandb:
  project: "dummy_test"
  entity: "your_wandb_entity"
  use: false
"""

with open("dummy_config.yaml", "w") as f:
    f.write(config_yaml)

# 4. Run training
subprocess.run(["python", "train.py", "dummy_config.yaml"])
