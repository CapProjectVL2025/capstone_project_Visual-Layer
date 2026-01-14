# Training pipeline

## Setup
1. Create Python environment and install deps:
```bash
pip install -r requirements.txt
```

2. ImageNet1k as ImageFolder: DATA_ROOT/train/<class_id>/*.jpg, DATA_ROOT/val/<class_id>/*.jpg

Edit config.py DATA_ROOT path if necessary

3. Provide WANDB credentials in .env file:
export WANDB_API_KEY=your_key_here

4. Run using:
```bash
python train.py
```

## Output
Checkpoints in runs/
