#!/usr/bin/env python3
import os

DATA_YAML = """
train: data/images/train
val: data/images/val
nc: 6
names: ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
"""

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    with open("data/train.yaml", "w") as f:
        f.write(DATA_YAML)

    print("=== TRAINING SETUP HOÀN TẤT ===")
    print("1. Put images/labels into data/images/train & data/labels/train")
    print(
        "2. Run: python train.py --img 640 --batch 16 --epochs 100 --data data/train.yaml --weights models/yolov5m.pt")