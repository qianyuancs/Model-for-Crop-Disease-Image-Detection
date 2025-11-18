import os
import json
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


from Adaptive_Weighted import prepare_adaptive_augmentation


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 61
BATCH_SIZE = 32
IMG_SIZE = 224
NUM_WORKERS = 4

VAL_JSON = r"C:\Users\21978\Desktop\大学\赛程知识\数模赛\数维杯\Question1\Yclean_labels.json"
VAL_IMG_DIR = "images"
MODEL_PATH = r"C:\Users\21978\Desktop\大学\赛程知识\数模赛\数维杯\Question1\checkpoints\resnet50_best.pth"

SAVE_DIR = "results_task1"
os.makedirs(SAVE_DIR, exist_ok=True)

@torch.no_grad()
def run_inference():
    print("Loading validation data...")
    _, val_loader, _ = prepare_adaptive_augmentation(
        train_json=VAL_JSON,
        train_img_dir=VAL_IMG_DIR,
        val_json=VAL_JSON,
        val_img_dir=VAL_IMG_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        img_size=IMG_SIZE
    )
    
    print("Loading model...")
    # 定义模型结构
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    
    # 加载训练好的权重
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    print("Running inference...")
    y_true = []
    y_pred = []
    
    for images, labels in tqdm(val_loader, desc="Testing"):
        images = images.to(DEVICE)
        
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    # 输出结果
    print("\n" + "="*50)
    print("问题一测试结果 (61分类)")
    print("="*50)
    print(f"Accuracy  = {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Macro F1  = {macro_f1:.4f}")
    print("="*50)
    
    # 保存结果
    results = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "total_samples": len(y_true)
    }
    
    json_path = os.path.join(SAVE_DIR, "Q1_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\nResults saved to: {json_path}")
    
    return results


if __name__ == "__main__":
    try:
        results = run_inference()
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()