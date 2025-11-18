# eval_fewshot_with_log.py
import os
import torch
from torchvision import models
from Adaptive_Weighted import prepare_adaptive_augmentation  
from datetime import datetime

# ==============================
# 配置
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 61
batch_size = 32
img_size = 224

EMPTY_TRAIN_JSON = r"D:\桌面\数维杯\31718_test\clean_labels.json"
EMPTY_TRAIN_DIR = r"D:\桌面\数维杯\images-31718"
VAL_JSON = r"D:\桌面\数维杯\images-4540\clean_labels.json"
VAL_IMG_DIR = r"D:\桌面\数维杯\images-4540\images"
MODEL_PATH = r"D:\桌面\数维杯\train2_result\final_model_gpu.pth"  
LOG_DIR = r"D:\桌面\数维杯\train2_result\logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ==============================
# 数据加载（只加载验证集）
# ==============================
print("============================================================")
print("开始加载验证集...")
print("============================================================")

val_loader, _, class_groups = prepare_adaptive_augmentation(
    train_json=EMPTY_TRAIN_JSON,
    train_img_dir=EMPTY_TRAIN_DIR,
    val_json=VAL_JSON,
    val_img_dir=VAL_IMG_DIR,
    batch_size=batch_size,
    num_workers=0,
    img_size=img_size
)

# ==============================
# 加载模型
# ==============================
print("============================================================")
print("加载训练好的模型...")
print("============================================================")

import torchvision.models as models
import torch.nn as nn

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(in_features, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.2),
    nn.Linear(512, num_classes)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ==============================
# 验证少样本类准确率
# ==============================
print("============================================================")
print("开始验证少样本类准确率...")
print("============================================================")

# 检测少样本类 (<10 张图)
def detect_fewshot_classes_from_loader(loader, threshold=10):
    counts = [0]*num_classes
    for _, labels in loader:
        for label in labels:
            counts[label.item()] += 1
    few_classes = [i for i,c in enumerate(counts) if c < threshold]
    return few_classes, counts

few_classes, class_counts = detect_fewshot_classes_from_loader(val_loader)
print(f"少样本类索引: {few_classes}")

few_correct, few_total = 0, 0
class_correct = {cls:0 for cls in few_classes}
class_total = {cls:0 for cls in few_classes}

with torch.no_grad():
    for images, labels in tqdm(val_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs,1)
        for cls in few_classes:
            mask = labels==cls
            if mask.sum()>0:
                class_correct[cls] += (preds[mask]==labels[mask]).sum().item()
                class_total[cls] += mask.sum().item()
                few_correct += (preds[mask]==labels[mask]).sum().item()
                few_total += mask.sum().item()

print("------------------------------------------------------------")
print("各少样本类准确率:")
for cls in few_classes:
    if class_total[cls]>0:
        acc = class_correct[cls]/class_total[cls]
        print(f"Class {cls}: {acc:.4f} ({class_correct[cls]}/{class_total[cls]})")
print("------------------------------------------------------------")
if few_total>0:
    print(f"少样本类总准确率: {few_correct/few_total:.4f} ({few_correct}/{few_total})")
print("验证完成")
