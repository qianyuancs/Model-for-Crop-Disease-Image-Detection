
import os
import json
import csv
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    precision_score, classification_report,
    confusion_matrix
)
from torchvision import transforms

from Q3_train import SeverityModel, label_to_species, label_to_severity

VAL_JSON = r"C:\Users\21978\Desktop\大学\赛程知识\数模赛\数维杯\Question1\Yclean_labels.json"
VAL_IMG_DIR = "images"
BEST_MODEL = r"C:\Users\21978\Desktop\大学\赛程知识\数模赛\数维杯\Question1\checkpoints_task3_v3\best_model_task3_v3.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

SAVE_DIR = "results_task3"
os.makedirs(SAVE_DIR, exist_ok=True)


# transforms

val_transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE*1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])


def load_annotations(json_path, img_dir):
    with open(json_path, 'r', encoding='utf-8') as f:
        ann = json.load(f)

    items = []
    for item in ann:
        lid = int(item['disease_class'])
        img_id = item['image_id']
        img_path = os.path.join(img_dir, img_id)
        species = label_to_species[lid]

        severity = label_to_severity(lid)   

        items.append((img_path, species, severity, lid))
    return items


@torch.no_grad()
def run_inference():
    print("Loading data...")
    val_items = load_annotations(VAL_JSON, VAL_IMG_DIR)

    # -------- load model --------
    print("Loading model...")
    model = SeverityModel(backbone_pretrain_path=None, freeze_backbone=False)
    ckpt = torch.load(BEST_MODEL, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    y_true = []
    y_pred = []

    # -------- inference --------
    print("Running inference...")
    for img_path, sp, sev, lid in tqdm(val_items):
        img = Image.open(img_path).convert("RGB")
        img = val_transform(img).unsqueeze(0).to(DEVICE)

        logits = model(img)
        pred = torch.argmax(logits, dim=1).item()

        y_true.append(sev)
        y_pred.append(pred)


    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    recalls = recall_score(y_true, y_pred, average=None, labels=[0,1,2])
    precisions = precision_score(y_true, y_pred, average=None, labels=[0,1,2])
    f1s = f1_score(y_true, y_pred, average=None, labels=[0,1,2])

    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])

    # 生成分类报告
    cls_report = classification_report(
        y_true, y_pred,
        digits=4,
        target_names=['Healthy','General','Serious'],
        output_dict=False
    )

    print("\n===== FINAL RESULTS =====")
    print(f"Accuracy = {acc:.4f}")
    print(f"Macro F1 = {macro_f1:.4f}")
    print(f"Recalls  = {recalls}")
    print("\nClassification Report:\n", cls_report)


    results_json = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": {
            "Healthy": {"precision": float(precisions[0]), "recall": float(recalls[0]), "f1": float(f1s[0])},
            "General": {"precision": float(precisions[1]), "recall": float(recalls[1]), "f1": float(f1s[1])},
            "Serious": {"precision": float(precisions[2]), "recall": float(recalls[2]), "f1": float(f1s[2])},
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": cls_report
    }

    json_path = os.path.join(SAVE_DIR, "Q3_metrics2.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=4, ensure_ascii=False)

    print(f"\nSaved JSON to: {json_path}")


    csv_path = os.path.join(SAVE_DIR, "Q3_metrics2.csv")
    with open(csv_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Precision", "Recall", "F1"])
        writer.writerow(["Healthy", precisions[0], recalls[0], f1s[0]])
        writer.writerow(["General", precisions[1], recalls[1], f1s[1]])
        writer.writerow(["Serious", precisions[2], recalls[2], f1s[2]])
        writer.writerow([])
        writer.writerow(["Overall Accuracy", acc])
        writer.writerow(["Macro F1", macro_f1])

    print(f"Saved CSV to: {csv_path}")


if __name__ == "__main__":
    run_inference()
