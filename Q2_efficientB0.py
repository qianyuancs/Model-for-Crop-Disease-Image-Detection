
import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms

from Adaptive_Weighted import prepare_adaptive_augmentation  # 你已有的增强模块


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
num_classes = 61
batch_size = 32
img_size = 224
seed = 42
stage1_epochs = 10
stage2_epochs = 20
lr_stage1 = 3e-3
lr_stage2 = 1e-4
head_dropout = 0.4
mixup_alpha = 0.2
use_mixup = True
label_smoothing = 0.1
SAVE_DIR = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

TRAIN_JSON = "Tclean_labels.json"
TRAIN_IMG_DIR = "images-31718"
VAL_JSON = "Yclean_labels.json"
VAL_IMG_DIR = "images"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# 检测少样本类 (< threshold)
def detect_fewshot_classes(train_loader, threshold=10):
    counts = [0]*num_classes
    for _, labels in train_loader:
        for label in labels:
            counts[label.item()] += 1
    few_classes = [i for i,c in enumerate(counts) if c < threshold]
    normal_classes = [i for i,c in enumerate(counts) if c >= threshold]
    return few_classes, normal_classes, counts

# MixUp
def mixup_data(x, y, alpha=0.2, device='cpu'):
    if alpha <= 0:
        return x, y, None, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam*x + (1-lam)*x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# 原型网络损失
def prototype_loss(features, labels, prototypes):
    loss = 0.0
    count = 0
    for i, cls in enumerate(labels):
        if cls.item() in prototypes:
            proto = prototypes[cls.item()].to(features.device)
            loss += ((features[i]-proto)**2).sum()
            count += 1
    return loss / max(count,1)


if __name__ == "__main__":
    set_seed(seed)


    train_loader, val_loader, class_groups = prepare_adaptive_augmentation(
        train_json=TRAIN_JSON,
        train_img_dir=TRAIN_IMG_DIR,
        val_json=VAL_JSON,
        val_img_dir=VAL_IMG_DIR,
        batch_size=batch_size,
        num_workers=4, 
        img_size=img_size,
        
    )


    few_classes, normal_classes, class_counts = detect_fewshot_classes(train_loader, threshold=10)
    print("极少样本类:", few_classes)

    # 类别权重
    class_weights = [1.0/(c+1e-6) for c in class_counts]
    class_weights = torch.tensor(class_weights).to(device)


    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=head_dropout),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )
    model = model.to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"Model total params: {total_params:,}, trainable params: {trainable_params:,}")


    for name, param in model.named_parameters():
        if not name.startswith("classifier"):
            param.requires_grad = False

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_stage1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage1_epochs+stage2_epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    # 初始化少样本原型
    prototypes = {cls: torch.zeros(512) for cls in few_classes}

    # Stage1 训练

    for epoch in range(stage1_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Stage1 Epoch {epoch+1}/{stage1_epochs}"):
            images, labels = images.to(device), labels.to(device)

            # MixUp 仅用于普通类
            if use_mixup:
                mask = torch.tensor([l in normal_classes for l in labels]).to(device)
                if mask.sum()>1:
                    images_mix, labels_a, labels_b, lam = mixup_data(images[mask], labels[mask], mixup_alpha, device=device)
                    outputs = model(images)
                    loss = lam*criterion(outputs[mask], labels_a)+(1-lam)*criterion(outputs[mask], labels_b)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            # 特征提取统一：使用 avgpool + flatten
            feats = model.features(images)
            feats = model.avgpool(feats)
            feats = torch.flatten(feats,1)
            feats = model.classifier[0](feats)

            # 原型损失
            proto_loss = prototype_loss(feats, labels, prototypes)

            # 动态更新原型（EMA）
            for cls in few_classes:
                cls_mask = labels==cls
                if cls_mask.sum()>0:
                    cls_feats = feats[cls_mask]
                    prototypes[cls] = 0.9*prototypes[cls] + 0.1*cls_feats.mean(0).detach().cpu()

            total_loss = loss + proto_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()*images.size(0)
            _, preds = torch.max(outputs.detach(),1)
            correct += torch.sum(preds==labels).item()
            total += labels.size(0)

        print(f"[Stage1] Epoch {epoch+1}: Train Loss={running_loss/total:.4f}, Train Acc={correct/total:.4f}")

        # 验证少样本类
        if val_loader is not None:
            model.eval()
            few_correct, few_total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs,1)
                    mask = torch.tensor([l in few_classes for l in labels])
                    if mask.sum()>0:
                        few_correct += (preds[mask]==labels[mask]).sum().item()
                        few_total += mask.sum().item()
            if few_total>0:
                print(f"[Stage1] Few-shot Val Acc: {few_correct/few_total:.4f}")
            model.train()

        scheduler.step()


    # Stage2 微调最后 block + classifier

    for name, param in model.named_parameters():
        if "features.6" in name or name.startswith("classifier"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_stage2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage2_epochs)

    for epoch in range(stage2_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Stage2 Epoch {epoch+1}/{stage2_epochs}"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            feats = model.features(images)
            feats = model.avgpool(feats)
            feats = torch.flatten(feats,1)
            feats = model.classifier[0](feats)

            proto_loss = prototype_loss(feats, labels, prototypes)

            # 动态更新原型
            for cls in few_classes:
                cls_mask = labels==cls
                if cls_mask.sum()>0:
                    cls_feats = feats[cls_mask]
                    prototypes[cls] = 0.9*prototypes[cls] + 0.1*cls_feats.mean(0).detach().cpu()

            total_loss = loss + proto_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()*images.size(0)
            _, preds = torch.max(outputs.detach(),1)
            correct += torch.sum(preds==labels).item()
            total += labels.size(0)

        print(f"[Stage2] Epoch {epoch+1}: Train Loss={running_loss/total:.4f}, Train Acc={correct/total:.4f}")

        # 验证少样本类
        if val_loader is not None:
            model.eval()
            few_correct, few_total = 0,0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs,1)
                    mask = torch.tensor([l in few_classes for l in labels])
                    if mask.sum()>0:
                        few_correct += (preds[mask]==labels[mask]).sum().item()
                        few_total += mask.sum().item()
            if few_total>0:
                print(f"[Stage2] Few-shot Val Acc: {few_correct/few_total:.4f}")
            model.train()

    print("训练完成")
