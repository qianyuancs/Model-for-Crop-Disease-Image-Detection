import os
import json
import random
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import csv
from sklearn.metrics import f1_score, accuracy_score, classification_report

# ==================== 配置 ====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 64  # 增大batch提速
NUM_EPOCHS_STAGE = [8, 8, 10]  # 总共25轮，压缩时间
LR = 1e-3
NUM_CLASSES_DISEASE = 61
NUM_CLASSES_SEVERITY = 3

# 数据路径
TRAIN_JSON = r"C:\Users\21978\Desktop\大学\赛程知识\数模赛\数维杯\Question1\relabelled_dataset.json"
VAL_JSON = r"C:\Users\21978\Desktop\大学\赛程知识\数模赛\数维杯\Question1\Yrelabelled_dataset.json"
TRAIN_IMG_DIR = "images-31718"
VAL_IMG_DIR = "images"
PRETRAIN_PATH = r"C:\Users\21978\Desktop\大学\赛程知识\数模赛\数维杯\Question1\checkpoints_task3_v3\best_model_task3_v3.pth"
SAVE_DIR = "outputs_progressive"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==================== 数据增强 ====================
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==================== Dataset ====================
class MultiTaskDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        with open(json_file, 'r', encoding='utf-8') as f:
            self.items = json.load(f)
        
        # 计算类别权重
        self.disease_counts = defaultdict(int)
        self.severity_counts = defaultdict(int)
        for item in self.items:
            self.disease_counts[item['disease_main_id']] += 1
            self.severity_counts[item['severity_id']] += 1
        
        # 逆频率权重
        total = len(self.items)
        self.disease_weights = {
            k: total / (len(self.disease_counts) * v) 
            for k, v in self.disease_counts.items()
        }
        self.severity_weights = {
            k: total / (len(self.severity_counts) * v) 
            for k, v in self.severity_counts.items()
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = os.path.join(self.img_dir, item['image_id'])
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        d_id = int(item['disease_main_id'])
        s_id = int(item['severity_id'])
        d_weight = self.disease_weights[d_id]
        s_weight = self.severity_weights[s_id]
        
        return img, d_id, s_id, d_weight, s_weight

# ==================== 模型（改进版：使用Embedding） ====================
class ProgressiveMultiTaskModel(nn.Module):
    def __init__(self, num_disease=61, num_severity=3, embedding_dim=128):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Identity()
        
        # 疾病分类头
        self.disease_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_disease)
        )
        
        # 疾病Embedding（条件建模的关键）
        self.disease_embedding = nn.Embedding(num_disease, embedding_dim)
        
        # 条件严重程度分类头
        self.severity_head = nn.Sequential(
            nn.Linear(2048 + embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_severity)
        )

    def forward(self, x):
        feat = self.backbone(x)
        disease_logits = self.disease_head(feat)
        
        # 条件建模：用预测的疾病ID获取embedding
        disease_pred = torch.argmax(disease_logits, dim=1)
        disease_emb = self.disease_embedding(disease_pred)
        
        cond_feat = torch.cat([feat, disease_emb], dim=1)
        severity_logits = self.severity_head(cond_feat)
        
        return disease_logits, severity_logits

# ==================== 评估函数 ====================
def evaluate(model, loader, device):
    model.eval()
    all_d_preds, all_d_labels = [], []
    all_s_preds, all_s_labels = [], []
    
    with torch.no_grad():
        for imgs, d_labels, s_labels, _, _ in loader:
            imgs = imgs.to(device)
            d_logits, s_logits = model(imgs)
            
            d_pred = torch.argmax(d_logits, dim=1)
            s_pred = torch.argmax(s_logits, dim=1)
            
            all_d_preds.extend(d_pred.cpu().numpy())
            all_d_labels.extend(d_labels.numpy())
            all_s_preds.extend(s_pred.cpu().numpy())
            all_s_labels.extend(s_labels.numpy())
    
    all_d_preds = np.array(all_d_preds)
    all_d_labels = np.array(all_d_labels)
    all_s_preds = np.array(all_s_preds)
    all_s_labels = np.array(all_s_labels)
    
    # 单任务指标
    disease_acc = accuracy_score(all_d_labels, all_d_preds)
    disease_f1 = f1_score(all_d_labels, all_d_preds, average='macro', zero_division=0)
    
    severity_acc = accuracy_score(all_s_labels, all_s_preds)
    severity_f1 = f1_score(all_s_labels, all_s_preds, average='macro', zero_division=0)
    
    # 联合准确率
    joint_correct = np.sum((all_d_preds == all_d_labels) & (all_s_preds == all_s_labels))
    joint_acc = joint_correct / len(all_d_labels)
    
    # 协同增益
    theoretical_joint = disease_acc * severity_acc
    synergy_gain = joint_acc - theoretical_joint
    
    # Severity召回率
    severity_recalls = []
    for i in range(NUM_CLASSES_SEVERITY):
        mask = (all_s_labels == i)
        if mask.sum() > 0:
            recall = np.mean(all_s_preds[mask] == all_s_labels[mask])
            severity_recalls.append(recall)
        else:
            severity_recalls.append(0.0)
    
    return {
        'disease_acc': disease_acc,
        'disease_f1': disease_f1,
        'severity_acc': severity_acc,
        'severity_f1': severity_f1,
        'severity_recalls': severity_recalls,
        'joint_acc': joint_acc,
        'theoretical_joint': theoretical_joint,
        'synergy_gain': synergy_gain
    }

# ==================== 训练主函数 ====================
def train_progressive():
    set_seed(42)
    
    # 加载数据
    print("Loading datasets...")
    train_dataset = MultiTaskDataset(TRAIN_JSON, TRAIN_IMG_DIR, train_transform)
    val_dataset = MultiTaskDataset(VAL_JSON, VAL_IMG_DIR, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=4, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=4, pin_memory=True)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # 初始化模型
    model = ProgressiveMultiTaskModel().to(DEVICE)
    
    # 加载预训练backbone
    if os.path.exists(PRETRAIN_PATH):
        print(f"Loading pretrained backbone from {PRETRAIN_PATH}")
        ckpt = torch.load(PRETRAIN_PATH, map_location='cpu')
        backbone_dict = {k.replace('backbone.', ''): v for k, v in ckpt.items() 
                        if 'backbone' in k or 'layer' in k or 'conv' in k or 'bn' in k}
        model.backbone.load_state_dict(backbone_dict, strict=False)
        print("Pretrained backbone loaded!")
    
    # 损失函数（带类别权重）
    disease_weight = torch.tensor([train_dataset.disease_weights.get(i, 1.0) 
                               for i in range(NUM_CLASSES_DISEASE)]).to(DEVICE)
    disease_criterion = nn.CrossEntropyLoss(weight=disease_weight)

    # 🔧 添加severity权重（提升General召回率）
    severity_weights = torch.tensor([1.0, 2.5, 1.3], dtype=torch.float32).to(DEVICE)
    severity_criterion = nn.CrossEntropyLoss(weight=severity_weights)
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sum(NUM_EPOCHS_STAGE))
    
    # 混合精度训练加速
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # 日志
    log_path = os.path.join(SAVE_DIR, "training_log.csv")
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow([
            'stage', 'epoch', 'train_loss',
            'val_disease_acc', 'val_disease_f1',
            'val_severity_acc', 'val_severity_f1',
            'val_joint_acc', 'synergy_gain',
            'recall_healthy', 'recall_general', 'recall_serious'
        ])
    
    # 保存每个Stage的结果
    stage_results = {}
    best_joint_f1 = 0
    
    # ==================== 渐进式训练 ====================
    current_epoch = 0
    for stage_idx, num_epochs in enumerate(NUM_EPOCHS_STAGE, 1):
        print(f"\n{'='*60}")
        print(f"Stage {stage_idx}: {num_epochs} epochs")
        print(f"{'='*60}")
        
        # Stage 1: 只训练疾病分类
        if stage_idx == 1:
            for p in model.backbone.parameters():
                p.requires_grad = False
            for p in model.severity_head.parameters():
                p.requires_grad = False
            for p in model.disease_embedding.parameters():
                p.requires_grad = False
            severity_weight = 0.0
            print("Stage 1: Training disease classification only (backbone frozen)")
        
        # Stage 2: 加入严重程度，冻结backbone
        elif stage_idx == 2:
            for p in model.backbone.parameters():
                p.requires_grad = False
            for p in model.severity_head.parameters():
                p.requires_grad = True
            for p in model.disease_embedding.parameters():
                p.requires_grad = True
            severity_weight = 0.6
            # 重建优化器
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                   lr=LR*0.5, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
            print("Stage 2: Training severity head (backbone still frozen)")
        
        # Stage 3: 全局微调
        else:
            for p in model.parameters():
                p.requires_grad = True
            severity_weight = 1.0
            # 分层学习率
            optimizer = optim.AdamW([
                {'params': model.backbone.parameters(), 'lr': LR*0.05},
                {'params': model.disease_head.parameters(), 'lr': LR*0.3},
                {'params': model.disease_embedding.parameters(), 'lr': LR*0.3},
                {'params': model.severity_head.parameters(), 'lr': LR*0.3}
            ], weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
            print("Stage 3: Fine-tuning all layers with layered learning rates")
        
        # 训练循环
        for epoch in range(1, num_epochs + 1):
            current_epoch += 1
            model.train()
            total_loss = 0
            
            pbar = tqdm(train_loader, desc=f"S{stage_idx}E{epoch}/{num_epochs}")
            for imgs, d_labels, s_labels, d_weights, s_weights in pbar:
                imgs = imgs.to(DEVICE, non_blocking=True)
                d_labels = d_labels.to(DEVICE, non_blocking=True)
                s_labels = s_labels.to(DEVICE, non_blocking=True)
                s_weights = s_weights.to(DEVICE, non_blocking=True)
                
                optimizer.zero_grad()
                
                # 混合精度前向传播
                if scaler:
                    with torch.cuda.amp.autocast():
                        d_logits, s_logits = model(imgs)
                        d_loss = disease_criterion(d_logits, d_labels)
                        s_loss = (severity_criterion(s_logits, s_labels) * s_weights).mean()
                        loss = d_loss + severity_weight * s_loss
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    d_logits, s_logits = model(imgs)
                    d_loss = disease_criterion(d_logits, d_labels)
                    s_loss = (severity_criterion(s_logits, s_labels) * s_weights).mean()
                    loss = d_loss + severity_weight * s_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            scheduler.step()
            avg_loss = total_loss / len(train_loader)
            
            # 验证
            metrics = evaluate(model, val_loader, DEVICE)
            
            print(f"\nEpoch {current_epoch} Results:")
            print(f"  Disease    - Acc: {metrics['disease_acc']:.4f}, F1: {metrics['disease_f1']:.4f}")
            print(f"  Severity   - Acc: {metrics['severity_acc']:.4f}, F1: {metrics['severity_f1']:.4f}")
            print(f"  Joint      - Acc: {metrics['joint_acc']:.4f}")
            print(f"  Synergy Gain: {metrics['synergy_gain']:+.4f} ({metrics['synergy_gain']*100:+.2f}%)")
            print(f"  Severity Recalls: {[f'{r:.3f}' for r in metrics['severity_recalls']]}")
            
            # 保存日志
            with open(log_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    stage_idx, epoch, avg_loss,
                    metrics['disease_acc'], metrics['disease_f1'],
                    metrics['severity_acc'], metrics['severity_f1'],
                    metrics['joint_acc'], metrics['synergy_gain'],
                    metrics['severity_recalls'][0],
                    metrics['severity_recalls'][1],
                    metrics['severity_recalls'][2]
                ])
            
            # 保存最佳模型
            joint_f1 = (metrics['disease_f1'] + metrics['severity_f1']) / 2
            if joint_f1 > best_joint_f1:
                best_joint_f1 = joint_f1
                torch.save({
                    'epoch': current_epoch,
                    'stage': stage_idx,
                    'model_state_dict': model.state_dict(),
                    'metrics': metrics
                }, os.path.join(SAVE_DIR, 'best_model.pth'))
                print(f"  ✓ Best model saved (Joint F1: {joint_f1:.4f})")
        
        # Stage结束评估
        print(f"\n{'='*60}")
        print(f"Stage {stage_idx} Complete - Final Evaluation")
        print(f"{'='*60}")
        final_metrics = evaluate(model, val_loader, DEVICE)
        stage_results[f'stage_{stage_idx}'] = final_metrics
        
        # 保存Stage模型
        torch.save({
            'stage': stage_idx,
            'model_state_dict': model.state_dict(),
            'metrics': final_metrics
        }, os.path.join(SAVE_DIR, f'stage_{stage_idx}_model.pth'))
        
        print(f"Disease F1: {final_metrics['disease_f1']:.4f}")
        print(f"Severity F1: {final_metrics['severity_f1']:.4f}")
        print(f"Joint Acc: {final_metrics['joint_acc']:.4f}")
        print(f"Synergy Gain: {final_metrics['synergy_gain']:+.4f}")
    
    # ==================== 最终对比分析 ====================
    print(f"\n{'='*60}")
    print("FINAL COMPARISON ANALYSIS")
    print(f"{'='*60}")
    
    print("\n【Progressive Learning Results】")
    for stage_name, metrics in stage_results.items():
        print(f"\n{stage_name.upper()}:")
        print(f"  Disease F1:    {metrics['disease_f1']:.4f}")
        print(f"  Severity F1:   {metrics['severity_f1']:.4f}")
        print(f"  Joint Acc:     {metrics['joint_acc']:.4f}")
        print(f"  Synergy Gain:  {metrics['synergy_gain']:+.4f} ({metrics['synergy_gain']*100:+.2f}%)")
    
    # 对比分析
    print(f"\n{'='*60}")
    print("SYNERGY EFFECT ANALYSIS")
    print(f"{'='*60}")
    
    final_metrics = stage_results['stage_3']
    print(f"\nFinal Performance:")
    print(f"  Disease Accuracy:     {final_metrics['disease_acc']:.4f} ({final_metrics['disease_acc']*100:.2f}%)")
    print(f"  Severity Accuracy:    {final_metrics['severity_acc']:.4f} ({final_metrics['severity_acc']*100:.2f}%)")
    print(f"  Combined Accuracy:    {final_metrics['joint_acc']:.4f} ({final_metrics['joint_acc']*100:.2f}%)")
    print(f"\nTheoretical Performance (if independent):")
    print(f"  Expected Joint Acc:   {final_metrics['theoretical_joint']:.4f} ({final_metrics['theoretical_joint']*100:.2f}%)")
    print(f"\nSynergy Gain:")
    print(f"  Actual - Theoretical: {final_metrics['synergy_gain']:+.4f} ({final_metrics['synergy_gain']*100:+.2f}%)")
    
    if final_metrics['synergy_gain'] > 0:
        print(f"  ✓ POSITIVE SYNERGY: Multi-task learning improves joint performance!")
    else:
        print(f"  ✗ NEGATIVE SYNERGY: Tasks may be interfering")
    
    # 保存完整分析报告
    report = {
        'stage_results': {k: {key: float(v) if isinstance(v, (int, float, np.number)) else v 
                             for key, v in metrics.items() if key != 'severity_recalls'}
                         for k, metrics in stage_results.items()},
        'final_analysis': {
            'disease_accuracy': float(final_metrics['disease_acc']),
            'severity_accuracy': float(final_metrics['severity_acc']),
            'joint_accuracy': float(final_metrics['joint_acc']),
            'theoretical_joint': float(final_metrics['theoretical_joint']),
            'synergy_gain': float(final_metrics['synergy_gain']),
            'synergy_percentage': float(final_metrics['synergy_gain'] * 100),
            'interpretation': 'Positive Synergy' if final_metrics['synergy_gain'] > 0 else 'Negative Synergy'
        }
    }
    
    with open(os.path.join(SAVE_DIR, 'analysis_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"✓ Logs saved to: {log_path}")
    print(f"✓ Analysis report: {os.path.join(SAVE_DIR, 'analysis_report.json')}")
    print(f"✓ Best model: {os.path.join(SAVE_DIR, 'best_model.pth')}")

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Stages: {NUM_EPOCHS_STAGE} (Total: {sum(NUM_EPOCHS_STAGE)} epochs)")
    train_progressive()