

import os
import json
import random
import csv
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm
import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import f1_score, recall_score, accuracy_score, classification_report
import torch.utils.data as data


TRAIN_JSON = r"C:\Users\21978\Desktop\大学\赛程知识\数模赛\数维杯\Question1\Tclean_labels.json"
VAL_JSON   = r"C:\Users\21978\Desktop\大学\赛程知识\数模赛\数维杯\Question1\Yclean_labels.json"
TRAIN_IMG_DIR =  "images-31718"
VAL_IMG_DIR   = "images"
RESNET50_PRETRAIN = r"C:\Users\21978\Desktop\大学\赛程知识\数模赛\数维杯\Question1\checkpoints\resnet50_best.pth"
SAVE_DIR = "./checkpoints_task3_v3"
os.makedirs(SAVE_DIR, exist_ok=True)

LOG_PATH = os.path.join(SAVE_DIR, "train_task3_v3.log")
METRICS_CSV = os.path.join(SAVE_DIR, "metrics_task3_v3.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 15
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42
NUM_WORKERS = 4

STRONG_AUG_UPPER = 200
WEAK_AUG_LOWER = 800


# logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

label_to_species = {
    0:'Apple',1:'Apple',2:'Apple',3:'Apple',4:'Apple',5:'Apple',
    6:'Cherry',7:'Cherry',8:'Cherry',
    9:'Corn',10:'Corn',11:'Corn',12:'Corn',13:'Corn',14:'Corn',15:'Corn',16:'Corn',
    17:'Grape',18:'Grape',19:'Grape',20:'Grape',21:'Grape',22:'Grape',23:'Grape',
    24:'Citrus',25:'Citrus',26:'Citrus',
    27:'Peach',28:'Peach',29:'Peach',
    30:'Pepper',31:'Pepper',32:'Pepper',
    33:'Potato',34:'Potato',35:'Potato',36:'Potato',37:'Potato',
    38:'Strawberry',39:'Strawberry',40:'Strawberry',
    41:'Tomato',42:'Tomato',43:'Tomato',44:'Tomato',45:'Tomato',46:'Tomato',47:'Tomato',
    48:'Tomato',49:'Tomato',50:'Tomato',51:'Tomato',52:'Tomato',53:'Tomato',54:'Tomato',
    55:'Tomato',56:'Tomato',57:'Tomato',58:'Tomato',59:'Tomato',60:'Tomato'
}

healthy_ids = [0, 6, 9, 17, 24, 27, 30, 33, 38, 41]
serious_ids = [2,5,8,11,13,15,16,19,21,23,26,29,32,35,37,40,43,45,47,49,51,53,55,57,59,60]

def label_to_severity(label_id: int) -> int:
    if label_id in healthy_ids:
        return 0
    elif label_id in serious_ids:
        return 2
    else:
        return 1


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()


def load_annotations(json_path, img_dir):
    with open(json_path, 'r', encoding='utf-8') as f:
        ann = json.load(f)
    items = []
    for item in ann:
        lid = int(item['disease_class'])
        img_id = item['image_id']
        img_path = os.path.join(img_dir, img_id)
        species = label_to_species.get(lid, "Unknown")
        severity = label_to_severity(lid)
        items.append((img_path, species, severity, lid))
    return items

class SeverityDataset(data.Dataset):
    def __init__(self, items, sp_sev_counts, spsev_weights, mode='train'):
        self.items = items
        self.sp_sev_counts = sp_sev_counts
        self.spsev_weights = spsev_weights 
        self.mode = mode

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, species, severity, lid = self.items[idx]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert('RGB')

        if self.mode == 'train':
            count = self.sp_sev_counts.get((species, severity), 0)
            if count <= STRONG_AUG_UPPER:
                img = strong_aug(img)
            elif count >= WEAK_AUG_LOWER:
                img = weak_aug(img)
            else:
                img = medium_aug(img)
        else:
            img = transforms.Resize(int(IMG_SIZE*1.15))(img)
            img = transforms.CenterCrop(IMG_SIZE)(img)

        img = to_tensor_norm(img)
        sample_weight = self.spsev_weights.get(species, {0:1.0,1:1.0,2:1.0}).get(severity, 1.0)
        return img, severity, sample_weight, species

class SeverityModel(nn.Module):
    def __init__(self, backbone_pretrain_path=None, freeze_backbone=True):
        super().__init__()
        backbone = models.resnet50(weights=None)
        if backbone_pretrain_path is not None:
            state = torch.load(backbone_pretrain_path, map_location='cpu')
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            state = {k: v for k, v in state.items() if not k.startswith('fc.')}
            backbone.load_state_dict(state, strict=False)
            logger.info(f"Loaded backbone weights (fc excluded) from {backbone_pretrain_path}")

        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.adapter = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )
        self.head = nn.Linear(512, 3)

    def forward(self, x):
        feat = self.backbone(x)
        x = self.adapter(feat)
        logits = self.head(x)
        return logits

def adjust_weights(spsev_weights, val_rec, threshold=0.75, factor=1.05, max_weight=3.0):
    for cls_id, rec in enumerate(val_rec):
        if rec < threshold:
            for sp in spsev_weights.keys():
                new_w = spsev_weights[sp][cls_id] * factor
                spsev_weights[sp][cls_id] = min(new_w, max_weight)
    return spsev_weights
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    rec_per_class = recall_score(y_true, y_pred, average=None, labels=[0,1,2])
    return acc, macro_f1, rec_per_class

weak_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9,1.0)),
    transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08)
])
medium_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8,1.0)),
    transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12)
])
strong_aug = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7,1.0)),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
])
to_tensor_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


def train_task3():
    logger.info("Starting train_task3_v3")

    train_items = load_annotations(TRAIN_JSON, TRAIN_IMG_DIR)
    val_items   = load_annotations(VAL_JSON, VAL_IMG_DIR)
    logger.info(f"Loaded {len(train_items)} train samples, {len(val_items)} val samples")

    species_severity_counts = defaultdict(Counter)
    for _p, sp, sev, _lid in train_items:
        species_severity_counts[sp][sev] += 1
    sp_sev_counts = {}
    for sp, cnts in species_severity_counts.items():
        for sev in [0,1,2]:
            sp_sev_counts[(sp, sev)] = cnts.get(sev, 0)

    # compute initial inverse-frequency weights
    eps = 1e-6
    spsev_weights = {}
    for sp, cnts in species_severity_counts.items():
        arr = np.array([cnts.get(0,0), cnts.get(1,0), cnts.get(2,0)], dtype=float) + eps
        inv = 1.0 / arr
        inv = inv / np.mean(inv)
        spsev_weights[sp] = {0: float(inv[0]), 1: float(inv[1]), 2: float(inv[2])}


    train_dataset = SeverityDataset(train_items, sp_sev_counts, spsev_weights, mode='train')
    val_dataset   = SeverityDataset(val_items, sp_sev_counts, spsev_weights, mode='val')
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = SeverityModel(backbone_pretrain_path=RESNET50_PRETRAIN, freeze_backbone=True)
    model = model.to(DEVICE)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(reduction='none')


    with open(METRICS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch","train_loss","train_acc","train_macro_f1","val_loss","val_acc","val_macro_f1","rec_healthy","rec_general","rec_serious","time_elapsed"])

    best_macro_f1 = -1.0
    best_model_path = os.path.join(SAVE_DIR, "best_model_task3_v3.pth")

 
    # Training loop

    for epoch in range(1, EPOCHS+1):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        total_samples = 0
        all_preds, all_labels = [], []

        for imgs, labels, sample_weights, species in tqdm(train_loader, desc=f"Train {epoch}/{EPOCHS}"):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            sample_weights = sample_weights.to(DEVICE).float()

            logits = model(imgs)
            loss_per_sample = criterion(logits, labels)
            weighted_loss = (loss_per_sample * sample_weights).mean()

            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            running_loss += weighted_loss.item() * imgs.size(0)
            total_samples += imgs.size(0)
            preds = torch.argmax(logits.detach(), dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        train_acc, train_macro_f1, train_rec = compute_metrics(np.array(all_labels), np.array(all_preds))
        avg_train_loss = running_loss / (total_samples + 1e-9)
        logger.info(f"[Epoch {epoch}] Train Loss={avg_train_loss:.4f} Acc={train_acc:.4f} MacroF1={train_macro_f1:.4f} Recalls={train_rec}")

 
        # Validation

        model.eval()
        val_losses, val_total = 0.0, 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for imgs, labels, sample_weights, species in tqdm(val_loader, desc="Validation"):
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                logits = model(imgs)
                loss_val = criterion(logits, labels).mean()  # ← 验证不加权
                val_losses += loss_val.item() * imgs.size(0)
                val_total += imgs.size(0)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds.tolist())
                val_labels.extend(labels.cpu().numpy().tolist())

        val_acc, val_macro_f1, val_rec = compute_metrics(np.array(val_labels), np.array(val_preds))
        avg_val_loss = val_losses / (val_total + 1e-9)
        logger.info(f"[Epoch {epoch}] Val Loss={avg_val_loss:.4f} Acc={val_acc:.4f} MacroF1={val_macro_f1:.4f} Recalls={val_rec}")
        logger.info("Validation classification report:\n" + classification_report(val_labels, val_preds, digits=4, target_names=['Healthy','General','Serious']))

 
        # Dynamic weight adjustment 
        if epoch <= 5:
            spsev_weights = adjust_weights(spsev_weights, val_rec, threshold=0.75, factor=1.05, max_weight=3.0)
            train_dataset.spsev_weights = spsev_weights
            logger.info(f"Adjusted weights: General mean={np.mean([w[1] for w in spsev_weights.values()]):.4f}")

        # CSV logging
        elapsed = time.time() - start_time
        with open(METRICS_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train_loss, train_acc, train_macro_f1, avg_val_loss, val_acc, val_macro_f1, val_rec[0], val_rec[1], val_rec[2], round(elapsed,2)])

        # 保存最佳模型
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_macro_f1': best_macro_f1
            }, best_model_path)
            logger.info(f"Saved new best model to {best_model_path} (macroF1={best_macro_f1:.4f})")

        scheduler.step()

    logger.info("Training finished.")
    logger.info(f"Best val macro-F1: {best_macro_f1:.4f}, saved at: {best_model_path}")


if __name__ == "__main__":
    train_task3()
