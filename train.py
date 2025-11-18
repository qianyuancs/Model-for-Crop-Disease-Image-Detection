import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

# 导入自适应增强模块
from Adaptive_Weighted import prepare_adaptive_augmentation, visualize_augmentation

# ==============================
# 配置参数
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 61
batch_size = 32
num_epochs = 20
learning_rate = 1e-3
img_size = 224
num_workers = 4

TRAIN_JSON = r"C:\Users\21978\Desktop\大学\赛程知识\数模赛\数维杯\Question1\Tclean_labels.json"
TRAIN_IMG_DIR = "images-31718"
VAL_JSON = r"C:\Users\21978\Desktop\大学\赛程知识\数模赛\数维杯\Question1\Yclean_labels.json"
VAL_IMG_DIR = "images"

SAVE_DIR = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

if __name__ == "__main__":

    # 数据加载
    train_loader, val_loader, class_groups = prepare_adaptive_augmentation(
        train_json=TRAIN_JSON,
        train_img_dir=TRAIN_IMG_DIR,
        val_json=VAL_JSON,
        val_img_dir=VAL_IMG_DIR,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size
    )

    # 模型
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # 优化设置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")

        # ------------------ 验证 ------------------
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                    images, labels = images.to(device), labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    correct_val += torch.sum(preds == labels).item()
                    total_val += labels.size(0)

            val_loss /= total_val
            val_acc = correct_val / total_val
            print(f"Epoch {epoch+1}: Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = os.path.join(SAVE_DIR, "resnet50_best.pth")
                torch.save(model.state_dict(), save_path)
                print(f"  → 保存最佳模型: {save_path}")

        scheduler.step()

    # 可视化
    visualize_augmentation(
        json_path=TRAIN_JSON,
        img_dir=TRAIN_IMG_DIR,
        class_groups=class_groups,
        disease_class=None,
        num_samples=8
    )

    print("训练完成，最佳验证准确率:", best_val_acc)
