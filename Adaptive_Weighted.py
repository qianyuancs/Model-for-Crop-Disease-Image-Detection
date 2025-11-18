
import matplotlib
matplotlib.use('Agg')
import os
import json
import numpy as np
from collections import Counter
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False

# 步骤1：分析JSON并分层类别

def analyze_json_and_categorize(json_path, save_analysis=True):

    print("=" * 60)
    print("开始分析数据集...")
    print("=" * 60)
    
    # 1. 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 2. 提取所有disease_class标签
    labels = [item['disease_class'] for item in data]
    class_counts = Counter(labels)
    
    # 3. 计算分位数（Q1=25%, Q3=75%）
    counts_array = np.array(list(class_counts.values()))
    q1, q3 = np.percentile(counts_array, [25, 75])
    mean_count = np.mean(counts_array)
    
    # 4. 按样本数量分成三档
    small_classes = []   # 小样本类（稀有病害）
    medium_classes = []  # 中样本类
    large_classes = []   # 大样本类（常见病害）
    
    for disease_class, count in class_counts.items():
        if count < q1:
            small_classes.append(disease_class)
        elif count < q3:
            medium_classes.append(disease_class)
        else:
            large_classes.append(disease_class)
    
    # 5. 打印分析结果
    print(f"\n数据集基本信息:")
    print(f"  - 总类别数: {len(class_counts)}")
    print(f"  - 总样本数: {sum(class_counts.values())}")
    print(f"  - 平均每类: {mean_count:.1f} 张")
    print(f"  - 最小类别: {min(class_counts.values())} 张")
    print(f"  - 最大类别: {max(class_counts.values())} 张")
    
    print(f"\n类别分层结果:")
    print(f"  小样本类 (< {q1:.0f}张): {len(small_classes)} 个类别")
    print(f"     → 将使用【强增强】策略")
    print(f"  中样本类 ({q1:.0f}-{q3:.0f}张): {len(medium_classes)} 个类别")
    print(f"     → 将使用【中等增强】策略")
    print(f"  大样本类 (>= {q3:.0f}张): {len(large_classes)} 个类别")
    print(f"     → 将使用【弱增强】策略")
    
    # 6. 显示具体哪些类别属于哪一档
    print(f"\n小样本类详情（需要强增强）:")
    for cls in sorted(small_classes)[:10]:  # 只显示前10个
        print(f"  - 类别 {cls}: {class_counts[cls]} 张")
    if len(small_classes) > 10:
        print(f"  ... 还有 {len(small_classes)-10} 个类别")
    
    # 7. 保存分析结果
    class_groups = {
        'small': small_classes,
        'medium': medium_classes,
        'large': large_classes
    }
    
    if save_analysis:
        analysis_result = {
            'class_counts': dict(class_counts),
            'class_groups': class_groups,
            'statistics': {
                'total_classes': len(class_counts),
                'total_samples': sum(class_counts.values()),
                'q1': float(q1),
                'q3': float(q3),
                'mean': float(mean_count)
            }
        }
        
        output_path = json_path.replace('.json', '_analysis.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)
        print(f"\n 分析结果已保存到: {output_path}")
    
    print("=" * 60)
    
    return class_counts, class_groups



# 步骤2：定义自适应增强策略


class AdaptiveAugmentation:
    """
    根据类别样本数量，自动选择不同强度的增强策略
    """
    
    def __init__(self, class_groups, img_size=456):
        """
        参数:
            class_groups: 从analyze_json_and_categorize()返回的分组结果
            img_size: 图像尺寸（EfficientNetB5推荐456）
        """
        self.class_groups = class_groups
        self.img_size = img_size
        
        # 强增强策略（用于小样本类）
        self.strong_aug = A.Compose([
            # 几何变换
            A.RandomResizedCrop(img_size, img_size, scale=(0.7, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=45, p=0.7),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
            
            # 颜色增强（模拟不同光照/病害阶段）
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            
            # 噪声和模糊（模拟不同拍摄条件）
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
            ], p=0.3),
            
            # 区域遮挡（增强鲁棒性）
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, 
                           fill_value=0, p=0.3),
            
            # 标准化
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # 中等增强策略用于中样本类
        self.medium_aug = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.4),
            A.GaussNoise(var_limit=(10, 30), p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # 弱增强策略（用于大样本类）
        self.weak_aug = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # 验证集变换（无增强）
        self.val_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def get_transform(self, disease_class, is_training=True):

        if not is_training:
            return self.val_transform
        
        # 根据类别选择增强策略
        if disease_class in self.class_groups['small']:
            return self.strong_aug
        elif disease_class in self.class_groups['medium']:
            return self.medium_aug
        else:
            return self.weak_aug



# 步骤3：创建自适应Dataset


class AdaptiveAugDataset(Dataset):

    
    def __init__(self, json_path, img_dir, class_groups, is_training=True, img_size=456):

        # 读取JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.img_dir = img_dir
        self.is_training = is_training
        
        # 初始化增强管道
        self.aug_pipeline = AdaptiveAugmentation(class_groups, img_size)
        
        print(f" Dataset初始化完成:")
        print(f"   - 样本数量: {len(self.data)}")
        print(f"   - 模式: {'训练' if is_training else '验证'}")
        print(f"   - 图像尺寸: {img_size}x{img_size}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        读取一张图像并应用自适应增强
        """
        # 1. 获取JSON中的信息
        item = self.data[idx]
        image_id = item['image_id']
        disease_class = item['disease_class']
        
        # 2. 读取图像
        img_path = os.path.join(self.img_dir, image_id)
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            print(f" 读取图像失败: {img_path}, 错误: {e}")
            # 返回黑色图像作为占位符
            image = np.zeros((456, 456, 3), dtype=np.uint8)
        
        # 3. 根据disease_class选择增强策略
        transform = self.aug_pipeline.get_transform(disease_class, self.is_training)
        
        # 4. 应用增强
        augmented = transform(image=image)
        image_tensor = augmented['image']
        
        return image_tensor, disease_class


# 步骤4：创建加权采样器


def create_weighted_sampler(class_counts, json_path):

    # 读取JSON获取所有标签
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    labels = [item['disease_class'] for item in data]
    
    # 计算每个类别的权重（样本数越少权重越高）
    total_samples = sum(class_counts.values())
    class_weights = {disease_class: total_samples / count 
                     for disease_class, count in class_counts.items()}
    
    # 为每个样本赋权重
    sample_weights = [class_weights[label] for label in labels]
    
    # 创建采样器
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # 允许重复采样
    )
    
    print(" 加权采样器创建完成")
    print(f"   - 小样本类将被更频繁采样")
    
    return sampler



# 步骤5：可视化增强效果


def visualize_augmentation(json_path, img_dir, class_groups, disease_class=None, num_samples=8):

    # 如果未指定类别，随机选一个小样本类
    if disease_class is None:
        disease_class = np.random.choice(class_groups['small'])
    
    # 读取该类别的一张图像
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 找到该类别的第一张图像
    target_item = None
    for item in data:
        if item['disease_class'] == disease_class:
            target_item = item
            break
    
    if target_item is None:
        print(f" 未找到类别 {disease_class} 的图像")
        return
    
    # 读取原始图像
    img_path = os.path.join(img_dir, target_item['image_id'])
    original_img = Image.open(img_path).convert('RGB')
    original_array = np.array(original_img)
    
    # 确定增强策略
    aug_pipeline = AdaptiveAugmentation(class_groups)
    if disease_class in class_groups['small']:
        transform = aug_pipeline.strong_aug
        aug_type = "强增强"
    elif disease_class in class_groups['medium']:
        transform = aug_pipeline.medium_aug
        aug_type = "中等增强"
    else:
        transform = aug_pipeline.weak_aug
        aug_type = "弱增强"
    
    # 生成多个增强版本
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(f'类别 {disease_class} - {aug_type}策略', fontsize=16, fontweight='bold')
    
    # 显示原始图像
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('原始图像', fontsize=12)
    axes[0, 0].axis('off')
    
    # 显示增强后的图像
    for i in range(1, 9):
        # 应用增强（去掉Normalize和ToTensor以便显示）
        temp_transform = A.Compose([t for t in transform.transforms 
                                   if not isinstance(t, (A.Normalize, ToTensorV2))])
        augmented = temp_transform(image=original_array)['image']
        
        ax = axes[i // 3, i % 3]
        ax.imshow(augmented)
        ax.set_title(f'增强 #{i}', fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    save_path = f'augmentation_visualization_class{disease_class}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f" 可视化结果已保存到: {save_path}")
    plt.show()




def prepare_adaptive_augmentation(train_json, train_img_dir, 
                                  val_json=None, val_img_dir=None,
                                  batch_size=32, num_workers=4, img_size=456):

    print("\n" + "="*60)
    print("开始自适应增强前置处理")
    print("="*60 + "\n")
    
    # 步骤1：分析数据集
    class_counts, class_groups = analyze_json_and_categorize(train_json)
    
    # 步骤2：创建训练集Dataset
    print("\n创建训练集Dataset...")
    train_dataset = AdaptiveAugDataset(
        json_path=train_json,
        img_dir=train_img_dir,
        class_groups=class_groups,
        is_training=True,
        img_size=img_size
    )
    
    # 步骤3：创建加权采样器
    print("\n创建加权采样器...")
    sampler = create_weighted_sampler(class_counts, train_json)
    
    # 步骤4：创建训练集DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # 使用加权采样器
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f" 训练集DataLoader创建完成")
    print(f"   - Batch size: {batch_size}")
    print(f"   - 总batch数: {len(train_loader)}")
    
    # 步骤5：创建验证集（如果提供）
    val_loader = None
    if val_json is not None and val_img_dir is not None:
        print("\n创建验证集Dataset...")
        val_dataset = AdaptiveAugDataset(
            json_path=val_json,
            img_dir=val_img_dir,
            class_groups=class_groups,
            is_training=False,  # 验证集不做增强
            img_size=img_size
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f" 验证集DataLoader创建完成")
        print(f"   - 样本数量: {len(val_dataset)}")
        print(f"   - 总batch数: {len(val_loader)}")
    
    print("\n" + "="*60)
    print(" 所有前置处理完成！可以直接喂给模型训练")
    print("="*60 + "\n")
    
    return train_loader, val_loader, class_groups


if __name__ == "__main__":


#修改参数看这里！！！！！

    TRAIN_JSON = r"C:\Users\21978\Desktop\大学\赛程知识\数模赛\数维杯\Question1\Tclean_labels.json" # 训练集JSON
    TRAIN_IMG_DIR = "images-31718"   # 你们的训练集图像文件夹
    
    VAL_JSON = r"C:\Users\21978\Desktop\大学\赛程知识\数模赛\数维杯\Question1\Yclean_labels.json"     # 验证集JSON
    VAL_IMG_DIR = "images"       # 验证集图像文件夹

    train_loader, val_loader, class_groups = prepare_adaptive_augmentation(
        train_json=TRAIN_JSON,
        train_img_dir=TRAIN_IMG_DIR,
        val_json=VAL_JSON,
        val_img_dir=VAL_IMG_DIR,
        batch_size=32,
        num_workers=4,
        img_size=224  
    )
    

    print("model = models.efficientnet_b5(pretrained=True)")
    print("model.classifier[1] = nn.Linear(model.classifier[1].in_features, 61)")
    print("model = model.to('cuda')")
    print("")
    print("# 训练循环")
    print("for epoch in range(epochs):")
    print("    for images, labels in train_loader:")
    print("        images, labels = images.to('cuda'), labels.to('cuda')")
    print("-" * 60)

    visualize_augmentation(
         json_path=TRAIN_JSON,
         img_dir=TRAIN_IMG_DIR,
         class_groups=class_groups,
         disease_class=None,  # 自动选一个小样本类
         num_samples=8
     )