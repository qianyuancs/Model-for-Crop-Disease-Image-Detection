import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

# ==================== 配置 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 32

# 数据路径（改成你的实际路径）
VAL_JSON = r"C:\Users\21978\Desktop\大学\赛程知识\数模赛\数维杯\Question1\Yrelabelled_dataset.json"
VAL_IMG_DIR = "images"
MODEL_PATH = r"C:\Users\21978\Desktop\大学\赛程知识\数模赛\数维杯\Question1\Q4_2_evaluation\best_model(2).pth"  # 你训练好的模型
SAVE_DIR = "Q4_2_evaluation_results_final"
os.makedirs(SAVE_DIR, exist_ok=True)

# 疾病名称映射（完整版）
label_id_to_name_en = {
    0: "Apple Healthy", 1: "Apple Scab (General)", 2: "Apple Scab (Serious)", 
    3: "Apple Frogeye Spot", 4: "Cedar Apple Rust (General)", 5: "Cedar Apple Rust (Serious)", 
    6: "Cherry Healthy", 7: "Cherry Powdery Mildew (General)", 8: "Cherry Powdery Mildew (Serious)", 
    9: "Corn Healthy", 10: "Cercospora (General)", 11: "Cercospora (Serious)", 
    12: "Corn Puccinia (General)", 13: "Corn Puccinia (Serious)", 
    14: "Corn Curvularia (General)", 15: "Corn Curvularia (Serious)",
    16: "Maize Dwarf Mosaic Virus", 17: "Grape Healthy", 18: "Grape Black Rot (General)",
    19: "Grape Black Rot (Serious)", 20: "Grape Black Measles (General)", 
    21: "Grape Black Measles (Serious)", 22: "Grape Leaf Blight (General)", 
    23: "Grape Leaf Blight (Serious)", 24: "Citrus Healthy", 25: "Citrus Greening (General)", 
    26: "Citrus Greening (Serious)", 27: "Peach Healthy", 28: "Peach Bacterial Spot (General)", 
    29: "Peach Bacterial Spot (Serious)", 30: "Pepper Healthy", 31: "Pepper Scab (General)", 
    32: "Pepper Scab (Serious)", 33: "Potato Healthy", 34: "Potato Early Blight (General)", 
    35: "Potato Early Blight (Serious)", 36: "Potato Late Blight (General)", 
    37: "Potato Late Blight (Serious)", 38: "Strawberry Healthy", 39: "Strawberry Scorch (General)", 
    40: "Strawberry Scorch (Serious)", 41: "Tomato Healthy", 42: "Tomato Powdery Mildew (General)", 
    43: "Tomato Powdery Mildew (Serious)", 44: "Tomato Bacterial Spot (General)", 
    45: "Tomato Bacterial Spot (Serious)", 46: "Tomato Early Blight (General)", 
    47: "Tomato Early Blight (Serious)", 48: "Tomato Late Blight (General)", 
    49: "Tomato Late Blight (Serious)", 50: "Tomato Leaf Mold (General)", 
    51: "Tomato Leaf Mold (Serious)", 52: "Tomato Target Spot (General)", 
    53: "Tomato Target Spot (Serious)", 54: "Tomato Septoria (General)", 
    55: "Tomato Septoria (Serious)", 56: "Tomato Spider Mite (General)", 
    57: "Tomato Spider Mite (Serious)", 58: "Tomato Yellow Leaf Curl Virus (General)", 
    59: "Tomato Yellow Leaf Curl Virus (Serious)", 60: "Tomato Mosaic Virus"
}

severity_names = ["Healthy", "General Disease", "Serious Disease"]

# 防治建议
def get_advice(disease_name, severity_name):
    if "Healthy" in disease_name or severity_name == "Healthy":
        return "作物健康，无需防治。建议定期监测，保持良好的田间管理。"
    elif severity_name == "Serious Disease":
        return f"检测到严重{disease_name}！建议：1) 立即喷施针对性高浓度农药；2) 隔离或清除严重病株；3) 加强通风，降低湿度；4) 咨询当地农技专家。"
    elif severity_name == "General Disease":
        return f"检测到轻度{disease_name}。建议：1) 喷施低浓度针对性农药；2) 摘除病叶；3) 加强田间管理；4) 密切监测病情发展。"
    else:
        return "保持常规管理，加强田间监测。"

# ==================== Dataset ====================
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class EvalDataset(Dataset):
    def __init__(self, json_file, img_dir, transform):
        self.img_dir = img_dir
        self.transform = transform
        with open(json_file, 'r', encoding='utf-8') as f:
            self.items = json.load(f)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = os.path.join(self.img_dir, item['image_id'])
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, int(item['disease_main_id']), int(item['severity_id']), \
               item['image_id'], item['disease_main_name']

# ==================== 模型（必须和训练时一致！）====================
class ProgressiveMultiTaskModel(nn.Module):
    """和训练代码完全一致的模型"""
    def __init__(self, num_disease=61, num_severity=3, embedding_dim=128):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Identity()
        
        self.disease_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_disease)
        )
        
        self.disease_embedding = nn.Embedding(num_disease, embedding_dim)
        
        self.severity_head = nn.Sequential(
            nn.Linear(2048 + embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_severity)
        )

    def forward(self, x):
        feat = self.backbone(x)
        disease_logits = self.disease_head(feat)
        disease_pred = torch.argmax(disease_logits, dim=1)
        disease_emb = self.disease_embedding(disease_pred)
        cond_feat = torch.cat([feat, disease_emb], dim=1)
        severity_logits = self.severity_head(cond_feat)
        return disease_logits, severity_logits

# ==================== 评估函数 ====================
def comprehensive_evaluation(model, dataloader, device):
    model.eval()
    
    all_d_preds, all_d_labels, all_d_probs = [], [], []
    all_s_preds, all_s_labels, all_s_probs = [], [], []
    all_image_ids, all_disease_names = [], []
    
    print("Running evaluation...")
    with torch.no_grad():
        for imgs, d_labels, s_labels, image_ids, disease_names in tqdm(dataloader):
            imgs = imgs.to(device)
            d_logits, s_logits = model(imgs)
            
            d_probs = F.softmax(d_logits, dim=1)
            s_probs = F.softmax(s_logits, dim=1)
            
            d_pred = torch.argmax(d_probs, dim=1)
            s_pred = torch.argmax(s_probs, dim=1)
            
            all_d_preds.extend(d_pred.cpu().numpy())
            all_d_labels.extend(d_labels.numpy())
            all_d_probs.extend(d_probs.cpu().numpy())
            
            all_s_preds.extend(s_pred.cpu().numpy())
            all_s_labels.extend(s_labels.numpy())
            all_s_probs.extend(s_probs.cpu().numpy())
            
            all_image_ids.extend(image_ids)
            all_disease_names.extend(disease_names)
    
    all_d_preds = np.array(all_d_preds)
    all_d_labels = np.array(all_d_labels)
    all_d_probs = np.array(all_d_probs)
    all_s_preds = np.array(all_s_preds)
    all_s_labels = np.array(all_s_labels)
    all_s_probs = np.array(all_s_probs)
    
    # 计算指标
    disease_acc = accuracy_score(all_d_labels, all_d_preds)
    disease_f1 = f1_score(all_d_labels, all_d_preds, average='macro', zero_division=0)
    
    severity_acc = accuracy_score(all_s_labels, all_s_preds)
    severity_f1 = f1_score(all_s_labels, all_s_preds, average='macro', zero_division=0)
    
    # Severity召回率
    severity_recalls = []
    for i in range(3):
        mask = (all_s_labels == i)
        if mask.sum() > 0:
            severity_recalls.append(np.mean(all_s_preds[mask] == all_s_labels[mask]))
        else:
            severity_recalls.append(0.0)
    
    # 联合指标
    joint_correct = np.sum((all_d_preds == all_d_labels) & (all_s_preds == all_s_labels))
    joint_acc = joint_correct / len(all_d_labels)
    
    # 协同增益
    theoretical_joint = disease_acc * severity_acc
    synergy_gain = joint_acc - theoretical_joint
    
    # 错误分析
    disease_only_wrong = np.sum((all_d_preds != all_d_labels) & (all_s_preds == all_s_labels))
    severity_only_wrong = np.sum((all_d_preds == all_d_labels) & (all_s_preds != all_s_labels))
    both_wrong = np.sum((all_d_preds != all_d_labels) & (all_s_preds != all_s_labels))
    
    # 混淆矩阵
    severity_cm = confusion_matrix(all_s_labels, all_s_preds, labels=[0, 1, 2])
    
    return {
        'disease_accuracy': float(disease_acc),
        'disease_f1': float(disease_f1),
        'disease_avg_confidence': float(np.mean(np.max(all_d_probs, axis=1))),
        'severity_accuracy': float(severity_acc),
        'severity_f1': float(severity_f1),
        'severity_recalls': [float(r) for r in severity_recalls],
        'severity_avg_confidence': float(np.mean(np.max(all_s_probs, axis=1))),
        'severity_confusion_matrix': severity_cm.tolist(),
        'joint_accuracy': float(joint_acc),
        'joint_f1': float((disease_f1 + severity_f1) / 2),
        'theoretical_joint': float(theoretical_joint),
        'synergy_gain': float(synergy_gain),
        'error_analysis': {
            'disease_only_wrong': int(disease_only_wrong),
            'severity_only_wrong': int(severity_only_wrong),
            'both_wrong': int(both_wrong),
            'both_correct': int(joint_correct)
        },
        'predictions': {
            'disease_preds': all_d_preds.tolist(),
            'disease_labels': all_d_labels.tolist(),
            'disease_probs': all_d_probs.tolist(),
            'severity_preds': all_s_preds.tolist(),
            'severity_labels': all_s_labels.tolist(),
            'severity_probs': all_s_probs.tolist(),
            'image_ids': all_image_ids,
            'disease_names': all_disease_names
        }
    }

# ==================== 生成诊断报告 ====================
def generate_diagnostic_reports(results, num_samples=None):
    predictions = results['predictions']
    
    if num_samples is None:
        num_samples = len(predictions['image_ids'])
    
    reports = []
    print(f"\nGenerating {num_samples} diagnostic reports...")
    
    for i in tqdm(range(num_samples)):
        d_pred = predictions['disease_preds'][i]
        d_probs = np.array(predictions['disease_probs'][i])
        s_pred = predictions['severity_preds'][i]
        s_probs = np.array(predictions['severity_probs'][i])
        
        # Top-3疾病预测
        top3_indices = np.argsort(d_probs)[-3:][::-1]
        top3_predictions = [
            {
                'disease_name': label_id_to_name_en.get(int(idx), f"Disease_{idx}"),
                'confidence': f"{d_probs[idx] * 100:.2f}%"
            }
            for idx in top3_indices
        ]
        
        pred_disease_name = label_id_to_name_en.get(d_pred, f"Disease_{d_pred}")
        pred_severity_name = severity_names[s_pred]
        
        disease_conf = float(d_probs[d_pred])
        severity_conf = float(s_probs[s_pred])
        
        # 风险等级
        if pred_severity_name == "Serious Disease":
            risk_level = "HIGH"
        elif pred_severity_name == "General Disease":
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # 可靠性
        if disease_conf > 0.9 and severity_conf > 0.9:
            reliability = "Very High"
        elif disease_conf > 0.7 and severity_conf > 0.7:
            reliability = "High"
        elif disease_conf > 0.5 and severity_conf > 0.5:
            reliability = "Medium"
        else:
            reliability = "Low"
        
        report = {
            'image_id': predictions['image_ids'][i],
            'diagnosis': {
                'predicted_disease': pred_disease_name,
                'predicted_disease_id': int(d_pred),
                'disease_confidence': f"{disease_conf * 100:.2f}%",
                'predicted_severity': pred_severity_name,
                'severity_confidence': f"{severity_conf * 100:.2f}%",
                'combined_confidence': f"{disease_conf * severity_conf * 100:.2f}%",
                'risk_level': risk_level,
                'reliability': reliability
            },
            'top3_disease_predictions': top3_predictions,
            'severity_distribution': {
                'healthy_prob': f"{s_probs[0] * 100:.2f}%",
                'general_prob': f"{s_probs[1] * 100:.2f}%",
                'serious_prob': f"{s_probs[2] * 100:.2f}%"
            },
            'recommendations': get_advice(pred_disease_name, pred_severity_name),
            'ground_truth': {
                'true_disease': predictions['disease_names'][i],
                'true_severity': severity_names[predictions['severity_labels'][i]],
                'disease_correct': bool(d_pred == predictions['disease_labels'][i]),
                'severity_correct': bool(s_pred == predictions['severity_labels'][i]),
                'both_correct': bool(d_pred == predictions['disease_labels'][i] and 
                                    s_pred == predictions['severity_labels'][i])
            }
        }
        
        reports.append(report)
    
    return reports

# ==================== 打印结果 ====================
def print_results(results):
    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*70)
    
    print("\n【Individual Task Performance】")
    print(f"  Disease Classification:")
    print(f"    - Accuracy:        {results['disease_accuracy']:.4f} ({results['disease_accuracy']*100:.2f}%)")
    print(f"    - Macro F1 Score:  {results['disease_f1']:.4f}")
    print(f"    - Avg Confidence:  {results['disease_avg_confidence']:.4f}")
    
    print(f"\n  Severity Grading:")
    print(f"    - Accuracy:        {results['severity_accuracy']:.4f} ({results['severity_accuracy']*100:.2f}%)")
    print(f"    - Macro F1 Score:  {results['severity_f1']:.4f}")
    print(f"    - Recall (Healthy):  {results['severity_recalls'][0]:.4f}")
    print(f"    - Recall (General):  {results['severity_recalls'][1]:.4f}")
    print(f"    - Recall (Serious):  {results['severity_recalls'][2]:.4f}")
    
    print("\n【Joint Task Performance】")
    print(f"  Combined Accuracy:  {results['joint_accuracy']:.4f} ({results['joint_accuracy']*100:.2f}%)")
    print(f"  Joint F1 Score:     {results['joint_f1']:.4f}")
    
    print("\n【Synergy Effect Analysis】")
    print(f"  Theoretical Joint Acc: {results['theoretical_joint']:.4f}")
    print(f"  Actual Joint Acc:      {results['joint_accuracy']:.4f}")
    print(f"  Synergy Gain:          {results['synergy_gain']:+.4f} ({results['synergy_gain']*100:+.2f}%)")
    
    if results['synergy_gain'] > 0:
        print(f"  ✓ POSITIVE SYNERGY!")
    
    print("\n【Error Pattern Analysis】")
    err = results['error_analysis']
    total = sum(err.values())
    print(f"  Both Correct:        {err['both_correct']} ({err['both_correct']/total*100:.2f}%)")
    print(f"  Only Disease Wrong:  {err['disease_only_wrong']} ({err['disease_only_wrong']/total*100:.2f}%)")
    print(f"  Only Severity Wrong: {err['severity_only_wrong']} ({err['severity_only_wrong']/total*100:.2f}%)")
    print(f"  Both Wrong:          {err['both_wrong']} ({err['both_wrong']/total*100:.2f}%)")
    
    print("="*70)

# ==================== 主函数 ====================
def main():
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    
    # 加载数据
    print("\nLoading validation dataset...")
    val_dataset = EvalDataset(VAL_JSON, VAL_IMG_DIR, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Loaded {len(val_dataset)} samples")
    
    # 加载模型
    print("\nLoading model...")
    model = ProgressiveMultiTaskModel().to(DEVICE)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("✓ Model loaded successfully")
    
    # 评估
    results = comprehensive_evaluation(model, val_loader, DEVICE)
    print_results(results)
    
    # 保存评估结果
    results_save = {k: v for k, v in results.items() if k != 'predictions'}
    with open(os.path.join(SAVE_DIR, "evaluation_results.json"), 'w', encoding='utf-8') as f:
        json.dump(results_save, f, indent=2, ensure_ascii=False)
    
    # 生成诊断报告
    all_reports = generate_diagnostic_reports(results, num_samples=len(val_dataset))
    
    # 保存完整报告
    with open(os.path.join(SAVE_DIR, "diagnostic_reports_full.json"), 'w', encoding='utf-8') as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
    
    # 保存示例报告（前20个）
    with open(os.path.join(SAVE_DIR, "diagnostic_reports_sample.json"), 'w', encoding='utf-8') as f:
        json.dump(all_reports[:20], f, indent=2, ensure_ascii=False)
    
    # 打印示例
    print("\n" + "="*70)
    print("SAMPLE DIAGNOSTIC REPORTS")
    print("="*70)
    for i, report in enumerate(all_reports[:3], 1):
        print(f"\n【Report #{i}】 {report['image_id']}")
        print(f"  Disease:   {report['diagnosis']['predicted_disease']}")
        print(f"  Severity:  {report['diagnosis']['predicted_severity']}")
        print(f"  Risk:      {report['diagnosis']['risk_level']}")
        print(f"  Disease Confidence:  {report['diagnosis']['disease_confidence']}")
        print(f"  Severity Confidence: {report['diagnosis']['severity_confidence']}")
        print(f"  Recommendation: {report['recommendations'][:80]}...")
        gt = report['ground_truth']
        status = "✓ Both Correct" if gt['both_correct'] else \
                ("⚠ Disease OK" if gt['disease_correct'] else "⚠ Severity OK" if gt['severity_correct'] else "✗ Both Wrong")
        print(f"  Status: {status}")
    
    print("\n" + "="*70)
    print("✓ Evaluation complete!")
    print(f"✓ Results saved to: {SAVE_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()