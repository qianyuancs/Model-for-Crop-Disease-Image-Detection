import json
import re
from collections import defaultdict, OrderedDict
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ========== 配置 ==========
JSON_PATH = r"C:\Users\21978\Desktop\大学\赛程知识\数模赛\数维杯\Question1\Yclean_labels.json"
OUT_DIR = "."  # 输出目录
os.makedirs(OUT_DIR, exist_ok=True)

# 0-60 标签映射（你之前给的字典）
label_id_to_name_en = {
    0: "Apple Healthy",
    1: "Apple Scab (General)",
    2: "Apple Scab (Serious)",
    3: "Apple Frogeye Spot",
    4: "Cedar Apple Rust (General)",
    5: "Cedar Apple Rust (Serious)",
    6: "Cherry Healthy",
    7: "Cherry Powdery Mildew (General)",
    8: "Cherry Powdery Mildew (Serious)",
    9: "Corn Healthy",
    10: "Cercospora Zeaemaydis Tehon and Daniels (General)",
    11: "Cercospora Zeaemaydis Tehon and Daniels (Serious)",
    12: "Corn Puccinia Polysora (General)",
    13: "Corn Puccinia Polysora (Serious)",
    14: "Corn Curvularia Leaf Spot (Fungus, General)",
    15: "Corn Curvularia Leaf Spot (Fungus, Serious)",
    16: "Maize Dwarf Mosaic Virus",
    17: "Grape Healthy",
    18: "Grape Black Rot (Fungus, General)",
    19: "Grape Black Rot (Fungus, Serious)",
    20: "Grape Black Measles (Fungus, General)",
    21: "Grape Black Measles (Fungus, Serious)",
    22: "Grape Leaf Blight (Fungus, General)",
    23: "Grape Leaf Blight (Fungus, Serious)",
    24: "Citrus Healthy",
    25: "Citrus Greening (General)",
    26: "Citrus Greening (Serious)",
    27: "Peach Healthy",
    28: "Peach Bacterial Spot (General)",
    29: "Peach Bacterial Spot (Serious)",
    30: "Pepper Healthy",
    31: "Pepper Scab (General)",
    32: "Pepper Scab (Serious)",
    33: "Potato Healthy",
    34: "Potato Early Blight (Fungus, General)",
    35: "Potato Early Blight (Fungus, Serious)",
    36: "Potato Late Blight (Fungus, General)",
    37: "Potato Late Blight (Fungus, Serious)",
    38: "Strawberry Healthy",
    39: "Strawberry Scorch (General)",
    40: "Strawberry Scorch (Serious)",
    41: "Tomato Healthy",
    42: "Tomato Powdery Mildew (General)",
    43: "Tomato Powdery Mildew (Serious)",
    44: "Tomato Bacterial Spot (Bacteria, General)",
    45: "Tomato Bacterial Spot (Bacteria, Serious)",
    46: "Tomato Early Blight (Fungus, General)",
    47: "Tomato Early Blight (Fungus, Serious)",
    48: "Tomato Late Blight (Water Mold, General)",
    49: "Tomato Late Blight (Water Mold, Serious)",
    50: "Tomato Leaf Mold (Fungus, General)",
    51: "Tomato Leaf Mold (Fungus, Serious)",
    52: "Tomato Target Spot (Bacteria, General)",
    53: "Tomato Target Spot (Bacteria, Serious)",
    54: "Tomato Septoria Leaf Spot (Fungus, General)",
    55: "Tomato Septoria Leaf Spot (Fungus, Serious)",
    56: "Tomato Spider Mite Damage (General)",
    57: "Tomato Spider Mite Damage (Serious)",
    58: "Tomato Yellow Leaf Curl Virus (General)",
    59: "Tomato Yellow Leaf Curl Virus (Serious)",
    60: "Tomato Mosaic Virus"
}

# ========== 工具函数 ==========
def extract_main_name(label_name: str) -> str:
    """去掉括号里的信息，得到疾病主类名"""
    base = re.sub(r"\(.*?\)", "", label_name).strip()
    return base

def extract_severity(label_name: str) -> int:
    """判定 severity:
       0 -> Healthy
       1 -> General
       2 -> Serious
       规则：包含 'Healthy' -> 0; 包含 'Serious' -> 2; else -> 1
    """
    ln = label_name.lower()
    if "healthy" in ln:
        return 0
    if "serious" in ln:
        return 2
    # 默认 general
    return 1

# ========== 构建主类字典（主类名 -> 主类 id） ==========
id_to_main_name = {i: extract_main_name(n) for i,n in label_id_to_name_en.items()}
main_names = sorted(list(set(id_to_main_name.values())))  # 排序以保证稳定 id
main_name_to_id = {name: idx for idx, name in enumerate(main_names)}

print(f"Found {len(main_names)} main disease classes.")

# ========== 读入原始标签 ==========
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)  # 假设是 list of {"disease_class": int, "image_id": str}

# 输出列表（逐条增强）
relabelled = []
# 统计汇总结构
summary = {name: {"total":0, "healthy":0, "general":0, "serious":0} for name in main_names}

# 处理每条标签
for entry in data:
    orig_id = int(entry["disease_class"])
    img_id = entry.get("image_id") or entry.get("image") or entry.get("filename")  # 兼容字段名
    orig_name = label_id_to_name_en.get(orig_id, "UNKNOWN")
    main_name = id_to_main_name[orig_id]
    main_id = main_name_to_id[main_name]
    sev = extract_severity(orig_name)  # 0/1/2

    # 统计
    summary[main_name]["total"] += 1
    if sev == 0:
        summary[main_name]["healthy"] += 1
    elif sev == 1:
        summary[main_name]["general"] += 1
    else:
        summary[main_name]["serious"] += 1

    # 构造增强条目
    new_entry = {
        "image_id": img_id,
        "orig_label_id": orig_id,
        "orig_label_name": orig_name,
        "disease_main_id": main_id,
        "disease_main_name": main_name,
        "severity_id": sev
    }
    relabelled.append(new_entry)

# ========== 保存 relabelled dataset JSON ==========
out_relabel_path = os.path.join(OUT_DIR, "Yrelabelled_dataset.json")
with open(out_relabel_path, "w", encoding="utf-8") as f:
    json.dump(relabelled, f, ensure_ascii=False, indent=2)
print("Saved relabelled dataset to:", out_relabel_path)

# ========== 保存 summary JSON ==========
out_summary_path = os.path.join(OUT_DIR, "Ydisease_summary.json")
with open(out_summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("Saved disease summary to:", out_summary_path)

# ========== 保存 summary CSV ==========
out_csv = os.path.join(OUT_DIR, "Ydisease_summary.csv")
with open(out_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["disease_main_name", "total", "healthy", "general", "serious"])
    for name in main_names:
        s = summary[name]
        writer.writerow([name, s["total"], s["healthy"], s["general"], s["serious"]])
print("Saved CSV to:", out_csv)

# ========== 画堆叠柱状图 ==========
# 为了可视化排序，我们按 total 降序排列主类
sorted_names = sorted(main_names, key=lambda n: summary[n]["total"], reverse=True)
healthy_vals = [summary[n]["healthy"] for n in sorted_names]
general_vals = [summary[n]["general"] for n in sorted_names]
serious_vals = [summary[n]["serious"] for n in sorted_names]
x = range(len(sorted_names))

plt.figure(figsize=(16,9))
plt.bar(x, healthy_vals, label="healthy")
plt.bar(x, general_vals, bottom=healthy_vals, label="general")
# compute bottom for serious: healthy + general
bottom_serious = [h + g for h,g in zip(healthy_vals, general_vals)]
plt.bar(x, serious_vals, bottom=bottom_serious, label="serious")
plt.xticks(x, sorted_names, rotation=90)
plt.ylabel("Number of images")
plt.title("Main disease counts (healthy/general/serious)")
plt.legend()
plt.tight_layout()
out_img = os.path.join(OUT_DIR, "main_disease_stacked.png")
plt.savefig(out_img, dpi=300)
plt.close()
print("Saved stacked bar chart to:", out_img)
