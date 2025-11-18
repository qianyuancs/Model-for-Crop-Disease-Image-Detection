import os
import json
import pandas as pd

IMG_DIR = "images-31718"       # 图像文件夹的名称
OUTPUT_JSON = "Tclean_labels.json"     #最后输出的json文件名字
OUTPUT_CSV = "Tclean_labels.csv"

records = []

for fname in os.listdir(IMG_DIR):
    if not fname.lower().endswith((".jpg")):
        continue

    try:
    
        class_str = fname.split("_")[0]
        disease_class = int(class_str)  
    except Exception as e:
        print("文件名格式错误：", fname)
        continue

    record = {
        "disease_class": disease_class,
        "image_id": fname
    }

    records.append(record)

# 保存 JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

# 保存 CSV
df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)

print("标签生成完成！")
print("生成 JSON 文件：", OUTPUT_JSON)
print("生成 CSV 文件：", OUTPUT_CSV)
print("共写入标签数量：", len(records))
