import os
import matplotlib

import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt


IMG_DIR = "images"

class_count = defaultdict(int)

for fname in os.listdir(IMG_DIR):
    # 过滤非图片文件
    if not fname.lower().endswith((".jpg")):
        continue
    
    try:
        cls = int(fname.split("_")[0])
        class_count[cls] += 1
    except:
        print("文件名格式不合法:", fname)

# 转成 DataFrame 方便分析
df = pd.DataFrame(sorted(class_count.items()), columns=["class", "count"])

df.to_csv("class_distribution.csv", index=False)
print(df)
print("\n总类别数：", len(df))
print("总样本数：", df["count"].sum())


#这里来一个图可视化
plt.figure(figsize=(14, 5))
plt.bar(df["class"], df["count"])
plt.xlabel("Class Index")
plt.ylabel("Number of Images")
plt.title("Class Distribution")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=300)
plt.show()

