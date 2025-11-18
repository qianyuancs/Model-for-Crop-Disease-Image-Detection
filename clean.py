import os
from PIL import Image
import pandas as pd
import hashlib
import shutil
import json

#本代码在完成数据清洗，共四个部分：1.检查坏图 2.删除官方标注为duplicate的图片 3.检查重复图片 4.检查孤立图片并删除


#图像/json的路径/名称
IMG_DIR = "images-31718"
JSON_FILE = "Tclean_labels.json"


#看图片有没有坏
bad_images = []

for fname in os.listdir(IMG_DIR):
    fpath = os.path.join(IMG_DIR, fname)
    try:
        img = Image.open(fpath)
        img.verify()  # 检查文件结构
    except Exception as e:
        bad_images.append((fname, str(e)))

# 保存日志
df_bad = pd.DataFrame(bad_images, columns=["filename", "error"])
df_bad.to_csv("bad_images.csv", index=False)
print("坏图数量：", len(df_bad))
# === 删除坏图 ===
os.makedirs("removed_bad", exist_ok=True)

for fname in df_bad["filename"]:
    src = os.path.join(IMG_DIR, fname)
    if os.path.exists(src):
        shutil.move(src, os.path.join("removed_bad", fname))

print("坏图已移动到 removed_bad/")




#下面的代码是认为json文件里会有类别标注为了“duplicate”，然后要找到这些图片并且删除

with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# 找到所有官方 duplicate 标注
duplicate_rows = df[df["disease_class"].astype(str).str.lower() == "duplicate"]

duplicate_rows.to_csv("official_duplicate_annotations.csv", index=False)

print("官方 duplicate 标注数量:", len(duplicate_rows))



dup_df = pd.read_csv("official_duplicate_annotations.csv")

os.makedirs("removed_duplicates", exist_ok=True)

for fname in dup_df["image_id"]:
    src = os.path.join(IMG_DIR, fname)
    if os.path.exists(src):
        shutil.move(src, os.path.join("removed_duplicates", fname))

print("所有 duplicate 标注图已移动到 removed_duplicates/")


# === 文件名包含 duplicate 的图片 ===
os.makedirs("removed_name_duplicate", exist_ok=True)

name_dup = []

for fname in os.listdir(IMG_DIR):
    # 名字中包含 duplicate（忽略大小写）
    if "副本" in fname.lower():
        name_dup.append(fname)
        src = os.path.join(IMG_DIR, fname)
        shutil.move(src, os.path.join("removed_name_duplicate", fname))

# 保存日志
pd.DataFrame({"filename": name_dup}).to_csv("filename_duplicate_images.csv", index=False)

print("根据文件名识别出的 duplicate 图片数量：", len(name_dup))
print("已移动到 removed_name_duplicate/")




#接下来这个代码是在检查有哪些图片是json里有的但是图像文件夹里没有的

# 读取 JSON
with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)
df = pd.DataFrame(data)

# 取 json 中所有 image_id
json_imgs = set(df["image_id"].tolist())

# 取文件夹中所有文件名
folder_imgs = set(os.listdir(IMG_DIR))

# JSON 里有，但文件夹没有
missing_imgs = json_imgs - folder_imgs

df_missing = pd.DataFrame({"image_id": list(missing_imgs)})
df_missing.to_csv("json_missing_images.csv", index=False)

print("JSON 中存在但文件夹缺失的图片数量：", len(df_missing))


#接下来这个代码是在检查有哪些图片是json里有的但是图像文件夹里没有的

# 文件夹有但 JSON 没有
extra_imgs = folder_imgs - json_imgs

df_extra = pd.DataFrame({"image_id": list(extra_imgs)})
df_extra.to_csv("folder_extra_images.csv", index=False)

print("文件夹存在但 JSON 未标注的图片数量：", len(df_extra))

#接下来的代码是要把这些孤立的图片删除
os.makedirs("removed_orphans", exist_ok=True)

for fname in extra_imgs:
    src = os.path.join(IMG_DIR, fname)
    if os.path.exists(src):
        shutil.move(src, os.path.join("removed_orphans", fname))

print("孤立图片已移动到 removed_orphans/")