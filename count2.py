import matplotlib
# 切换到非 GUI 后端
matplotlib.use('Agg')
import json
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# =============================
# severity 映射规则
# =============================
healthy_ids = [0, 6, 9, 17, 24, 27, 30, 33, 38, 41]
serious_ids = [2,5,8,11,13,15,16,19,21,23,26,29,32,35,37,40,43,45,47,49,51,53,55,57,59,60]

def label_to_severity(label_id):
    if label_id in healthy_ids:
        return 0  # Healthy
    elif label_id in serious_ids:
        return 2  # Serious
    else:
        return 1  # General

# =============================
# label_id -> 作物映射
# =============================
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

# =============================
# 读取 JSON 文件
# =============================
json_file =r"C:\Users\21978\Desktop\大学\赛程知识\数模赛\数维杯\Question1\Tclean_labels.json" # 替换为你的 JSON 文件路径
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# =============================
# 按作物分组统计 severity
# =============================
species_severity_counts = defaultdict(lambda: Counter())

for item in data:
    label_id = item['disease_class']
    species = label_to_species[label_id]
    severity = label_to_severity(label_id)
    species_severity_counts[species][severity] += 1

# =============================
# 打印每种作物统计
# =============================
severity_names = {0:'Healthy',1:'General',2:'Serious'}
for species, counts in species_severity_counts.items():
    print(f"{species}:")
    for sev in range(3):
        print(f"  {severity_names[sev]}: {counts.get(sev,0)}")
    print()

# =============================
# 绘制柱状图
# =============================
import numpy as np

species_list = list(species_severity_counts.keys())
x = np.arange(len(species_list))
width = 0.25

healthy_vals = [species_severity_counts[s].get(0,0) for s in species_list]
general_vals = [species_severity_counts[s].get(1,0) for s in species_list]
serious_vals = [species_severity_counts[s].get(2,0) for s in species_list]

plt.figure(figsize=(12,6))
plt.bar(x - width, healthy_vals, width, label='Healthy', color='green')
plt.bar(x, general_vals, width, label='General', color='orange')
plt.bar(x + width, serious_vals, width, label='Serious', color='red')

plt.xticks(x, species_list, rotation=45)
plt.ylabel("Number of Images")
plt.title("Severity Distribution per Crop")
plt.legend()
plt.tight_layout()
plt.savefig("severity_distribution_per_crop.png")
print("柱状图已保存为 severity_distribution_per_crop.png")
plt.close()  # 关闭 figure 防止显示报错


