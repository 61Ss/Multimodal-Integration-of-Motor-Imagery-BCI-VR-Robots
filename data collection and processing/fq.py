import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 1. 读取 CSV 并重命名列
df = pd.read_csv('electrode_coords.csv')                      # 假设列是 channel_label, x_position, y_position
df = df.rename(columns={
    'channel_label': 'name',
    'x_position':    'x',
    'y_position':    'y'
})

# 2. 提取 2D 坐标和通道名
#    DataFrame.to_numpy() 返回 numpy.ndarray，可直接索引用于 scatter :contentReference[oaicite:1]{index=1}
coords2d = df[['x', 'y']].to_numpy()   # shape = (118, 2)
names2d  = df['name'].tolist()         # list of 118 channel names

# 3. 简单按前缀分区并配色（可根据需要调整）
region_map = {'F': 'Frontal', 'C': 'Central',
              'P': 'Parietal', 'O': 'Occipital',
              'T': 'Temporal'}
ch_regions = [region_map.get(ch[0], 'Other') for ch in names2d]
unique_regs = sorted(set(ch_regions))
cmap       = plt.get_cmap('tab10')
color_map  = {reg: cmap(i) for i, reg in enumerate(unique_regs)}
colors     = [color_map[r] for r in ch_regions]

# 4. 绘制散点并标注
plt.figure(figsize=(7,7))
plt.scatter(coords2d[:,0], coords2d[:,1],
            c=colors, s=80, edgecolors='k')
for (x, y), ch in zip(coords2d, names2d):
    plt.text(x, y, ch, ha='center', va='center', fontsize=7)

# 5. 添加图例
legend_handles = [
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor=color_map[reg], markersize=8,
           label=reg)
    for reg in unique_regs
]
plt.legend(handles=legend_handles, title='Region',
           loc='upper right', frameon=False)

plt.title('118 EEG Electrodes Region Visualization (10–20)', pad=15)
plt.axis('off')
plt.tight_layout()
plt.show()
