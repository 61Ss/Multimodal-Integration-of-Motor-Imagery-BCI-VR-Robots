import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 1. 读取 CSV 并重命名列
df = pd.read_csv('electrode_coords.csv')
df = df.rename(columns={
    'channel_label': 'name',
    'x_position':    'x',
    'y_position':    'y'
})

# 2. 提取 2D 坐标和通道名
coords2d = df[['x', 'y']].to_numpy()   # shape = (118, 2)
names2d  = df['name'].tolist()         # list of 118 channel names

# 3. 基于静息态功能网络的粗粒度分区并配色
#    Yeo 7 网聚合为三大模块：
#      - Visual：Visual 网络
#      - Somatomotor：Somatomotor 网络
#      - Association：DorsalAttention, VentralAttention, Limbic, FrontoparietalControl, DefaultMode
#    这里假设电极前缀与网络的近似对应：
#      O → Visual
#      C → Somatomotor
#      其余 F,P,T → Association
region_map = {
    'O': 'Visual',
    'C': 'Somatomotor',
    'F': 'Association',
    'P': 'Association',
    'T': 'Association'
}
# 按通道名前缀映射到功能网络
ch_networks = [region_map.get(ch[0], 'Other') for ch in names2d]
unique_nets = sorted(set(ch_networks))

# 生成颜色映射
cmap      = plt.get_cmap('tab10')
color_map = {net: cmap(i) for i, net in enumerate(unique_nets)}
colors    = [color_map[n] for n in ch_networks]

# 4. 绘制散点并标注
plt.figure(figsize=(7,7))
plt.scatter(coords2d[:,0], coords2d[:,1],
            c=colors, s=80, edgecolors='k')
for (x, y), ch in zip(coords2d, names2d):
    plt.text(x, y, ch, ha='center', va='center', fontsize=7)

# 5. 添加图例
legend_handles = [
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor=color_map[net], markersize=8,
           label=net)
    for net in unique_nets
]
plt.legend(handles=legend_handles, title='Network',
           loc='upper right', frameon=False)

plt.title('118 EEG Electrodes Coarse Resting-State Networks', pad=15)
plt.axis('off')
plt.tight_layout()
plt.show()
