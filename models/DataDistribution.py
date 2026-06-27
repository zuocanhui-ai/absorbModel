# 数据分布对比图 - 非标准化 vs 标准化 (5×2布局)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import gaussian_kde

# 设置绘图样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ========== 读取数据 ==========
print("Loading data...")
data = pd.read_csv("../data/Data.csv")

feature_columns = [
    'SA (m2/g)', 'Dav (nm)', 'VTot (cm3/g)', 'C (wt%)',
    'O/C', '(O+N)/C', 'pH', 'T (K)', 'C0 (mg/L)', 'SLR (g/L)'
]
print(f"Feature columns: {feature_columns}")
print(f"Data shape: {data.shape}")

# ========== 绘制5×2分布对比图 ==========
print("\nGenerating 5×2 distribution comparison plots...")

fig, axes = plt.subplots(5, 2, figsize=(12, 18))
axes = axes.flatten()

for idx, col in enumerate(feature_columns):
    ax = axes[idx]

    # 原始数据
    original_data = data[col].values

    # 绘制原始数据的直方图（浅蓝色柱状图）
    n, bins, patches = ax.hist(original_data, bins=20, density=True,
                               alpha=0.6, color='skyblue', edgecolor='black',
                               linewidth=0.8)

    # 原始数据的核密度估计（蓝色实线）
    density_original = gaussian_kde(original_data)
    xs_original = np.linspace(original_data.min(), original_data.max(), 200)
    ax.plot(xs_original, density_original(xs_original),
            color='blue', linewidth=2.5, linestyle='-',
            label='Non-normalized data')

    # 正态分布曲线（红色虚线）
    mu, std = original_data.mean(), original_data.std()
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 200)
    p = stats.norm.pdf(x, mu, std)
    ax.plot(x, p, 'r--', linewidth=2.5, label='Normal Distribution')

    # 设置标题和标签
    ax.set_title(f'({chr(97 + idx)}) {col}', fontsize=12, fontweight='bold')
    ax.set_xlabel(col, fontsize=10)
    ax.set_ylabel('Density', fontsize=10)

    # 设置网格和图例
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=9, loc='upper right')
    ax.tick_params(labelsize=9)

    # 设置y轴范围从0开始
    ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('../figures/data_distribution_comparison_5x2.png', dpi=300, bbox_inches='tight')
print("✓ 5×2 distribution comparison plot saved as: data_distribution_comparison_5x2.png")
plt.show()

print("\n========== Plot Generation Completed ==========")