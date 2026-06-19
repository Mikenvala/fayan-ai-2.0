#!/usr/bin/env python3
"""生成多模型对比雷达图（从 multi_benchmark_results.json 读取）"""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# Load results
with open(sys.argv[1] if len(sys.argv) > 1 else "multi_benchmark_results.json") as f:
    data = json.load(f)

# Find Chinese font
zh_font = None
# Try Songti SC first (system font), fallback to PingFang
for fp in ['/System/Library/Fonts/Supplemental/Songti.ttc', '/System/Library/Fonts/PingFang.ttc', '/System/Library/Fonts/STHeiti Light.ttc']:
    if os.path.exists(fp):
        zh_font = fm.FontProperties(fname=fp)
        break

cats_order = ["法条适用", "罪名判断", "量刑推理", "案例分析", "程序问题"]
models = list(data.keys())
N = len(cats_order)

fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(projection='polar'))
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

colors = {'MiniMax-M2.7': '#3B82F6', 'DeepSeek-V3': '#10B981', 
          'Kimi (Moonshot)': '#F59E0B', '智谱GLM-4-Flash': '#8B5CF6'}

ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
if zh_font:
    ax.set_xticklabels(cats_order, fontsize=13, fontproperties=zh_font)
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20%','40%','60%','80%','100%'], fontsize=9)

for model_name, model_data in data.items():
    scores = [model_data["categories"][c]["accuracy"] for c in cats_order]
    scores += scores[:1]
    color = colors.get(model_name, '#9CA3AF')
    ax.fill(angles, scores, color=color, alpha=0.08)
    ax.plot(angles, scores, color=color, linewidth=2.5, marker='o', markersize=9,
            markerfacecolor='white', markeredgecolor=color, markeredgewidth=2, label=model_name)

ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=11, framealpha=0.9)
ax.set_title('四款中国大模型法律能力五维对比', fontsize=16, fontweight='bold', pad=25,
             fontproperties=zh_font if zh_font else None)

# Summary table
summary_text = f"30题法律评测基准 | 总分率排名:\n"
sorted_models = sorted(data.items(), key=lambda x: x[1]["accuracy"], reverse=True)
for i, (name, d) in enumerate(sorted_models, 1):
    summary_text += f"{i}. {name}: {d['accuracy']:.1f}%\n"

fig.text(0.5, -0.02, summary_text.strip(), ha='center', fontsize=10, color='#374151',
         family=zh_font.get_name() if zh_font else 'sans-serif')

out = os.path.join(os.path.dirname(__file__), "figures", "fig06_multimodel_radar.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.tight_layout()
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {out}")
