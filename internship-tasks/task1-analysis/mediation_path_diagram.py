import os, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path
import matplotlib.font_manager as fm
import numpy as np

heiti_medium = '/System/Library/Fonts/STHeiti Medium.ttc'
heiti_light = '/System/Library/Fonts/STHeiti Light.ttc'
for f in [heiti_medium, heiti_light]:
    if os.path.exists(f):
        fm.fontManager.addfont(f)
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'STHeiti', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT, exist_ok=True)

# ── 参数 ──
p = {
    'c': 0.0556, 'c_prime': 0.0515,
    'a1': 0.0684, 'a2': -0.0000,
    'b1': 0.05994, 'b2': 0.03662,
    'ind1': 0.00410, 'ind1_lo': 0.00223, 'ind1_hi': 0.00608,
}

RED   = '#DC2626'
GRAY  = '#9CA3AF'
GREEN = '#059669'
DARK  = '#1F2937'

# ── 精确坐标布局 ──
BW, BH = 2.8, 1.2
HALF_W, HALF_H = BW / 2, BH / 2

BOX_T  = np.array([1.8, 4.5])
BOX_M1 = np.array([6.0, 6.8])
BOX_M2 = np.array([6.0, 2.2])
BOX_Y  = np.array([10.2, 4.5])

def box_edge(center, direction):
    x, y = center
    return {
        'left':   (x - HALF_W, y),
        'right':  (x + HALF_W, y),
        'top':    (x, y + HALF_H),
        'bottom': (x, y - HALF_H),
        'tl':     (x - HALF_W, y + HALF_H),
        'tr':     (x + HALF_W, y + HALF_H),
        'bl':     (x - HALF_W, y - HALF_H),
        'br':     (x + HALF_W, y - HALF_H),
    }[direction]

ARROWS = [
    ('a', box_edge(BOX_T, 'tr'), box_edge(BOX_M1, 'left'),
     f"a1 = {p['a1']:+.4f}***", RED, 2.8),
    ('b', box_edge(BOX_T, 'br'), box_edge(BOX_M2, 'left'),
     f"a2 ≈ 0 (n.s.)", GRAY, 2.0),
    ('c', box_edge(BOX_M1, 'right'), box_edge(BOX_Y, 'tl'),
     f"b1 = {p['b1']*100:+.2f}pp***", RED, 2.8),
    ('d', box_edge(BOX_M2, 'right'), box_edge(BOX_Y, 'bl'),
     f"b2 = {p['b2']*100:+.2f}pp***", RED, 2.8),
    ('e', box_edge(BOX_T, 'right'), box_edge(BOX_Y, 'left'),
     f"c' = {p['c_prime']*100:+.2f}pp\n(PSM匹配后不显著)", GRAY, 2.5),
]

# ============================================================
# 图A: 因果中介路径图
# ============================================================
fig, ax = plt.subplots(figsize=(17, 10))
ax.set_xlim(-0.5, 12.5)
ax.set_ylim(-0.5, 9.5)
ax.set_aspect('equal')
ax.axis('off')

# ── 画变量盒 ──
boxes = [
    (BOX_T,  '提及涉案金额\n(处理变量T)',  '#FEF2F2', RED),
    (BOX_M1, '争议复杂度 M1\n判别标准篇幅(log)', '#FFF7ED', '#EA580C'),
    (BOX_M2, '概念丰富度 M2\n关键词数量', '#FFF7ED', '#EA580C'),
    (BOX_Y,  '驳回判决\n(结果变量Y)', '#ECFDF5', GREEN),
]
for (cx, cy), text, fc, ec in boxes:
    rect = FancyBboxPatch(
        (cx - HALF_W, cy - HALF_H), BW, BH,
        boxstyle='round,pad=0.12',
        facecolor=fc, edgecolor=ec, linewidth=2.8, zorder=5
    )
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=11.5,
            fontweight='bold', color=DARK, zorder=6)

# ── 画箭头 ──
def mid_vec(p1, p2, offset_angle=90, offset_dist=0.45):
    mx = (p1[0] + p2[0]) / 2
    my = (p1[1] + p2[1]) / 2
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = np.sqrt(dx*dx + dy*dy)
    if length == 0:
        return mx, my
    nx = -dy / length
    ny = dx / length
    rad = np.radians(offset_angle)
    rx = nx * np.cos(rad) - ny * np.sin(rad)
    ry = nx * np.sin(rad) + ny * np.cos(rad)
    return mx + rx * offset_dist, my + ry * offset_dist

for _, start, end, label, color, lw in ARROWS:
    linestyle = 'dashed' if 'PSM' in label else 'solid'
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                linestyle=linestyle, shrinkA=0, shrinkB=0),
                zorder=4)

    mx, my = mid_vec(start, end, 90, 0.45)
    if 'a2' in label:   mx, my = mid_vec(start, end, 90, 0.35)
    if 'b1' in label:   mx, my = mid_vec(start, end, 90, 0.50)
    if 'b2' in label:   mx, my = mid_vec(start, end, -90, 0.50)
    if "c'" in label:   mx, my = (BOX_T[0] + BOX_Y[0]) / 2, 4.5 + 0.9

    ax.text(mx, my, label, ha='center', va='center', fontsize=10,
            fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor='none', alpha=0.92), zorder=7)

# ═══════════════════════════════════════════════════════════
# 间接效应绿色凸曲线 — 精确 Bezier
# 从 T右(3.2,4.5) → Y左(8.8,4.5)，顶点在 M1底部(6.0,6.2)
# ═══════════════════════════════════════════════════════════
P0 = np.array([3.2, 4.5])      # T 右边缘中心
P3 = np.array([8.8, 4.5])      # Y 左边缘中心
# 对称控制点: y=6.767 使得 t=0.5 时顶点y=6.2 (M1底部)
Y_CTRL = 6.767
P1 = np.array([4.4, Y_CTRL])
P2 = np.array([7.6, Y_CTRL])
VERTEX = np.array([6.0, 6.2])  # M1 底部中心

# 生成 Bezier 曲线点
t_vals = np.linspace(0, 1, 200)
curve_x = ((1-t_vals)**3 * P0[0] + 3*(1-t_vals)**2*t_vals * P1[0] +
           3*(1-t_vals)*t_vals**2 * P2[0] + t_vals**3 * P3[0])
curve_y = ((1-t_vals)**3 * P0[1] + 3*(1-t_vals)**2*t_vals * P1[1] +
           3*(1-t_vals)*t_vals**2 * P2[1] + t_vals**3 * P3[1])

# 画绿色曲线
ax.plot(curve_x, curve_y, color=GREEN, lw=4.0, zorder=3, alpha=0.85)

# 顶点绿色圆点
ax.scatter(*VERTEX, color=GREEN, s=80, zorder=8, edgecolors='white', linewidths=1.5)

# 顶点到 M1 底部的虚线连接
ax.plot([VERTEX[0], BOX_M1[0]], [VERTEX[1], BOX_M1[1] - HALF_H],
        color=GREEN, lw=1.5, linestyle=':', zorder=3, alpha=0.7)

# 曲线两端小三角标记

# 在曲线上绘制箭头标记（80%位置）
idx80 = int(len(t_vals) * 0.80)
ax.annotate('', xy=(curve_x[idx80 + 1], curve_y[idx80 + 1]),
            xytext=(curve_x[idx80], curve_y[idx80]),
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=2.5, alpha=0.9),
            zorder=3)

# 间接效应标签（M1右侧空白区，不遮挡M1方框）
label_x = 7.6
label_y = 5.8
# 虚线连接标签到曲线中点
mid_curve_x = (P0[0] + P3[0]) / 2 + 0.5
mid_curve_y = VERTEX[1]
ax.plot([label_x - 0.5, mid_curve_x], [label_y, mid_curve_y],
        color=GREEN, lw=1.0, linestyle=':', alpha=0.5, zorder=2)
ax.text(label_x, label_y,
        f"间接效应: a1 x b1 = {p['ind1']*100:+.3f}pp ***\n"
        f"中介占比 7.4% | 95%CI [{p['ind1_lo']*100:+.3f}, {p['ind1_hi']*100:+.3f}]pp",
        ha='center', va='center', fontsize=9.5, fontweight='bold', color=DARK,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#D1D5DB', linewidth=0.8, alpha=0.95), zorder=9)

# ── 总效应 ──
tx, ty = BOX_T[0], BOX_T[1] - HALF_H - 1.0
ax.plot([BOX_T[0], BOX_T[0]], [BOX_T[1] - HALF_H, ty + 0.3],
        color=GRAY, lw=2, linestyle='--', alpha=0.5, zorder=2)
ax.text(tx, ty,
        f"总效应 c = {p['c']*100:+.2f}pp ***",
        ha='center', va='center', fontsize=11.5, fontweight='bold', color=DARK,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F3F4F6',
                  edgecolor=GRAY, linewidth=1.5), zorder=8)

# ── 图例 ──
leg_items = [
    mpatches.Patch(color='#FEF2F2', label='处理变量 (T)'),
    mpatches.Patch(color='#FFF7ED', label='中介变量 (M1, M2)'),
    mpatches.Patch(color='#ECFDF5', label='结果变量 (Y)'),
    plt.Line2D([0],[0], color=RED, lw=2.5, label='*** p<0.001'),
    plt.Line2D([0],[0], color=GRAY, lw=2.0, linestyle='dashed', label='n.s.'),
    plt.Line2D([0],[0], color=GREEN, lw=3.5, label='间接效应路径'),
]
ax.legend(handles=leg_items, loc='lower right', fontsize=9.5,
          framealpha=0.92, edgecolor='#D1D5DB', ncol=3,
          bbox_to_anchor=(0.98, 0.03))

ax.set_title('因果中介路径图：金额提及 → 驳回判决的传导机制',
             fontsize=15, fontweight='bold', pad=25)

fig.tight_layout(pad=0.5)
fig.savefig(os.path.join(OUTPUT, 'fig_mediation_path.png'), dpi=200)
plt.close(fig)
print("✅ fig_mediation_path.png 已生成（绿色凸曲线）")

# ============================================================
# 图B: 效应分解表
# ============================================================
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.axis('off')

rows = [
    ['路径', '描述', '系数', '效应量(pp)', '95%CI', '显著性'],
    ['总效应 c',    'T → Y',                '—', f'{p["c"]*100:+.2f}',      '—', '***'],
    ['直接效应 c\'', 'T → Y | M1, M2',       '—', f'{p["c_prime"]*100:+.2f}', '—', 'PSM后n.s.'],
    ['a1',          'T → M1 (争议复杂度)',   f'{p["a1"]:+.4f}', '—', '—', '***'],
    ['a2',          'T → M2 (概念丰富度)',   '≈0', '—', '—', 'n.s.'],
    ['b1',          'M1 → Y | T',           f'{p["b1"]*100:+.3f}pp', '—', '—', '***'],
    ['b2',          'M2 → Y | T',           f'{p["b2"]*100:+.3f}pp', '—', '—', '***'],
    ['间接 a1 x b1',  'T→M1→Y',               '—', f'{p["ind1"]*100:+.3f}',
     f'[{p["ind1_lo"]*100:+.3f}, {p["ind1_hi"]*100:+.3f}]', '***'],
    ['间接 a2 x b2',  'T→M2→Y',               '—', '≈0', '—', 'n.s.'],
    ['中介占比',     '(a1 x b1)/c',            '—', '7.4%', '—', '—'],
]

tbl = ax.table(cellText=rows, cellLoc='center',
               colWidths=[0.14, 0.26, 0.16, 0.14, 0.22, 0.08],
               loc='center', edges='horizontal')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)

for j in range(6):
    tbl[0, j].set_facecolor('#1F2937')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

for i in range(1, len(rows)):
    bg = '#F9FAFB' if i % 2 == 0 else 'white'
    for j in range(6):
        tbl[i, j].set_facecolor(bg)
    if rows[i][5] == '***':
        tbl[i, 5].set_text_props(color=RED, fontweight='bold')
        tbl[i, 0].set_facecolor('#FEF2F2')
    if rows[i][5] == 'PSM后n.s.':
        tbl[i, 5].set_text_props(color=GRAY)

tbl.scale(1.0, 1.9)
ax.set_title('因果中介效应分解', fontsize=14, fontweight='bold', pad=10, y=1.06)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT, 'fig_mediation_table.png'), dpi=200)
plt.close(fig)
print("✅ fig_mediation_table.png 已生成")
print("全部完成")
