import os, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm
import numpy as np

heiti = '/System/Library/Fonts/STHeiti Light.ttc'
if os.path.exists(heiti):
    fm.fontManager.addfont(heiti)
    plt.rcParams['font.sans-serif'] = ['Heiti TC']
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

# 颜色
RED   = '#DC2626'  # 显著/处理变量
GRAY  = '#9CA3AF'  # 不显著
GREEN = '#059669'  # 间接效应
DARK  = '#1F2937'  # 文字
WHITE = '#FFFFFF'

# ── 精确坐标布局 ──
# 盒尺寸
BW, BH = 2.8, 1.2          # 宽,高
HALF_W = BW / 2
HALF_H = BH / 2

# 盒中心坐标
BOX_T  = np.array([1.8, 4.5])   # T: 处理变量
BOX_M1 = np.array([6.0, 6.8])   # M1: 争议复杂度
BOX_M2 = np.array([6.0, 2.2])   # M2: 概念丰富度
BOX_Y  = np.array([10.2, 4.5])  # Y: 结果变量

def box_edge(center, direction):
    """返回盒子的边坐标: 'top','bottom','left','right','tl','tr','bl','br'"""
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

# 精确箭头起止点
ARROWS = [
    # a1: T → M1
    ('a', box_edge(BOX_T, 'tr'), box_edge(BOX_M1, 'left'),
     f"a1 = {p['a1']:+.4f}***", RED, 2.8),
    # a2: T → M2
    ('b', box_edge(BOX_T, 'br'), box_edge(BOX_M2, 'left'),
     f"a2 ≈ 0 (n.s.)", GRAY, 2.0),
    # b1: M1 → Y
    ('c', box_edge(BOX_M1, 'right'), box_edge(BOX_Y, 'tl'),
     f"b1 = {p['b1']*100:+.2f}pp***", RED, 2.8),
    # b2: M2 → Y
    ('d', box_edge(BOX_M2, 'right'), box_edge(BOX_Y, 'bl'),
     f"b2 = {p['b2']*100:+.2f}pp***", RED, 2.8),
    # c': T → Y 直接效应（虚线）
    ('e', box_edge(BOX_T, 'right'), box_edge(BOX_Y, 'left'),
     f"c' = {p['c_prime']*100:+.2f}pp\n(PSM匹配后不显著)", GRAY, 2.5),
]

# ============================================================
fig, ax = plt.subplots(figsize=(17, 10))
ax.set_xlim(-0.5, 12.5)
ax.set_ylim(-0.5, 9.5)
ax.set_aspect('equal')
ax.axis('off')

# ── 画背景网格（淡色辅助线，用于调试，最终可注释掉）──
# for x in np.arange(0, 12.5, 1):
#     ax.axvline(x, color='#E5E7EB', lw=0.5, zorder=0)
# for y in np.arange(0, 9.5, 1):
#     ax.axhline(y, color='#E5E7EB', lw=0.5, zorder=0)

# ── 画四个变量盒 ──
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

# ── 画箭头（精确坐标）──
def mid_vec(p1, p2, offset_angle=90, offset_dist=0.45):
    """返回箭头中点的偏移位置，用于放标签"""
    mx = (p1[0] + p2[0]) / 2
    my = (p1[1] + p2[1]) / 2
    # 垂直方向偏移
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = np.sqrt(dx*dx + dy*dy)
    if length == 0:
        return mx, my
    # 垂直单位向量
    nx = -dy / length
    ny = dx / length
    rad = np.radians(offset_angle)
    # 旋转 offset_angle
    rx = nx * np.cos(rad) - ny * np.sin(rad)
    ry = nx * np.sin(rad) + ny * np.cos(rad)
    return mx + rx * offset_dist, my + ry * offset_dist

for _, start, end, label, color, lw in ARROWS:
    # 箭头线
    style = 'arc3,rad=0' if color != GRAY else 'arc3,rad=0'
    linestyle = 'dashed' if 'PSM' in label else 'solid'
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                linestyle=linestyle, shrinkA=0, shrinkB=0),
                zorder=4)
    
    # 标签位置: 箭头中点上方（垂直于箭头方向偏移）
    mx, my = mid_vec(start, end, offset_angle=90, offset_dist=0.55)
    # 对特定箭头微调
    if 'a2' in label:   mx, my = mid_vec(start, end, 90, 0.35)
    if 'b1' in label:   mx, my = mid_vec(start, end, 90, 0.50)
    if 'b2' in label:   mx, my = mid_vec(start, end, -90, 0.50)
    if "c'" in label:   mx, my = (BOX_T[0] + BOX_Y[0]) / 2, 4.5 + 0.9
    
    ax.text(mx, my, label, ha='center', va='center', fontsize=10,
            fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor='none', alpha=0.92), zorder=7)

# ── 间接效应弧线标注 ──
# 从 T 上方 → M1 上方画一个弧
arc_start = (BOX_T[0] + 0.5, BOX_T[1] + HALF_H + 0.2)
arc_end   = (BOX_M1[0] - 0.5, BOX_M1[1] + HALF_H + 0.2)
ax.annotate('', xy=arc_end, xytext=arc_start,
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=4.5,
                            connectionstyle='arc3,rad=0.4', shrinkA=0, shrinkB=0),
            zorder=3)
ix = (arc_start[0] + arc_end[0]) / 2 - 0.5
iy = max(arc_start[1], arc_end[1]) + 1.3
ax.text(ix, iy,
        f"间接效应 a1×b1 = {p['ind1']*100:+.3f}pp ***\n"
        f"中介占比 7.4%\n"
        f"95%CI [{p['ind1_lo']*100:+.3f}, {p['ind1_hi']*100:+.3f}]pp",
        ha='center', va='center', fontsize=10, fontweight='bold', color=GREEN,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor=GREEN, linewidth=2, alpha=0.95), zorder=8)

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
    plt.Line2D([0],[0], color=GRAY, lw=2.0, linestyle='dashed', label='n.s. (不显著)'),
    plt.Line2D([0],[0], color=GREEN, lw=4, label='间接效应路径'),
]
ax.legend(handles=leg_items, loc='lower right', fontsize=9.5,
          framealpha=0.92, edgecolor='#D1D5DB', ncol=3,
          bbox_to_anchor=(0.98, 0.03))

ax.set_title('因果中介路径图：金额提及 → 驳回判决的传导机制',
             fontsize=15, fontweight='bold', pad=25)

fig.tight_layout(pad=0.5)
fig.savefig(os.path.join(OUTPUT, 'fig_mediation_path.png'), dpi=200)
plt.close(fig)
print("✅ fig_mediation_path.png 精确路径图已生成")


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
    ['间接 a1×b1',  'T→M1→Y',               '—', f'{p["ind1"]*100:+.3f}',
     f'[{p["ind1_lo"]*100:+.3f}, {p["ind1_hi"]*100:+.3f}]', '***'],
    ['间接 a2×b2',  'T→M2→Y',               '—', '≈0', '—', 'n.s.'],
    ['中介占比',     '(a1×b1)/c',            '—', '7.4%', '—', '—'],
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
    # 显著行: 第5列(结论)红色
    if rows[i][5] == '***':
        tbl[i, 5].set_text_props(color=RED, fontweight='bold')
        tbl[i, 0].set_facecolor('#FEF2F2')
    if rows[i][5] == 'PSM后n.s.':
        tbl[i, 5].set_text_props(color=GRAY)

tbl.scale(1.0, 1.9)
ax.set_title('表：因果中介效应分解', fontsize=14, fontweight='bold', pad=10, y=1.06)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT, 'fig_mediation_table.png'), dpi=200)
plt.close(fig)
print("✅ fig_mediation_table.png 已生成")
print("全部完成")
