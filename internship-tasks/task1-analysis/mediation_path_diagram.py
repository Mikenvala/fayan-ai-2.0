import os, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

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

RS = '#DC2626'   # 显著箭头
RN = '#9CA3AF'   # 不显著箭头
RI = '#059669'   # 间接效应

# ============================================================
# 图A: 路径图
# ============================================================
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(-1, 12)
ax.set_ylim(-1, 9.5)
ax.axis('off')

def box(x, y, w, h, txt, fc, ec):
    r = mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h,
                                 boxstyle='round,pad=0.15',
                                 facecolor=fc, edgecolor=ec, linewidth=2.8)
    ax.add_patch(r)
    ax.text(x, y, txt, ha='center', va='center', fontsize=11.5, fontweight='bold', color='#1F2937')

def arrow(x1, y1, x2, y2, label, color, lw=2.5, fs=10, offset_y=0.2):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, shrinkA=10, shrinkB=10))
    mx, my = (x1+x2)/2, (y1+y2)/2 + offset_y
    ax.text(mx, my, label, ha='center', va='bottom', fontsize=fs,
            fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.92))

# 画4个变量盒
box(1.5, 4.5,  3.2, 1.4, '提及涉案金额\n（处理变量 T）',      '#FEF2F2', '#DC2626')
box(6.0, 7.0,  3.4, 1.4, '争议复杂度 M₁\n判别标准篇幅(log)',  '#FFF8F0', '#D97706')
box(6.0, 2.0,  3.4, 1.4, '概念丰富度 M₂\n关键词数量',         '#FFF8F0', '#D97706')
box(10., 4.5,  2.8, 1.4, '驳回判决\n（结果变量 Y）',          '#ECFDF5', '#059669')

# 箭头: a路径
arrow(3.1, 5.1, 4.3, 6.6, f"a₁ = {p['a1']:+.4f}***", RS, 2.8, 10.5)
arrow(3.1, 3.9, 4.3, 2.2, f"a₂ ≈ 0 (n.s.)", RN, 2.0, 9)

# 箭头: b路径
arrow(7.7, 7.0, 8.6, 5.1, f"b₁ = {p['b1']*100:+.3f}pp***", RS, 2.8, 10.5, 0.15)
arrow(7.7, 2.0, 8.6, 3.9, f"b₂ = {p['b2']*100:+.3f}pp***", RS, 2.8, 10.5, 0.15)

# 箭头: c' 直接效应 (虚线表示PSM后不显著)
ax.annotate('', xy=(8.6, 4.5), xytext=(3.1, 4.5),
            arrowprops=dict(arrowstyle='->', color=RN, lw=2.5,
                            linestyle='dashed', shrinkA=10, shrinkB=10))
ax.text(5.85, 5.2, f"c' = {p['c_prime']*100:+.2f}pp\n（PSM匹配后不显著）",
        ha='center', fontsize=9.5, fontweight='bold', color=RN,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=RN, alpha=0.9))

# 间接效应标注
ax.annotate('', xy=(8.5, 7.0), xytext=(4.5, 5.5),
            arrowprops=dict(arrowstyle='->', color=RI, lw=4.5,
                            connectionstyle='arc3,rad=0.35', shrinkA=8, shrinkB=8))
ax.text(7.2, 8.2,
        f"间接效应 = {p['ind1']*100:+.3f}pp ***\n"
        f"中介占比 7.4%\n"
        f"95%CI [{p['ind1_lo']*100:+.3f},{p['ind1_hi']*100:+.3f}]pp",
        ha='center', fontsize=10, fontweight='bold', color=RI,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=RI, alpha=0.93))

# 总效应
ax.text(1.5, 2.8, f"总效应: c = {p['c']*100:+.2f}pp ***",
        ha='center', fontsize=11, fontweight='bold', color='#374151',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F3F4F6', edgecolor='#9CA3AF'))
ax.plot([1.5, 1.5], [3.5, 3.1], color='#9CA3AF', lw=2, linestyle='--', alpha=0.5)

# 图例
leg = [
    mpatches.Patch(color='#FEF2F2', label='处理变量 (T)'),
    mpatches.Patch(color='#FFF8F0', label='中介变量 (M)'),
    mpatches.Patch(color='#ECFDF5', label='结果变量 (Y)'),
    plt.Line2D([0],[0], color=RS, lw=2.5, label='*** p<0.001'),
    plt.Line2D([0],[0], color=RN, lw=2.0, linestyle='--', label='n.s. (不显著)'),
    plt.Line2D([0],[0], color=RI, lw=4, label='间接效应路径'),
]
ax.legend(handles=leg, loc='lower right', fontsize=9, framealpha=0.9)

ax.set_title('因果中介路径图：金额提及 → 驳回判决的传导机制',
             fontsize=15, fontweight='bold', pad=22)
fig.tight_layout(pad=0.5)
fig.savefig(os.path.join(OUTPUT, 'fig_mediation_path.png'), dpi=200)
plt.close(fig)
print("✅ fig_mediation_path.png")


# ============================================================
# 图B: 效应分解表
# ============================================================
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.axis('off')

rows = [
    ['路径', '描述', '系数', '效应量(pp)', '95%CI', '结论'],
    ['总效应 c',    'T → Y',                '—', f'{p["c"]*100:+.2f}', '—', '***'],
    ['直接效应 c\'', 'T → Y | M₁, M₂',       '—', f'{p["c_prime"]*100:+.2f}', '—', 'PSM后n.s.'],
    ['a₁',          'T → M₁ (争议复杂度)',    f'{p["a1"]:+.4f}', '—', '—', '***'],
    ['a₂',          'T → M₂ (概念丰富度)',    '≈0', '—', '—', 'n.s.'],
    ['b₁',          'M₁ → Y | T',           f'{p["b1"]*100:+.3f}pp', '—', '—', '***'],
    ['b₂',          'M₂ → Y | T',           f'{p["b2"]*100:+.3f}pp', '—', '—', '***'],
    ['间接 a₁×b₁',  'T→M₁→Y',               '—', f'{p["ind1"]*100:+.3f}',
     f'[{p["ind1_lo"]*100:+.3f},{p["ind1_hi"]*100:+.3f}]', '***'],
    ['间接 a₂×b₂',  'T→M₂→Y',               '—', '≈0', '—', 'n.s.'],
    ['中介占比',     '(a₁×b₁)/c',            '—', '7.4%', '—', '—'],
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
    for j in range(6):
        if i % 2 == 0:
            tbl[i, j].set_facecolor('#F9FAFB')
    # 显著行高亮
    if rows[i][-1] == '***':
        tbl[i, 0].set_facecolor('#FEF2F2')
        tbl[i, 5].set_text_props(color='#DC2626', fontweight='bold')
    if rows[i][-1] == 'PSM后n.s.':
        tbl[i, 5].set_text_props(color='#9CA3AF')

tbl.scale(1.0, 1.9)
ax.set_title('因果中介效应分解表', fontsize=14, fontweight='bold', pad=10, y=1.06)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT, 'fig_mediation_table.png'), dpi=200)
plt.close(fig)
print("✅ fig_mediation_table.png")
print("两张图已全部生成")
