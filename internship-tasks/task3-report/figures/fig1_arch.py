import os
os.environ['MPLCONFIGDIR'] = os.path.dirname(os.path.abspath(__file__))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
for f in ['/System/Library/Fonts/STHeiti Medium.ttc', '/System/Library/Fonts/STHeiti Light.ttc']:
    if os.path.exists(f): fm.fontManager.addfont(f)
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'STHeiti', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'

from matplotlib.patches import FancyBboxPatch
C1='#1E3A5F'; C2='#2563EB'; C3='#7C3AED'; C4='#059669'; WHITE='#FFFFFF'

# Figure height extended to accommodate title gap
fig, ax = plt.subplots(figsize=(16, 10.5))
ax.set_xlim(-1, 16)
ax.set_ylim(-1, 12.2)  # +0.2 to give breathing room
ax.set_aspect('equal')
ax.axis('off')

# Title: was 11.3 → now 11.6 (clear of L1 top at 10.8)
TITLE_Y = 11.6
ax.text(7.5, TITLE_Y, '法眼AI 2.0 四层系统架构', ha='center', va='center',
        fontsize=22, fontweight='bold', color=C1)

# Layers: each shifted down by 0.3 from original
# Original: (9.2, 1.9), (7.0, 1.9), (4.5, 2.2), (2.2, 2.0)
# New:      (8.9, 1.9), (6.7, 1.9), (4.2, 2.2), (1.9, 2.0)
# Layer 1 top = 8.9+1.9=10.8, gap from title (11.6-10.8)=0.8 ✓
layers = [
    (8.9, 1.9, C1, [
        ('', ''),
        ('数据仪表盘', 'ECharts'),
        ('智能问答', 'ChatGPT风格'),
        ('模拟辩论', 'SSE流式'),
        ('报告生成', 'HTML + PDF')
    ]),
    (6.7, 1.9, C2, [
        ('', ''),
        ('POST /chat/sse', '智能对话'),
        ('POST /debate/sse', '辩论流式'),
        ('POST /report', '报告生成')
    ]),
    (4.2, 2.2, C3, [
        ('', ''),
        ('检索Agent', 'BM25+TF-IDF'),
        ('分析Agent', 'LLM法律推理'),
        ('校验Agent', '引用准确性'),
        ('辩论Agent', '原告/被告/法官')
    ]),
    (1.9, 2.0, C4, [
        ('', ''),
        ('裁判文书', '10,241 CSV'),
        ('TF-IDF索引', 'BM25Okapi'),
        ('SQLite/JSON', '关键词缓存')
    ]),
]

layer_titles = [
    ('前端层 (Presentation)', C1),
    ('API 层 (Application)', C2),
    ('业务逻辑层 (Business Logic)', C3),
    ('数据层 (Data)', C4),
]

for idx, (y, h, color, items) in enumerate(layers):
    x0, x1 = 0.2, 14.8
    # Background fill
    rect = FancyBboxPatch((x0, y), x1-x0, h, boxstyle='round,pad=0.15',
                           facecolor=color, edgecolor=color, linewidth=0, alpha=0.1, zorder=2)
    ax.add_patch(rect)
    # Border
    rect2 = FancyBboxPatch((x0, y), x1-x0, h, boxstyle='round,pad=0.15',
                            facecolor='none', edgecolor=color, linewidth=3, zorder=3)
    ax.add_patch(rect2)
    
    # Layer title badge (top-left inside the box)
    ltitle, lcolor = layer_titles[idx]
    # Small colored badge for layer name
    badge_w = 6.0
    badge_h = 0.55
    badge = FancyBboxPatch((x0 + 0.6, y + h - badge_h - 0.12), badge_w, badge_h,
                            boxstyle='round,pad=0.08', facecolor=color, edgecolor=color, linewidth=0, zorder=5)
    ax.add_patch(badge)
    ax.text(x0 + 0.6 + badge_w/2, y + h - badge_h/2 - 0.12, ltitle,
            ha='center', va='center', fontsize=11, fontweight='bold', color=WHITE, zorder=6)
    
    if idx == 0:
        # 前端层: 4 tab cards
        tw = 3.2; gap = 0.35; sx = x0 + 0.5
        for i, (name, desc) in enumerate(zip(items[1:], ['ECharts', 'ChatGPT风格', 'SSE流式', 'HTML + PDF'])):
            bx = sx + i*(tw+gap)
            card_h = h - 1.1
            card_y = y + 0.15
            box = FancyBboxPatch((bx, card_y), tw, card_h, boxstyle='round,pad=0.08',
                                  facecolor=WHITE, edgecolor=color, linewidth=1.8, alpha=0.95, zorder=4)
            ax.add_patch(box)
            ax.text(bx+tw/2, card_y + card_h*0.62, name, ha='center', va='center',
                    fontsize=12, fontweight='bold', color=color, zorder=5)
            ax.text(bx+tw/2, card_y + card_h*0.22, desc, ha='center', va='center',
                    fontsize=9.5, color='#6B7280', zorder=5)
    elif idx == 1:
        # API层: white text on blue bg
        for i, (name, desc) in enumerate(zip(items[1:], ['智能对话', '辩论流式', '报告生成'])):
            iy = y + h - 1.05 - i*0.45
            ax.text(x0+1.2, iy, name, ha='left', va='center',
                    fontsize=12, fontweight='bold', color=WHITE, zorder=5)
            ax.text(x0+7.2, iy, desc, ha='left', va='center',
                    fontsize=10.5, color=WHITE, alpha=0.9, zorder=5)
    elif idx == 2:
        # 业务层
        for i, (name, desc) in enumerate(zip(items[1:], ['BM25+TF-IDF', 'LLM法律推理', '引用准确性', '原告/被告/法官'])):
            iy = y + h - 0.95 - i*0.43
            ax.text(x0+1.2, iy, name, ha='left', va='center',
                    fontsize=12, fontweight='bold', color=color, zorder=5)
            ax.text(x0+7.8, iy, desc, ha='left', va='center',
                    fontsize=10.5, color='#374151', zorder=5)
    else:
        # 数据层
        for i, (name, desc) in enumerate(zip(items[1:], ['10,241 CSV', 'BM25Okapi', '关键词缓存'])):
            iy = y + h - 0.95 - i*0.52
            ax.text(x0+1.2, iy, name, ha='left', va='center',
                    fontsize=12, fontweight='bold', color=color, zorder=5)
            ax.text(x0+6.2, iy, desc, ha='left', va='center',
                    fontsize=10.5, color='#374151', zorder=5)

# Arrows between layers
for i in range(3):
    y_from = layers[i][0]
    y_to = layers[i+1][0] + layers[i+1][1]
    ax.annotate('', xy=(7.5, y_to+0.02), xytext=(7.5, y_from-0.02),
                arrowprops=dict(arrowstyle='<->', color='#9CA3AF', lw=2.5), zorder=2)

# Debug rectangles (comment out for production)
# ax.add_patch(plt.Rectangle((0, TITLE_Y-0.2), 15, 0.02, color='red', zorder=99))
# ax.add_patch(plt.Rectangle((0, layers[0][0]+layers[0][1]), 15, 0.02, color='red', zorder=99))

fig.savefig('image1_new.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('Done: image1_new.png')
print(f'  Title Y={TITLE_Y}')
print(f'  L1 top={layers[0][0]+layers[0][1]:.1f}, gap={TITLE_Y-(layers[0][0]+layers[0][1]):.2f}')
print(f'  L2 top={layers[1][0]+layers[1][1]:.1f}')
print(f'  L3 top={layers[2][0]+layers[2][1]:.1f}')
print(f'  L4 top={layers[3][0]+layers[3][1]:.1f}')
