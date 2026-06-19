#!/usr/bin/env python3
"""
修复论文4张插图：错位、缺失、渲染问题
=====================================
图1: 四层因果推断研究框架 → 用matplotlib图形替换ASCII艺术
图2: Rosenbaum敏感性曲线 → 改进标注位置
图3: 因果森林分析 → 修复颜色映射和布局
图4: 因果中介路径图 → 修复坐标对齐(offset_angle bug)
"""

import os, warnings, re, csv
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, ConnectionPatch
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import stats as sc_stats
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import statsmodels.api as sm

# ── 路径设置 ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)
CSV_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'all_cases_perfect.csv')

# ── 中文字体 ──
heiti_path = '/System/Library/Fonts/STHeiti Light.ttc'
if os.path.exists(heiti_path):
    fm.fontManager.addfont(heiti_path)
    plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 150})

# ── 颜色 ──
RED = '#DC2626'
GREEN = '#059669'
ORANGE = '#D97706'
BLUE = '#2563EB'
DARK = '#1F2937'
GRAY = '#9CA3AF'
LIGHT_RED = '#FEF2F2'
LIGHT_GREEN = '#ECFDF5'
LIGHT_ORANGE = '#FFF7ED'
LIGHT_BLUE = '#EFF6FF'
PRIMARY = '#7F1D1D'

print("=" * 70)
print("  论文插图修复脚本")
print("=" * 70)


# ================================================================
# 数据加载（与 causal_advanced.py 保持一致）
# ================================================================
def load_data():
    print("\n[0/4] 加载数据...")
    with open(CSV_PATH, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read().replace('\x00', '')
    reader = csv.DictReader(content.splitlines())
    rows = []
    for r in reader:
        fn = r.get('文件名', '')
        desc = r.get('案件描述', '')
        standard = r.get('判别标准', '')
        judgment = r.get('判决结果', '')
        kws = []
        for i in range(1, 11):
            kw = r.get(f'关键词_{i:02d}', '').strip()
            if kw: kws.append(kw)
        yr = re.search(r'\[(\d{4})\]', fn)
        year = int(yr.group(1)) if yr and 1990 <= int(yr.group(1)) <= 2030 else None
        has_amount = bool(re.search(r'(\d+)\s*(?:万|元|千元)', desc[:500]))
        if not judgment: y_dismiss = 0
        elif '驳回' in judgment and '维持' not in judgment and '支持' not in judgment:
            y_dismiss = 1
        elif '驳回' in judgment: y_dismiss = 0
        else: y_dismiss = 0
        CRIM_KW = {'罪','盗窃','抢劫','杀人','故意伤害','诈骗','强奸','走私'}
        ADMIN_KW = {'行政','行政复议','行政诉讼','国家赔偿','征地','拆迁'}
        crim_s = sum(1 for kw in kws if any(ck in kw for ck in CRIM_KW))
        admin_s = sum(1 for kw in kws if any(ak in kw for ak in ADMIN_KW))
        if crim_s > admin_s and crim_s > 0: broad = '刑事'
        elif admin_s > crim_s and admin_s > 0: broad = '行政'
        else: broad = '民事'
        rows.append({
            'year': year, 'broad_type': broad,
            'desc_len': len(desc), 'standard_len': len(standard),
            'n_keywords': len(kws), 'has_amount': int(has_amount),
            'y_dismiss': y_dismiss,
        })
    df = pd.DataFrame(rows).dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    df['log_desc_len'] = np.log1p(df['desc_len'])
    df['log_standard_len'] = np.log1p(df['standard_len'])
    print(f"  有效记录: {len(df)}")
    return df


def get_psm_matches(df):
    """PSM匹配（与causal_advanced.py一致）"""
    T = df['has_amount'].values
    Y = df['y_dismiss'].values
    X_raw = pd.DataFrame({
        'log_desc_len': df['log_desc_len'],
        'log_standard_len': df['log_standard_len'],
        'n_keywords': df['n_keywords'],
        'is_civil': (df['broad_type'] == '民事').astype(int),
        'is_criminal': (df['broad_type'] == '刑事').astype(int),
        'year_offset': df['year'] - 2014,
    })
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    ps_model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=5000)
    ps_model.fit(X_scaled, T)
    ps = ps_model.predict_proba(X_scaled)[:, 1]
    ps = np.clip(ps, 0.01, 0.99)
    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    nn.fit(ps[control_idx].reshape(-1, 1))
    matched_T, matched_C = [], []
    caliper = 0.05
    for i in treated_idx:
        if ps[i] < 0.01 or ps[i] > 0.99: continue
        dist, idx = nn.kneighbors([[ps[i]]])
        if dist[0][0] <= caliper:
            matched_T.append(i)
            matched_C.append(control_idx[idx[0][0]])
    matched_T, matched_C = np.array(matched_T), np.array(matched_C)
    print(f"  PSM匹配: {len(matched_T)} 对")
    return matched_T, matched_C, Y, X_raw, X_scaled, ps, T


# ================================================================
# 图1: 四层因果推断研究框架（替换ASCII艺术）
# ================================================================
def make_figure1():
    """
    用matplotlib Patch对象构建四层框架图，
    替换原来无法正确渲染的ASCII艺术框线。
    """
    print("\n[1/4] 生成图1: 四层因果推断研究框架...")

    fig, ax = plt.subplots(figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 18)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── 配色 ──
    LAYER_COLORS = {
        1: ('#FEF2F2', RED, '第一层：基准效应估计'),
        2: ('#FFF7ED', ORANGE, '第二层：稳健性评估'),
        3: ('#EFF6FF', BLUE, '第三层：异质性探索'),
        4: ('#ECFDF5', GREEN, '第四层：机制解释'),
    }

    # ── 标题 ──
    ax.text(10, 17.5, '四层因果推断研究框架', ha='center', va='center',
            fontsize=18, fontweight='bold', color=DARK)
    ax.text(10, 16.8,
            '从"统计关联" → "因果识别" → "稳健性评估" → "异质性探索" → "机制解释"',
            ha='center', va='center', fontsize=11, color=GRAY)

    # ── 左侧：数据输入 ──
    data_box = FancyBboxPatch((0.5, 6.5), 2.5, 5, boxstyle='round,pad=0.2',
                               facecolor='#F5F5F5', edgecolor=DARK, linewidth=2)
    ax.add_patch(data_box)
    ax.text(1.75, 9, '观测数据\n10,241条\n裁判文书', ha='center', va='center',
            fontsize=10, fontweight='bold', color=DARK)
    ax.text(1.75, 7.5, '(2014-2025)', ha='center', va='center', fontsize=8, color=GRAY)

    # 数据 → 第一层 箭头
    ax.annotate('', xy=(3.8, 14.5), xytext=(3.0, 10.5),
                arrowprops=dict(arrowstyle='->', color=DARK, lw=2.5))
    ax.annotate('', xy=(3.8, 7.5), xytext=(3.0, 8.5),
                arrowprops=dict(arrowstyle='->', color=DARK, lw=2.5))

    # ── 四层：从上到下排列 ──
    layer_defs = [
        (1,  4.0, 13.5, 'PSM 倾向得分匹配',
         'Logistic估计倾向得分 → 1:1最近邻匹配\nSMD平衡性检验 → ATT估计\n→ 回答："有没有效应？"',
         'PSM后ATT=+0.59pp (p=0.46)\n金额提及对驳回无独立因果效应'),
        (2,  4.0, 9.5, 'IPW + 双重稳健 + Rosenbaum',
         'IPW稳定化权重 → 双重稳健估计\nRosenbaum Γ参数扫描\n→ 回答："结论有多稳？"',
         'IPW ATE=+1.49pp | DR ATE=+1.64pp\n交叉验证一致 | Γcritical=1.00'),
        (3,  4.0, 5.5, '因果森林 (Causal Forest)',
         'Double ML框架 → 200棵诚实树\nCATE(x) 异质性估计\n→ 回答："对谁有效应？"',
         '10.9%样本显著 | 行政案件CATE=+3.04pp\n篇幅Q3 CATE=+3.02pp (倒U型)'),
        (4,  4.0, 1.5, '因果中介分析',
         'Baron-Kenny三步法 + Bootstrap\nT→M→Y 效应分解\n→ 回答："效应怎么发生的？"',
         '争议复杂度中介7.4%\n间接效应+0.41pp [0.22,0.61]pp'),
    ]

    LAYER_W, LAYER_H = 14.5, 3.2
    for lid, lx, ly, title, method_text, result_text in layer_defs:
        bg_color, accent_color, layer_label = LAYER_COLORS[lid]

        # 层背景
        rect = FancyBboxPatch((lx, ly), LAYER_W, LAYER_H,
                               boxstyle='round,pad=0.15',
                               facecolor=bg_color, edgecolor=accent_color,
                               linewidth=2.5, alpha=0.9, zorder=2)
        ax.add_patch(rect)

        # 层号标签（左侧竖排）
        ax.text(lx + 0.25, ly + LAYER_H/2, layer_label, ha='left', va='center',
                fontsize=8, fontweight='bold', color=accent_color, rotation=90, zorder=3)

        # 层标题
        ax.text(lx + 1.0, ly + LAYER_H - 0.4, title, ha='left', va='top',
                fontsize=13, fontweight='bold', color=accent_color, zorder=3)

        # 方法描述（左半部分）
        ax.text(lx + 1.0, ly + LAYER_H/2 - 0.1, method_text, ha='left', va='center',
                fontsize=8.5, color=DARK, zorder=3, linespacing=1.5)

        # 结果发现（右半部分，带背景框）
        result_rect = FancyBboxPatch((lx + 8.0, ly + 0.35), 6.2, LAYER_H - 0.7,
                                      boxstyle='round,pad=0.1',
                                      facecolor='white', edgecolor=accent_color,
                                      linewidth=1.2, alpha=0.85, zorder=3)
        ax.add_patch(result_rect)
        ax.text(lx + 11.1, ly + LAYER_H/2, f'📊 {result_text}', ha='center', va='center',
                fontsize=8.5, color=DARK, zorder=4, linespacing=1.6)

    # ── 层间箭头 ──
    for i in range(1, 4):
        _, _, y_top = LAYER_COLORS[i]
        _, _, y_bot = LAYER_COLORS[i+1]
        # 找到当前层底部和下一层顶部
        ly_top_bottom = [ly for lid, lx, ly, _, _, _ in layer_defs if lid == i][0]
        ly_bot_top = [ly + LAYER_H for lid, lx, ly, _, _, _ in layer_defs if lid == i+1][0]
        ax.annotate('', xy=(11.25, ly_bot_top + 0.1), xytext=(11.25, ly_top_bottom - 0.1),
                    arrowprops=dict(arrowstyle='->', color=GRAY, lw=2.5, connectionstyle='arc3,rad=0'),
                    zorder=5)

    # ── 底部：方法论说明 ──
    note_text = ('⚠️ 方法论局限与未来方向\n'
                 'PSM/IPW: 仅处理可观测混淆 | 因果森林: 小样本子组CATE方差较大 | '
                 '中介分析: Baron-Kenny假设在观测数据中难以全满足\n'
                 '未来: 获取法官/法院/证据信息 | Legal-BERT深层语义特征 | '
                 '合成DID或Callaway-Sant\'Anna估计量')
    note_rect = FancyBboxPatch((4.0, 0.0), 14.5, 1.2, boxstyle='round,pad=0.1',
                                facecolor='#FFF5F5', edgecolor=RED, linewidth=1.2, alpha=0.9)
    ax.add_patch(note_rect)
    ax.text(11.25, 0.6, note_text, ha='center', va='center', fontsize=7.8,
            color=DARK, zorder=3, linespacing=1.4)

    # ── 左下角图例 ──
    ax.text(0.3, 0.5, '© 王志达  南京农业大学', fontsize=7, color=GRAY, alpha=0.6)

    fig.tight_layout(pad=0.5)
    outpath = os.path.join(FIGURES_DIR, 'fig_causal_advanced_dashboard.png')
    fig.savefig(outpath, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ {outpath}")


# ================================================================
# 图2: Rosenbaum敏感性曲线（改进标注位置）
# ================================================================
def make_figure2(matched_T, matched_C, Y):
    """
    Rosenbaum敏感性分析。改进：
    - 当Γcritical=1.00时，标注不遮挡曲线
    - 使用更清晰的图表样式
    """
    print("\n[2/4] 生成图2: Rosenbaum敏感性曲线...")

    D = Y[matched_T].astype(float) - Y[matched_C].astype(float)
    D = D[D != 0]
    n = len(D)

    absD = np.abs(D)
    ranks = sc_stats.rankdata(absD)
    W_obs = ranks[D > 0].sum()

    gammas = np.arange(1.0, 5.1, 0.25)
    p_upper, p_lower = [], []

    for gamma in gammas:
        p_plus = gamma / (1 + gamma)
        E_plus = p_plus * n * (n + 1) / 2
        V_plus = p_plus * (1 - p_plus) * n * (n + 1) * (2 * n + 1) / 6
        z_upper = (W_obs - E_plus) / np.sqrt(V_plus)
        p_upper.append(1 - norm.cdf(z_upper))

        p_minus = 1 / (1 + gamma)
        E_minus = p_minus * n * (n + 1) / 2
        V_minus = p_minus * (1 - p_minus) * n * (n + 1) * (2 * n + 1) / 6
        z_lower = (W_obs - E_minus) / np.sqrt(V_minus)
        p_lower.append(1 - norm.cdf(z_lower))

    gamma_critical = None
    for gamma, pu in zip(gammas, p_upper):
        if pu > 0.05:
            gamma_critical = gamma
            break

    print(f"  非零差值对: {n}")
    print(f"  Γ_critical = {gamma_critical}")

    # ── 绘图 ──
    fig, ax = plt.subplots(figsize=(10, 6.5))

    ax.plot(gammas, p_upper, 'o-', color=RED, linewidth=2.5, markersize=6,
            label='p值上界 (最不利情况)', zorder=5)
    ax.plot(gammas, p_lower, 's--', color=GREEN, linewidth=2.5, markersize=6,
            label='p值下界 (最有利情况)', zorder=5)
    ax.fill_between(gammas, p_lower, p_upper, alpha=0.12, color=RED, zorder=2)
    ax.axhline(0.05, color=GRAY, linestyle='--', alpha=0.8, linewidth=1.5,
               label='p = 0.05 显著性阈值', zorder=3)

    if gamma_critical:
        ax.axvline(gamma_critical, color=ORANGE, linestyle='--', alpha=0.9,
                   linewidth=1.8, zorder=3)

        # 关键改进：标注位置根据 gamma_critical 的值动态调整
        if gamma_critical <= 1.25:
            # 边界情况：标注放在图右上角而非曲线旁
            ax.text(0.97, 0.92,
                    f'Γcritical = {gamma_critical:.2f}\n'
                    f'(结论对不可观测混淆敏感)',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=12, fontweight='bold', color=RED,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                              edgecolor=RED, linewidth=2, alpha=0.95), zorder=10)
        else:
            ax.annotate(f'Γ$_c$ = {gamma_critical:.2f}',
                        xy=(gamma_critical, 0.05),
                        xytext=(gamma_critical + 0.5, 0.12),
                        fontsize=12, fontweight='bold', color=ORANGE,
                        arrowprops=dict(arrowstyle='->', color=ORANGE, lw=2),
                        zorder=10)

    ax.set_xlabel('Γ（不可观测混淆强度）', fontsize=13, fontweight='bold')
    ax.set_ylabel('p 值', fontsize=13, fontweight='bold')
    ax.set_title('Rosenbaum 敏感性分析', fontsize=15, fontweight='bold', pad=12)
    ax.legend(fontsize=10.5, loc='center right', framealpha=0.9)
    ax.grid(alpha=0.25)
    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(-0.02, max(p_upper) * 1.15)

    # 添加解释性文本框
    explanation = (
        'Γ 解读：\n'
        'Γ=1 → 无不可观测混淆\n'
        'Γ=2 → 两个看似相同的案例，\n'
        '       实际被处理的概率可差2倍\n'
        f'当前 Γcritical={gamma_critical:.2f} → 即使极微弱的\n'
        '不可观测混淆也能使p>0.05'
    )
    ax.text(0.03, 0.97, explanation, transform=ax.transAxes, ha='left', va='top',
            fontsize=8.5, color=DARK, linespacing=1.4,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFBEB',
                      edgecolor=ORANGE, linewidth=1, alpha=0.9), zorder=8)

    fig.tight_layout(pad=0.5)
    outpath = os.path.join(FIGURES_DIR, 'fig_causal_rosenbaum.png')
    fig.savefig(outpath, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ {outpath}")


# ================================================================
# 图3: 因果森林分析（修复颜色映射和布局）
# ================================================================
def make_figure3(df):
    """
    因果森林CATE分析。改进：
    - 修复broad_type颜色映射（strip空格）
    - 增加子图间距避免标签重叠
    - 散点图添加颜色图例
    """
    print("\n[3/4] 生成图3: 因果森林分析...")

    from econml.dml import CausalForestDML

    T = df['has_amount'].values
    Y = df['y_dismiss'].values.astype(float)
    X_features = pd.DataFrame({
        'log_desc_len': df['log_desc_len'],
        'log_standard_len': df['log_standard_len'],
        'n_keywords': df['n_keywords'],
        'is_civil': (df['broad_type'] == '民事').astype(int),
        'is_criminal': (df['broad_type'] == '刑事').astype(int),
        'year_offset': df['year'] - 2014,
    })

    print("  训练因果森林（约需1-2分钟）...")
    cf = CausalForestDML(
        model_t=GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        model_y=GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        discrete_treatment=True,
        n_estimators=200,
        min_samples_leaf=50,
        max_depth=6,
        random_state=42,
        n_jobs=1,
    )
    cf.fit(Y, T, X=X_features.values)

    cate = cf.effect(X_features.values)
    cate_ci = cf.effect_interval(X_features.values, alpha=0.05)

    print(f"  ATE (因果森林): {cate.mean()*100:+.2f}pp")
    sig_pct = ((cate_ci[0] > 0) | (cate_ci[1] < 0)).mean() * 100
    print(f"  显著异质性占比: {sig_pct:.1f}%")

    # ── 分组CATE ──
    groups = {}
    # 修复：strip空格，确保正确匹配
    df['broad_type_clean'] = df['broad_type'].str.strip()
    for bt in ['民事', '刑事', '行政']:
        mask = df['broad_type_clean'] == bt
        if mask.sum() > 30:
            cate_g = cate[mask.values]
            ci_g = cate_ci[0][mask.values], cate_ci[1][mask.values]
            sig_g = ((ci_g[0] > 0) | (ci_g[1] < 0)).mean() * 100
            groups[bt] = {'cate_mean': cate_g.mean(), 'sig_pct': sig_g, 'n': mask.sum()}

    # 按篇幅分组
    df['cate'] = cate
    df['len_bin'] = pd.qcut(df['desc_len'], 4, labels=['Q1(最短)', 'Q2(较短)', 'Q3(较长)', 'Q4(最长)'])
    for label in ['Q1(最短)', 'Q2(较短)', 'Q3(较长)', 'Q4(最长)']:
        mask = df['len_bin'] == label
        if mask.sum() > 30:
            cate_g = cate[mask.values]
            ci_g = cate_ci[0][mask.values], cate_ci[1][mask.values]
            sig_g = ((ci_g[0] > 0) | (ci_g[1] < 0)).mean() * 100
            groups[f'篇幅{label}'] = {'cate_mean': cate_g.mean(), 'sig_pct': sig_g, 'n': mask.sum()}

    # 特征重要性
    importance = cf.feature_importances_
    imp_sorted = sorted(zip(X_features.columns, importance), key=lambda x: -x[1])

    # ── 绘图：增大尺寸和间距 ──
    fig = plt.figure(figsize=(22, 16))

    # (0,0): CATE 直方图
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.hist(cate*100, bins=60, color=PRIMARY, alpha=0.75, edgecolor='white', linewidth=0.3)
    ax1.axvline(cate.mean()*100, color=ORANGE, linestyle='--', linewidth=2.5,
                label=f'ATE = {cate.mean()*100:+.2f}pp')
    ax1.axvline(0, color='black', linewidth=1.5)
    ax1.set_xlabel('CATE（百分点）', fontsize=11)
    ax1.set_ylabel('频数', fontsize=11)
    ax1.set_title('CATE 分布', fontweight='bold', fontsize=13)
    ax1.legend(fontsize=9, loc='upper right')

    # (0,1): 按案件类型分组
    ax2 = fig.add_subplot(2, 3, 2)
    bt_labels = [k for k in ['民事', '刑事', '行政'] if k in groups]
    bt_means = [groups[k]['cate_mean']*100 for k in bt_labels]
    bt_sig = [groups[k]['sig_pct'] for k in bt_labels]
    bt_colors = [BLUE, RED, ORANGE][:len(bt_labels)]
    x_pos = np.arange(len(bt_labels))
    bars = ax2.bar(x_pos, bt_means, color=bt_colors, edgecolor='white', linewidth=1.2, width=0.6)
    for i, (v, s, n) in enumerate(zip(bt_means, bt_sig, [groups[k]['n'] for k in bt_labels])):
        ax2.text(i, v + 0.04, f'{s:.1f}%\n(n={n})', ha='center', fontsize=8.5, fontweight='bold',
                 color=DARK)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bt_labels, fontsize=11)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_ylabel('CATE（百分点）', fontsize=11)
    ax2.set_title('CATE 按案件类型', fontweight='bold', fontsize=13)
    ax2.grid(axis='y', alpha=0.3)

    # (0,2): 按篇幅分组
    ax3 = fig.add_subplot(2, 3, 3)
    len_labels = [k for k in ['篇幅Q1(最短)', '篇幅Q2(较短)', '篇幅Q3(较长)', '篇幅Q4(最长)'] if k in groups]
    len_means = [groups[k]['cate_mean']*100 for k in len_labels]
    len_sig = [groups[k]['sig_pct'] for k in len_labels]
    x_pos2 = np.arange(len(len_labels))
    # 渐变色突出Q3最高
    len_colors = [BLUE, '#60A5FA', RED, '#FCA5A5'][:len(len_labels)]
    bars2 = ax3.bar(x_pos2, len_means, color=len_colors, edgecolor='white', linewidth=1.2, width=0.6)
    for i, (v, s) in enumerate(zip(len_means, len_sig)):
        color = RED if i == 2 else DARK  # Q3 highlight
        ax3.text(i, v + 0.04, f'{s:.1f}%', ha='center', fontsize=8.5,
                 fontweight='bold', color=color)
    ax3.set_xticks(x_pos2)
    ax3.set_xticklabels([l.replace('篇幅', '') for l in len_labels], fontsize=9.5)
    ax3.axhline(0, color='black', linewidth=0.8)
    ax3.set_ylabel('CATE（百分点）', fontsize=11)
    ax3.set_title('CATE 按案件篇幅（倒U型）', fontweight='bold', fontsize=13)
    ax3.grid(axis='y', alpha=0.3)

    # (1,0): 特征重要性
    ax4 = fig.add_subplot(2, 3, 4)
    feat_names_short = {
        'log_standard_len': '判别标准篇幅(log)',
        'log_desc_len': '案件描述篇幅(log)',
        'year_offset': '年份偏量',
        'is_civil': '民事案件',
        'is_criminal': '刑事案件',
        'n_keywords': '关键词数量',
    }
    names = [feat_names_short.get(x[0], x[0]) for x in imp_sorted]
    values = [x[1] for x in imp_sorted]
    colors_imp = plt.cm.Reds(np.linspace(0.35, 0.95, len(imp_sorted)))
    ax4.barh(names, values, color=colors_imp, edgecolor='white', height=0.6)
    ax4.set_xlabel('特征重要性', fontsize=11)
    ax4.set_title('CATE 异质性来源', fontweight='bold', fontsize=13)
    ax4.grid(axis='x', alpha=0.3)
    ax4.invert_yaxis()

    # (1,1): CATE × 篇幅散点图（修复颜色映射）
    ax5 = fig.add_subplot(2, 3, 5)
    sample_idx = np.random.choice(len(cate), min(5000, len(cate)), replace=False)

    # 修复：使用干净的数值映射和颜色条
    bt_map = {'民事': 0, '刑事': 1, '行政': 2}
    df_sample = df.iloc[sample_idx]
    bt_values = df_sample['broad_type_clean'].map(bt_map).fillna(0).values
    cate_sample = cate[sample_idx] * 100
    desc_len_sample = df['desc_len'].values[sample_idx]

    sc = ax5.scatter(desc_len_sample, cate_sample,
                     c=bt_values, cmap='RdYlBu_r', alpha=0.35, s=8, edgecolors='none')
    ax5.axhline(0, color='black', linewidth=0.8)
    ax5.set_xlabel('案件描述字数', fontsize=11)
    ax5.set_ylabel('CATE（百分点）', fontsize=11)
    ax5.set_title('CATE × 案件篇幅', fontweight='bold', fontsize=13)
    ax5.grid(alpha=0.3)

    # 添加颜色图例
    cbar = plt.colorbar(sc, ax=ax5, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['民事', '刑事', '行政'], fontsize=9)
    cbar.set_label('案件类型', fontsize=10)

    # (1,2): 关键发现文本
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    # 构建分组摘要
    bt_summary = '\n'.join([
        f"  • {bt}: CATE={groups[bt]['cate_mean']*100:+.2f}pp  "
        f"({groups[bt]['sig_pct']:.1f}%显著, n={groups[bt]['n']})"
        for bt in ['民事', '刑事', '行政'] if bt in groups
    ])
    len_summary = '\n'.join([
        f"  • {l.replace('篇幅','')}: CATE={groups[l]['cate_mean']*100:+.2f}pp  "
        f"({groups[l]['sig_pct']:.1f}%显著)"
        for l in ['篇幅Q1(最短)', '篇幅Q2(较短)', '篇幅Q3(较长)', '篇幅Q4(最长)'] if l in groups
    ])

    summary_text = (
        f'🌲 因果森林分析摘要\n'
        f'{"─" * 34}\n\n'
        f'整体ATE = {cate.mean()*100:+.2f}pp\n'
        f'CATE范围: [{cate.min()*100:+.1f}, {cate.max()*100:+.1f}]pp\n'
        f'CATE SD = {cate.std()*100:.2f}pp\n'
        f'显著异质性: {sig_pct:.1f}% 样本\n\n'
        f'按案件类型:\n{bt_summary}\n\n'
        f'按案件篇幅:\n{len_summary}\n\n'
        f'💡 关键发现:\n'
        f'整体ATT不显著，但10.9%样本\n'
        f'存在显著异质性效应。行政案件\n'
        f'和中等篇幅案件效应最强，呈\n'
        f'倒U型分布。'
    )
    ax6.text(0.05, 0.98, summary_text, transform=ax6.transAxes,
             ha='left', va='top', fontsize=9, fontfamily='monospace',
             color=DARK, linespacing=1.3,
             bbox=dict(boxstyle='round,pad=0.7', facecolor='#F0F9FF',
                       edgecolor=BLUE, linewidth=1.8, alpha=0.95))

    fig.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.0)
    outpath = os.path.join(FIGURES_DIR, 'fig_causal_forest.png')
    fig.savefig(outpath, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ {outpath}")


# ================================================================
# 图4: 因果中介路径图（修复offset_angle bug和坐标对齐）
# ================================================================
def make_figure4():
    """
    精确坐标布局的因果中介路径图。

    修复：
    1. BUG1: offset_angle=90 → 改为 0（正确：垂直偏移）
    2. BUG2: 弧线标注坐标微调避免与盒子重叠
    3. BUG3: 箭头精确连接到盒子边缘
    """
    print("\n[4/4] 生成图4: 因果中介路径图...")

    # ── 参数（与原文一致）──
    p = {
        'c': 0.0556, 'c_prime': 0.0515,
        'a1': 0.0684, 'a2': -0.0000,
        'b1': 0.05994, 'b2': 0.03662,
        'ind1': 0.00410, 'ind1_lo': 0.00223, 'ind1_hi': 0.00608,
    }

    # ── 盒尺寸 ──
    BW, BH = 3.0, 1.4
    HALF_W, HALF_H = BW/2, BH/2

    # ── 盒中心坐标（扩大间距以减少重叠）──
    BOX_T  = np.array([2.0, 4.5])
    BOX_M1 = np.array([7.0, 7.2])
    BOX_M2 = np.array([7.0, 1.8])
    BOX_Y  = np.array([12.0, 4.5])

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

    # ── 箭头定义 ──
    ARROWS = [
        # a1: T → M1
        ('a', box_edge(BOX_T, 'tr'), box_edge(BOX_M1, 'left'),
         f"a₁ = {p['a1']:+.4f}***", RED, 3.0, 'solid'),
        # a2: T → M2
        ('b', box_edge(BOX_T, 'br'), box_edge(BOX_M2, 'left'),
         f"a₂ ≈ 0 (n.s.)", GRAY, 2.2, 'dashed'),
        # b1: M1 → Y
        ('c', box_edge(BOX_M1, 'right'), box_edge(BOX_Y, 'tl'),
         f"b₁ = {p['b1']*100:+.3f}pp***", RED, 3.0, 'solid'),
        # b2: M2 → Y
        ('d', box_edge(BOX_M2, 'right'), box_edge(BOX_Y, 'bl'),
         f"b₂ = {p['b2']*100:+.3f}pp***", RED, 3.0, 'solid'),
        # c': T → Y 直接效应
        ('e', box_edge(BOX_T, 'right'), box_edge(BOX_Y, 'left'),
         f"c' = {p['c_prime']*100:+.2f}pp\n(PSM匹配后不显著)", GRAY, 2.5, 'dashed'),
    ]

    # ── FIXED: 修复 mid_vec 函数 ──
    # 原来 offset_angle=90 旋转了垂直向量导致标签平行于箭头
    # 正确的做法：offset_angle=0 保持垂直方向
    def mid_point_perpendicular(p1, p2, dist=0.65):
        """返回箭头中点的垂直上方/下方位置"""
        mx = (p1[0] + p2[0]) / 2
        my = (p1[1] + p2[1]) / 2
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = np.sqrt(dx*dx + dy*dy)
        if length < 1e-9:
            return mx, my + dist
        # 垂直单位向量（逆时针旋转90°）
        nx = -dy / length
        ny = dx / length
        return mx + nx * dist, my + ny * dist

    # ── 绘图 ──
    fig, ax = plt.subplots(figsize=(18, 11))
    ax.set_xlim(-1, 14.5)
    ax.set_ylim(-1, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── 画变量盒 ──
    boxes = [
        (BOX_T,  '提及涉案金额\n(处理变量 T)',  LIGHT_RED, RED),
        (BOX_M1, '争议复杂度 M₁\n判别标准篇幅(log)', LIGHT_ORANGE, '#EA580C'),
        (BOX_M2, '概念丰富度 M₂\n关键词数量', LIGHT_ORANGE, '#EA580C'),
        (BOX_Y,  '驳回判决\n(结果变量 Y)',  LIGHT_GREEN, GREEN),
    ]
    for (cx, cy), text, fc, ec in boxes:
        rect = FancyBboxPatch(
            (cx - HALF_W, cy - HALF_H), BW, BH,
            boxstyle='round,pad=0.15',
            facecolor=fc, edgecolor=ec, linewidth=3.0, zorder=5
        )
        ax.add_patch(rect)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=11.5,
                fontweight='bold', color=DARK, zorder=6, linespacing=1.2)

    # ── 画箭头和标签（修复定位）──
    for _, start, end, label, color, lw, style in ARROWS:
        linestyle = style  # 'solid' or 'dashed'
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                    linestyle=linestyle, shrinkA=0, shrinkB=0),
                    zorder=4)

        # FIXED: 使用修正后的垂直偏移函数
        if 'b1' in label:
            # b1: 上方的箭头，标签放上方
            mx, my = mid_point_perpendicular(start, end, dist=0.60)
        elif 'b2' in label:
            # b2: 下方的箭头，标签放下方
            mx, my = mid_point_perpendicular(start, end, dist=-0.60)
        elif 'a2' in label:
            # a2: 下方箭头，标签放下方
            mx, my = mid_point_perpendicular(start, end, dist=-0.45)
        elif "c'" in label:
            # c': 水平箭头，标签放上方（手工定位最佳位置）
            mx = (BOX_T[0] + BOX_Y[0]) / 2
            my = BOX_T[1] + HALF_H + 1.0
        else:
            # a1: 默认上方
            mx, my = mid_point_perpendicular(start, end, dist=0.65)

        ax.text(mx, my, label, ha='center', va='center', fontsize=10,
                fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='none', alpha=0.94), zorder=7)

    # ── 间接效应弧线标注（修复坐标）──
    # 从T上方画一条弧到M1上方，表示间接效应路径
    arc_start = (BOX_T[0] + 0.8, BOX_T[1] + HALF_H + 0.3)
    arc_end   = (BOX_M1[0] - 0.8, BOX_M1[1] + HALF_H + 0.3)
    ax.annotate('', xy=arc_end, xytext=arc_start,
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=5.0,
                                connectionstyle='arc3,rad=0.45', shrinkA=0, shrinkB=0),
                zorder=3)

    # 弧线上方的标注
    ix = (arc_start[0] + arc_end[0]) / 2 - 0.3
    iy = max(arc_start[1], arc_end[1]) + 1.8
    ax.text(ix, iy,
            f'间接效应 a₁×b₁ = {p["ind1"]*100:+.3f}pp ***\n'
            f'中介占比 7.4%\n'
            f'95%CI [{p["ind1_lo"]*100:+.3f}, {p["ind1_hi"]*100:+.3f}]pp',
            ha='center', va='center', fontsize=10.5, fontweight='bold', color=GREEN,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor=GREEN, linewidth=2.5, alpha=0.96), zorder=8)

    # ── 总效应标注（下方居中）──
    tx, ty = BOX_T[0], BOX_T[1] - HALF_H - 0.8
    ax.plot([BOX_T[0], BOX_T[0]], [BOX_T[1] - HALF_H, ty + 0.2],
            color=GRAY, lw=2, linestyle='--', alpha=0.6, zorder=2)
    ax.text(BOX_T[0] + 2.5, ty - 0.05,
            f'总效应 c = {p["c"]*100:+.2f}pp ***',
            ha='center', va='center', fontsize=12, fontweight='bold', color=DARK,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#F3F4F6',
                      edgecolor=GRAY, linewidth=1.8), zorder=8)

    # ── M2不显著路径的灰化说明 ──
    ax.text(BOX_M2[0] + 2.2, BOX_M2[1] - HALF_H - 0.4,
            '(间接效应 ≈ 0，不显著)', ha='center', va='top',
            fontsize=8.5, color=GRAY, style='italic', zorder=8)

    # ── 图例 ──
    leg_items = [
        mpatches.Patch(color=LIGHT_RED, label='处理变量 (T)'),
        mpatches.Patch(color=LIGHT_ORANGE, label='中介变量 (M₁, M₂)'),
        mpatches.Patch(color=LIGHT_GREEN, label='结果变量 (Y)'),
        plt.Line2D([0], [0], color=RED, lw=3, label='p<0.001 (显著)'),
        plt.Line2D([0], [0], color=GRAY, lw=2.5, linestyle='dashed', label='n.s. (不显著)'),
        plt.Line2D([0], [0], color=GREEN, lw=4.5, label='显著间接效应路径'),
    ]
    ax.legend(handles=leg_items, loc='lower right', fontsize=10,
              framealpha=0.94, edgecolor='#D1D5DB', ncol=3,
              bbox_to_anchor=(0.98, 0.03))

    ax.set_title('因果中介路径图：金额提及 → 驳回判决的传导机制',
                 fontsize=15, fontweight='bold', pad=20)

    fig.tight_layout(pad=0.5)
    outpath = os.path.join(FIGURES_DIR, 'fig_mediation_path.png')
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ {outpath}")


# ================================================================
# 主流程
# ================================================================
def main():
    df = load_data()
    matched_T, matched_C, Y, X_raw, X_scaled, ps, T = get_psm_matches(df)

    make_figure1()  # 四层因果推断研究框架
    make_figure2(matched_T, matched_C, Y)  # Rosenbaum敏感性曲线
    make_figure3(df)  # 因果森林分析
    make_figure4()  # 因果中介路径图

    print("\n" + "=" * 70)
    print("  ✅ 全部4张插图修复完成！")
    print("=" * 70)
    print(f"""
  修复内容:
    图1 (fig_causal_advanced_dashboard.png):
      ✓ ASCII艺术框线 → matplotlib FancyBboxPatch
      ✓ 中文等宽字体错位 → 标准文本 + 补丁
      ✓ 添加层间箭头和结果摘要

    图2 (fig_causal_rosenbaum.png):
      ✓ Γcritical=1.00边界标注 → 右上角文本框
      ✓ 添加Γ解读说明

    图3 (fig_causal_forest.png):
      ✓ 修复 broad_type 颜色映射（strip空格）
      ✓ 增大子图间距（h_pad/w_pad=3/2）
      ✓ 添加散点图颜色图例
      ✓ 新增分析摘要面板

    图4 (fig_mediation_path.png):
      ✓ 修复 mid_vec offset_angle bug（90°→0°垂直偏移）
      ✓ 重新计算所有标注位置
      ✓ 改进弧线和总效应标注
      ✓ 增大盒间距避免重叠

  新图片已保存到: {FIGURES_DIR}/
    """)


if __name__ == "__main__":
    main()
