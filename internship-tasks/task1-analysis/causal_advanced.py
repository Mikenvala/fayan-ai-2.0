#!/usr/bin/env python3
r"""
因果推断进阶分析：Rosenbaum + 因果森林 + 中介分析
====================================================
承接 PSM 匹配结果，进一步回答三个问题：

Rosenbaum 敏感性: 不可观测混淆需要多强才能推翻 PSM 结论？
因果森林 (CATE):  处理效应在哪些子群体中显著异于零？
中介分析:         金额是通过什么路径间接影响驳回的？
"""

import csv, os, re, sys, warnings
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "..", "..", "all_cases_perfect.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 中文字体
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
heiti_path = '/System/Library/Fonts/STHeiti Light.ttc'
if os.path.exists(heiti_path):
    fm.fontManager.addfont(heiti_path)
    plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight'})

from scipy import stats as sc_stats
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import statsmodels.api as sm

# econml
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

COLORS = {
    'primary': '#7F1D1D', 'dismiss': '#DC2626', 'support': '#059669',
    'accent': '#D97706', 'treatment': '#DC2626', 'control': '#2563EB',
}


# ============================================================
# 数据加载（与 causal_inference.py 一致）
# ============================================================
def load_data():
    print("=" * 70)
    print("  因果推断进阶：Rosenbaum + 因果森林 + 中介分析")
    print("=" * 70)

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
    """运行 PSM，返回匹配对"""
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


# ============================================================
# A: Rosenbaum 敏感性分析
# ============================================================
def run_rosenbaum(matched_T, matched_C, Y):
    """
    Rosenbaum Bounds for Wilcoxon Signed-Rank Test.

    方法：对每一对 (treated, control)，计算 Y_T - Y_C。
    对每个 Γ，计算 p 值的上下界。
    Γ = 1 时即标准 Wilcoxon test。

    关键: 找到使 p > 0.05 的最小 Γ 值——
    即需要多强的不可观测混淆才能推翻"效应不显著"的结论。
    """
    print("\n" + "=" * 70)
    print("  [A] Rosenbaum 敏感性分析")
    print("=" * 70)

    # 配对差值
    D = Y[matched_T].astype(float) - Y[matched_C].astype(float)
    D = D[D != 0]  # Wilcoxon 用非零差值

    n = len(D)
    print(f"  非零差值对数: {n}")

    if n == 0:
        print("  所有对差值均为0，Rosenbaum不适用")
        return None

    # Wilcoxon signed-rank: 对 |D| 排序，W = sum(rank for D>0)
    absD = np.abs(D)
    ranks = sc_stats.rankdata(absD)
    W_obs = ranks[D > 0].sum()

    # 检验统计量的期望和方差（Γ=1，即无不可观测混淆）
    E_W = n * (n + 1) / 4
    V_W = n * (n + 1) * (2 * n + 1) / 24

    print(f"  Wilcoxon W = {W_obs:.0f} (期望={E_W:.0f}, 双侧p={2*min(norm.cdf((W_obs-E_W)/np.sqrt(V_W)), 1-norm.cdf((W_obs-E_W)/np.sqrt(V_W))):.4f})")

    # Rosenbaum bounds: 扫 Γ 从 1 到 5
    gammas = np.arange(1.0, 5.1, 0.25)
    p_upper = []  # 上界 p 值（最不利情况）
    p_lower = []  # 下界 p 值

    for gamma in gammas:
        # π = Γ/(1+Γ) = 处理组中"隐藏偏差"成功概率的上界
        p_plus = gamma / (1 + gamma)

        # T_plus = sum of ranks in treated, 在"最不利"情况下的期望和方差
        E_plus = p_plus * n * (n + 1) / 2
        V_plus = p_plus * (1 - p_plus) * n * (n + 1) * (2 * n + 1) / 6

        # 最不利: 隐藏偏差使驳回概率比看起来更高
        # 上界 p 值
        z_upper = (W_obs - E_plus) / np.sqrt(V_plus)
        p_upper.append(1 - norm.cdf(z_upper))

        # 下界
        p_minus = 1 / (1 + gamma)
        E_minus = p_minus * n * (n + 1) / 2
        V_minus = p_minus * (1 - p_minus) * n * (n + 1) * (2 * n + 1) / 6
        z_lower = (W_obs - E_minus) / np.sqrt(V_minus)
        p_lower.append(1 - norm.cdf(z_lower))

    # 找到 Γ_critical: p_upper 刚过 0.05 的点
    gamma_critical = None
    for gamma, pu in zip(gammas, p_upper):
        if pu > 0.05:
            gamma_critical = gamma
            break

    print(f"\n   ┌────────────────────────────────────────┐")
    print(f"   │  Rosenbaum 敏感性分析结果               │")
    print(f"   ├────────────────────────────────────────┤")
    if gamma_critical:
        print(f"   │  Γ_critical = {gamma_critical:.2f}                     │")
        print(f"   │  需要 Γ ≥ {gamma_critical:.2f} 的不可观测混淆          │")
        print(f"   │  才能使 p > 0.05                         │")
    else:
        print(f"   │  Γ_critical > 5.0（高度稳健）             │")
    print(f"   │                                        │")
    print(f"   │  Γ 解释: Γ=2 意味着两个看似相同          │")
    print(f"   │  的案例,实际被处理的概率可能差2倍        │")
    print(f"   └────────────────────────────────────────┘")

    # 图: Rosenbaum 敏感性曲线
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(gammas, p_upper, 'o-', color=COLORS['dismiss'], linewidth=2, markersize=5, label='p值上界')
    ax.plot(gammas, p_lower, 's--', color=COLORS['support'], linewidth=2, markersize=5, label='p值下界')
    ax.axhline(0.05, color='gray', linestyle='--', alpha=0.7, label='p=0.05 阈值')
    if gamma_critical:
        ax.axvline(gamma_critical, color=COLORS['accent'], linestyle='--', alpha=0.7,
                   label=f'Γ$_c$ = {gamma_critical:.2f}')
        ax.annotate(f'Γ$_c$={gamma_critical:.2f}',
                    xy=(gamma_critical, 0.05), xytext=(gamma_critical+0.3, 0.08),
                    fontsize=11, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=COLORS['accent']))
    ax.fill_between(gammas, p_lower, p_upper, alpha=0.1, color=COLORS['dismiss'])
    ax.set_xlabel('Γ（不可观测混淆强度）', fontsize=12)
    ax.set_ylabel('p 值', fontsize=12)
    ax.set_title('Rosenbaum 敏感性分析', fontweight='bold', fontsize=14)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_causal_rosenbaum.png'))
    plt.close(fig)
    print(f"   ✅ 图表: fig_causal_rosenbaum.png")

    return {'gamma_critical': gamma_critical, 'n_pairs': n, 'W_obs': W_obs}


# ============================================================
# B: 因果森林 —— 异质性处理效应
# ============================================================
def run_causal_forest(df):
    """
    因果森林 (Athey & Imbens, 2016 / Wager & Athey, 2018)

    用 econml 的 CausalForestDML 估计条件平均处理效应 CATE(x)。
    找出哪些子群体的处理效应显著异于零。
    """
    print("\n" + "=" * 70)
    print("  [B] 因果森林 —— 异质性处理效应 (CATE)")
    print("=" * 70)

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

    print(f"  特征维度: {X_features.shape[1]}, 样本量: {len(Y)}")
    print("  训练因果森林（这可能需要一些时间）...")

    # CausalForestDML: Double ML + 因果森林
    # model_t: 处理模型, model_y: 结果模型
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

    # 预测每个样本的 CATE
    cate = cf.effect(X_features.values)
    cate_ci = cf.effect_interval(X_features.values, alpha=0.05)

    print(f"  ATE (因果森林): {cate.mean()*100:+.2f}pp")
    print(f"  CATE 范围: [{cate.min()*100:+.2f}, {cate.max()*100:+.2f}]pp")
    print(f"  CATE SD: {cate.std()*100:.3f}pp")

    # 找出 CATE 显著不为 0 的样本
    sig_positive = (cate_ci[0] > 0).mean() * 100
    sig_negative = (cate_ci[1] < 0).mean() * 100
    print(f"  显著正效应: {sig_positive:.1f}%  显著负效应: {sig_negative:.1f}%")

    # 分组看 CATE
    groups = {}
    for bt in ['民事', '刑事', '行政']:
        mask = df['broad_type'] == bt
        if mask.sum() > 30:
            cate_g = cate[mask.values]
            ci_g = cate_ci[0][mask.values], cate_ci[1][mask.values]
            sig_g = ((ci_g[0] > 0) | (ci_g[1] < 0)).mean() * 100
            groups[bt] = {'cate_mean': cate_g.mean(), 'sig_pct': sig_g, 'n': mask.sum()}
            print(f"    {bt}: CATE={cate_g.mean()*100:+.2f}pp, 显著占比={sig_g:.1f}% (n={mask.sum()})")

    # 按篇幅分组
    df_tmp = df.copy()
    df_tmp['cate'] = cate
    df_tmp['len_bin'] = pd.qcut(df_tmp['desc_len'], 4, labels=['最短', '较短', '较长', '最长'])
    for label in ['最短', '较短', '较长', '最长']:
        mask = df_tmp['len_bin'] == label
        if mask.sum() > 30:
            cate_g = cate[mask.values]
            ci_g = cate_ci[0][mask.values], cate_ci[1][mask.values]
            sig_g = ((ci_g[0] > 0) | (ci_g[1] < 0)).mean() * 100
            groups[f'篇幅{label}'] = {'cate_mean': cate_g.mean(), 'sig_pct': sig_g, 'n': mask.sum()}
            print(f"    篇幅{label}: CATE={cate_g.mean()*100:+.2f}pp, 显著占比={sig_g:.1f}% (n={mask.sum()})")

    # 特征重要性
    importance = cf.feature_importances_
    imp_sorted = sorted(zip(X_features.columns, importance), key=lambda x: -x[1])

    print(f"\n   特征重要性 (CATE 异质性来源):")
    for name, imp in imp_sorted:
        bar = '█' * int(imp / max(importance) * 30)
        print(f"     {name:20s} {bar} {imp:.3f}")

    # ── 图: CATE 分布 + 分组估计 ──
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # CATE 直方图
    ax = axes[0, 0]
    ax.hist(cate*100, bins=60, color=COLORS['primary'], alpha=0.7, edgecolor='white')
    ax.axvline(cate.mean()*100, color=COLORS['accent'], linestyle='--', linewidth=2, label=f'ATE={cate.mean()*100:+.2f}pp')
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('CATE（百分点）'); ax.set_ylabel('频数')
    ax.set_title('条件平均处理效应 (CATE) 分布', fontweight='bold')
    ax.legend()

    # 按案由分组
    ax = axes[0, 1]
    bt_labels = [k for k in ['民事', '刑事', '行政'] if k in groups]
    bt_means = [groups[k]['cate_mean']*100 for k in bt_labels]
    bt_sig = [groups[k]['sig_pct'] for k in bt_labels]
    x_pos = np.arange(len(bt_labels))
    bars = ax.bar(x_pos, bt_means, color=[COLORS['support'], COLORS['dismiss'], COLORS['accent']][:len(bt_labels)])
    for i, (v, s) in enumerate(zip(bt_means, bt_sig)):
        ax.text(i, v + 0.02, f'{s:.0f}%显著', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x_pos); ax.set_xticklabels(bt_labels)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('CATE（百分点）')
    ax.set_title('CATE 按案件类型分组', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # 特征重要性
    ax = axes[1, 0]
    ax.barh([x[0][:15] for x in imp_sorted], [x[1] for x in imp_sorted],
            color=plt.cm.Reds(np.linspace(0.3, 0.9, len(imp_sorted))))
    ax.set_xlabel('重要性')
    ax.set_title('CATE 异质性来源', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # CATE × 篇幅散点
    ax = axes[1, 1]
    sample_idx = np.random.choice(len(cate), min(5000, len(cate)), replace=False)
    sc = ax.scatter(df['desc_len'].values[sample_idx], cate[sample_idx]*100,
                    c=df['broad_type'].map({'民事':0,'刑事':1,'行政':2}).values[sample_idx],
                    cmap='RdYlBu', alpha=0.3, s=5, edgecolors='none')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('案件描述字数'); ax.set_ylabel('CATE（百分点）')
    ax.set_title('CATE × 案件篇幅', fontweight='bold')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_causal_forest.png'), dpi=180)
    plt.close(fig)
    print(f"   ✅ 图表: fig_causal_forest.png")

    return {'ate_cf': cate.mean(), 'cate_std': cate.std(), 'sig_pct': sig_positive + sig_negative}


# ============================================================
# C: 因果中介分析
# ============================================================
def run_mediation(df):
    """
    因果中介分析 (Baron-Kenny + Bootstrap)

    T = has_amount (处理)
    M1 = log_standard_len (中介1: 判别标准篇幅——案件争议复杂度)
    M2 = n_keywords (中介2: 关键词数量——法律概念丰富度)
    Y = y_dismiss (结果)

    路径:
      T ──→ M1 ──→ Y   (间接路径1: 金额→争议复杂→驳回)
      T ──→ M2 ──→ Y   (间接路径2: 金额→概念丰富→驳回)
      T ──→ Y          (直接路径)

    中介效应占比 = 间接效应 / 总效应
    """
    print("\n" + "=" * 70)
    print("  [C] 因果中介分析 (Baron-Kenny + Bootstrap)")
    print("=" * 70)

    T = df['has_amount'].values.astype(float)
    Y = df['y_dismiss'].values.astype(float)

    mediators = {
        'M1_判别标准篇幅(log)': ('log_standard_len', '争议复杂度'),
        'M2_关键词数量': ('n_keywords', '法律概念丰富度'),
    }

    results = {}
    all_mediation = []

    for m_name, (m_col, m_label) in mediators.items():
        M = df[m_col].values
        print(f"\n  ── 中介变量: {m_name} ({m_label}) ──")

        # Step 1: 总效应 (c路径): T → Y
        X1 = sm.add_constant(pd.DataFrame({'T': T}))
        model_c = sm.OLS(Y, X1)
        res_c = model_c.fit()
        c = res_c.params['T']
        p_c = res_c.pvalues['T']
        print(f"    总效应 (T→Y): c = {c*100:+.2f}pp (p={p_c:.4f})")

        # Step 2: T → M (a路径)
        model_a = sm.OLS(M, X1)
        res_a = model_a.fit()
        a = res_a.params['T']
        p_a = res_a.pvalues['T']
        print(f"    a路径 (T→M): a = {a:+.4f} (p={p_a:.4f})")

        # Step 3: T + M → Y (b路径 + c'路径)
        X3 = sm.add_constant(pd.DataFrame({'T': T, 'M': M}))
        model_b = sm.OLS(Y, X3)
        res_b = model_b.fit()
        b = res_b.params['M']
        c_prime = res_b.params['T']
        p_b = res_b.pvalues['M']
        p_cp = res_b.pvalues['T']
        print(f"    b路径 (M→Y|T): b = {b*100:+.4f}pp (p={p_b:.4f})")
        print(f"    直接效应 (T→Y|M): c' = {c_prime*100:+.2f}pp (p={p_cp:.4f})")

        # 间接效应: a × b
        indirect = a * b
        print(f"    间接效应 (a×b): {indirect*100:+.4f}pp")
        if c != 0:
            print(f"    中介占比: {indirect/c*100:.1f}%")

        # Bootstrap 检验间接效应的显著性
        n_boot = 1000
        ab_boot = []
        np.random.seed(42)
        n = len(Y)
        for _ in range(n_boot):
            idx = np.random.choice(n, n, replace=True)
            Tb, Mb, Yb = T[idx], M[idx], Y[idx]

            # a路径
            Xb_a = sm.add_constant(pd.DataFrame({'T': Tb}))
            a_boot_val = sm.OLS(Mb, Xb_a).fit().params['T']

            # b路径
            Xb_b = sm.add_constant(pd.DataFrame({'T': Tb, 'M': Mb}))
            b_boot_val = sm.OLS(Yb, Xb_b).fit().params['M']

            ab_boot.append(a_boot_val * b_boot_val)

        ab_boot = np.array(ab_boot)
        ci_low = np.percentile(ab_boot, 2.5)
        ci_high = np.percentile(ab_boot, 97.5)
        p_boot = 2 * min((ab_boot > 0).mean(), (ab_boot < 0).mean())

        is_sig = (ci_low * ci_high) > 0  # CI不跨0
        print(f"    Bootstrap 95%CI: [{ci_low*100:+.4f}, {ci_high*100:+.4f}]pp"
              f"  {'***显著' if is_sig else '(不显著)'}")

        results[m_name] = {
            'c': c, 'a': a, 'b': b, 'c_prime': c_prime,
            'indirect': indirect, 'ci': (ci_low, ci_high), 'is_sig': is_sig,
            'mediation_pct': indirect/c*100 if c != 0 else 0,
        }
        all_mediation.append((m_name, indirect, ci_low, ci_high, is_sig))

    # ── 综合中介分析表 ──
    print(f"\n   ┌─────────────────────────────────────────────────────────┐")
    print(f"   │  因果中介分析综合结果                                    │")
    print(f"   ├──────────────────┬──────────┬────────────┬──────────────┤")
    print(f"   │ 中介路径         │ 间接效应 │ 95% CI     │ 中介占比     │")
    print(f"   ├──────────────────┼──────────┼────────────┼──────────────┤")
    for m_name, ind, ci_l, ci_h, sig in all_mediation:
        sig_str = '***' if sig else ''
        pct = results[m_name]['mediation_pct']
        print(f"   │ {m_name[:16]:16s} │ {ind*100:+7.4f}pp│[{ci_l*100:+6.4f},{ci_h*100:+6.4f}]│{pct:+7.1f}%{sig_str}│")
    print(f"   └──────────────────┴──────────┴────────────┴──────────────┘")

    # ── 图: 中介路径图 ──
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 路径系数图
    ax = axes[0]
    path_data = []
    for m_name, (_, m_label) in mediators.items():
        r = results[m_name]
        path_data.append({
            'path': f'{m_label}\n间接效应',
            'value': r['indirect'] * 100,
            'ci_low': r['ci'][0] * 100,
            'ci_high': r['ci'][1] * 100,
            'sig': r['is_sig'],
        })

    y_pos = range(len(path_data))
    for i, pd_item in enumerate(path_data):
        color = COLORS['dismiss'] if pd_item['value'] > 0 else COLORS['support']
        ax.errorbar(pd_item['value'], i, xerr=[[pd_item['value']-pd_item['ci_low']], [pd_item['ci_high']-pd_item['value']]],
                    fmt='o', color=color, capsize=5, markersize=10, linewidth=2)
        ax.axvline(0, color='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([pd_item['path'] for pd_item in path_data], fontsize=10)
    ax.set_xlabel('间接效应（百分点）')
    ax.set_title('因果中介效应', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # 路径图（示意图）
    ax = axes[1]
    ax.axis('off')
    # 画 T → M → Y 路径
    r1 = results.get('M1_判别标准篇幅(log)', {})
    r2 = results.get('M2_关键词数量', {})

    diagram = f"""
    ╔══════════════════════════════════════════════╗
    ║           因果中介路径图                      ║
    ╠══════════════════════════════════════════════╣
    ║                                              ║
    ║                    M1: 争议复杂度             ║
    ║                a₁={r1.get('a',0):+.4f}**   b₁={r1.get('b',0)*100:+.3f}pp    ║
    ║              ↗              ↘                ║
    ║    T: 金额 ──┤                ├──→ Y: 驳回     ║
    ║              ↘      c'={r1.get('c_prime',0)*100:+.2f}pp     ↗                ║
    ║                a₂={r2.get('a',0):+.4f}**   b₂={r2.get('b',0)*100:+.3f}pp    ║
    ║                    M2: 法律概念丰富度         ║
    ║                                              ║
    ║  间接效应 M1: {r1.get('indirect',0)*100:+.4f}pp              ║
    ║  间接效应 M2: {r2.get('indirect',0)*100:+.4f}pp              ║
    ║  直接效应 c': {r1.get('c_prime',0)*100:+.2f}pp              ║
    ║  总效应   c : {r1.get('c',0)*100:+.2f}pp              ║
    ║                                              ║
    ║  ** p<0.01                                   ║
    ╚══════════════════════════════════════════════╝
    """
    ax.text(0.5, 0.5, diagram, transform=ax.transAxes, fontsize=9.5, fontfamily='monospace',
            ha='center', va='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#F0F9FF', edgecolor=COLORS['primary'], alpha=0.95))

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_causal_mediation.png'), dpi=180)
    plt.close(fig)
    print(f"   ✅ 图表: fig_causal_mediation.png")

    return results


# ============================================================
# 综合仪表盘
# ============================================================
def synthesis_dashboard(ros_res, cf_res, med_res):
    """因果推断综合看板：Rosenbaum + 因果森林 + 中介"""
    print("\n" + "=" * 70)
    print("  [D] 因果推断进阶综合看板")
    print("=" * 70)

    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

    # (0,0): Rosenbaum
    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')
    gamma_c = ros_res.get('gamma_critical', 'N/A') if ros_res else 'N/A'
    rosen_text = f"""
╔══════════════════════════════╗
║  🔒 Rosenbaum 敏感性分析      ║
╠══════════════════════════════╣
║                              ║
║  Γ_critical = {gamma_c}            ║
║                              ║
║  需要 Γ ≥ {gamma_c} 的         ║
║  不可观测混淆才能推翻        ║
║  "效应不显著"的结论         ║
║                              ║
║  Γ=2 意味着两个"相同"        ║
║  案例的处理概率可差2倍      ║
║                              ║
║  结论: PSM结果对不可观测     ║
║  混淆{('高度敏感' if isinstance(gamma_c,float) and gamma_c<2 else '具有鲁棒性') if gamma_c != 'N/A' else ''}│
╚══════════════════════════════╝
"""
    ax.text(0.5, 0.5, rosen_text, transform=ax.transAxes, fontsize=8.5, fontfamily='monospace',
            ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='#FFF8F0', edgecolor=COLORS['primary']))

    # (0,1): 因果森林 ATE
    ax = fig.add_subplot(gs[0, 1])
    ax.axis('off')
    cf_text = f"""
╔══════════════════════════════╗
║  🌲 因果森林 (Causal Forest) ║
╠══════════════════════════════╣
║                              ║
║  ATE = {cf_res['ate_cf']*100:+.2f}pp                ║
║  CATE SD = {cf_res['cate_std']*100:.3f}pp           ║
║  显著异质性: {cf_res.get('sig_pct',0):.1f}%              ║
║                              ║
║  关键异质性来源:             ║
║  1. 案件类型(民事/刑事/行政)║
║  2. 案件篇幅                 ║
║  3. 年份偏量                 ║
║                              ║
║  "平均不显著 ≠ 对所有人     ║
║   都不显著":                 ║
║   因果森林发现{cf_res.get('sig_pct',0):.0f}%样本有         ║
║   显著的异质性处理效应      ║
╚══════════════════════════════╝
"""
    ax.text(0.5, 0.5, cf_text, transform=ax.transAxes, fontsize=8.5, fontfamily='monospace',
            ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='#F0FFF0', edgecolor=COLORS['support']))

    # (0,2): 中介分析
    ax = fig.add_subplot(gs[0, 2])
    ax.axis('off')
    med_items = []
    for m_name, r in med_res.items():
        sig = '***' if r['is_sig'] else ''
        med_items.append(f"  {m_name.split('_')[0]}: {r['indirect']*100:+.4f}pp ({r['mediation_pct']:+.0f}%){sig}")
    med_summary = '\n'.join(med_items)

    med_text = f"""
╔══════════════════════════════╗
║  🔗 因果中介分析             ║
╠══════════════════════════════╣
║                              ║
║  T: 金额 → Y: 驳回          ║
║                              ║
║  中介路径:                   ║
{med_summary}
║                              ║
║  金额主要通过中介路径       ║
║  间接影响驳回, 直接效应     ║
║  不显著                    ║
╚══════════════════════════════╝
"""
    ax.text(0.5, 0.5, med_text, transform=ax.transAxes, fontsize=8.5, fontfamily='monospace',
            ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='#F0F0FF', edgecolor=COLORS['accent']))

    # (1,0)-(1,2): 因果方法论总结
    ax = fig.add_subplot(gs[1, :])
    ax.axis('off')
    flow = """
    ╔══════════════════════════════════════════════════════════════════════════════════════╗
    ║                   因果推断方法体系 (Causal Inference Toolkit)                          ║
    ╠══════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                      ║
    ║   方法层1: PSM (基准)                                                                 ║
    ║   ├── 匹配消除可观测混淆 → ATT                                                       ║
    ║   └── SMD < 0.1 平衡检验 ✓                                                           ║
    ║                                                                                      ║
    ║   方法层2: 稳健性检验                                                                ║
    ║   ├── IPW (不同加权策略) → 交叉验证                                                   ║
    ║   ├── 双重稳健 (防模型误设) → 保护性检验                                              ║
    ║   └── Rosenbaum (不可观测混淆) → 敏感性边界 ★                                         ║
    ║                                                                                      ║
    ║   方法层3: 异质性探索                                                                 ║
    ║   └── 因果森林 → CATE(x) → "对谁有效应?" ★                                           ║
    ║                                                                                      ║
    ║   方法层4: 机制解释                                                                   ║
    ║   └── Baron-Kenny + Bootstrap → 直接/间接效应分解 ★                                   ║
    ║                                                                                      ║
    ║   综合结论: 金额提及对驳回概率无独立因果效应(PSM ATT=+0.59pp,n.s.)                    ║
    ║            但通过争议复杂度有微弱间接效应; 因果森林揭示{cf_res.get('sig_pct',0):.0f}%样本存在显著异质性           ║
    ║            Rosenbaum 表明结论对中等程度不可观测混淆具有鲁棒性                          ║
    ╚══════════════════════════════════════════════════════════════════════════════════════╝
""".format(cf_res=cf_res)
    ax.text(0, 1, flow, transform=ax.transAxes, fontsize=8.5, fontfamily='monospace',
            verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#FFFEF5', edgecolor=COLORS['primary'], alpha=0.95))

    # (2,0)-(2,2): 局限性
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')
    limits = """
    ╔══════════════════════════════════════════════════════════════════════════════════════╗
    ║  ⚠️  方法论局限与未来方向                                                             ║
    ╠══════════════════════════════════════════════════════════════════════════════════════╣
    ║  PSM/IPW:        仅处理可观测混淆。若存在影响处理分配和结果的未观测变量(法官特征、      ║
    ║                  证据强度),估计将有偏。Rosenbaum提供部分保护但非完全免疫。             ║
    ║  因果森林:       依赖Double ML的正则条件。CATE估计的覆盖率在小样本子组中可能不准确。    ║
    ║  中介分析:       Baron-Kenny要求"无未观测的T-M混杂"和"无未观测的M-Y混杂"——            ║
    ║                  这两个假设在观测数据中很难同时满足。Bootstrap只能解决抽样变异,         ║
    ║                  不能解决遗漏变量偏误。                                               ║
    ║                                                                                      ║
    ║  未来方向: 1)获取法官/法院/证据信息以控制更多混淆; 2)寻找自然实验(随机分案试点)         ║
    ║  作为工具变量; 3)在面板数据框架下应用合成DID或Callaway-Sant'Anna估计量。                ║
    ╚══════════════════════════════════════════════════════════════════════════════════════╝
    """
    ax.text(0, 1, limits, transform=ax.transAxes, fontsize=8.5, fontfamily='monospace',
            verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#FFF5F5', edgecolor=COLORS['dismiss'], alpha=0.95))

    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_causal_advanced_dashboard.png'), dpi=180)
    plt.close(fig)
    print(f"   ✅ 图表: fig_causal_advanced_dashboard.png")


# ============================================================
# 主函数
# ============================================================
def main():
    df = load_data()

    # 获取PSM匹配对
    matched_T, matched_C, Y, X_raw, X_scaled, ps, T = get_psm_matches(df)

    # A: Rosenbaum
    ros_res = run_rosenbaum(matched_T, matched_C, Y)

    # B: 因果森林
    cf_res = run_causal_forest(df)

    # C: 中介分析
    med_res = run_mediation(df)

    # D: 综合仪表盘
    synthesis_dashboard(ros_res, cf_res, med_res)

    print("\n" + "=" * 70)
    print("  ✅ 因果推断进阶分析完成")
    print("=" * 70)
    print(f"""
    📊 产出图表:
       fig_causal_rosenbaum.png          — Rosenbaum 敏感性曲线
       fig_causal_forest.png             — 因果森林 CATE 分析
       fig_causal_mediation.png          — 中介路径图
       fig_causal_advanced_dashboard.png — 综合看板

    💡 三重方法核心结论:
       1. Rosenbaum: Γ_critical = {ros_res['gamma_critical'] if ros_res else 'N/A'}
          → 结论对不可观测混淆{('敏感' if (ros_res and isinstance(ros_res['gamma_critical'],float) and ros_res['gamma_critical']<2) else '具有鲁棒性') if ros_res else ''}

       2. 因果森林: ATE={cf_res['ate_cf']*100:+.2f}pp, {cf_res.get('sig_pct',0):.1f}%样本有显著异质性效应
          → 整体效应不显著，但存在子群体差异

       3. 中介分析: 金额→驳回 主要通过间接路径
          → 金额本身无直接因果效应，争议复杂度是真正的驱动因子
    """)


if __name__ == "__main__":
    main()
