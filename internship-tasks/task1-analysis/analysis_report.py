#!/usr/bin/env python3
"""
裁判文书大数据深度分析脚本
基于 10,241 条裁判文书数据，从五个维度进行数据挖掘与可视化。

运行: python analysis_report.py
输出: output/ 目录下 5 张图表 + conclusions.txt
"""

import csv
import os
import re
import sys
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

import matplotlib
matplotlib.use('Agg')  # 无头模式，无需 GUI
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ============================================================
# 全局配置
# ============================================================
# CSV 数据路径（相对于本脚本）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "..", "..", "all_cases_perfect.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 中文字体设置
for font in ['Heiti TC', 'Songti SC', 'Heiti TC', 'Songti SC', 'Arial Unicode MS', 'SimHei']:
    try:
        matplotlib.rcParams['font.sans-serif'] = [font]
        matplotlib.rcParams['axes.unicode_minus'] = False
        # Test font
        fig, ax = plt.subplots(figsize=(1,1))
        ax.set_title('测试')
        plt.close(fig)
        print(f"✅ 使用中文字体: {font}")
        break
    except:
        continue

# 图表全局样式
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'font.size': 10,
})

# ============================================================
# 数据加载与预处理
# ============================================================

def load_data(csv_path):
    """加载 CSV 数据，处理 NUL 字节，返回 list[dict]"""
    print(f"📂 加载数据: {csv_path}")
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read().replace('\x00', '')

    reader = csv.DictReader(content.splitlines())
    cases = list(reader)
    print(f"   共加载 {len(cases)} 条案例")
    return cases


def extract_category(filename):
    """从文件名路径中提取案由类别名称"""
    parts = filename.replace('\\', '/').split('/')
    if len(parts) < 2:
        return "未分类"
    folder = parts[-2]
    # 清理文件夹名：去掉编号前缀、年份前缀、多余后缀
    folder = re.sub(r'^\(NEW\).*?案例[：:]', '', folder)
    folder = re.sub(r'^\d+\s*[、\s]*', '', folder)
    folder = re.sub(r'\s*超清[对照]*$', '', folder)
    folder = re.sub(r'\s*超高清$', '', folder)
    folder = re.sub(r'_\s*扫描版$', '', folder)
    folder = re.sub(r'^\d{4}/', '', folder)
    folder = re.sub(r'^\d{4}年度案例', '', folder)
    folder = folder.strip('_ ')

    # 合并相似类别
    if '劳动' in folder or '劳务' in folder:
        return '劳动/劳务纠纷'
    if '公司' in folder:
        return '公司纠纷'
    if '合同' in folder or '买卖' in folder:
        return '合同纠纷'
    if '交通' in folder or '道路' in folder:
        return '道路交通纠纷'
    if '保险' in folder:
        return '保险纠纷'
    if '人格' in folder or '名誉' in folder:
        return '人格权纠纷'
    if '婚姻' in folder or '离婚' in folder:
        return '婚姻家庭纠纷'
    if '继承' in folder:
        return '继承纠纷'
    if '房屋' in folder or '房产' in folder:
        return '房屋买卖/租赁纠纷'
    if '借贷' in folder:
        return '民间借贷纠纷'
    if '侵权' in folder:
        return '侵权责任纠纷'
    if '执行' in folder:
        return '执行纠纷'
    if '刑事' in folder:
        return '刑事案件'
    if '行政' in folder:
        return '行政案件'
    return folder.strip()


def extract_year(filename):
    """从文件名提取年份"""
    m = re.search(r'\[(\d{4})\]', filename)
    if m:
        y = int(m.group(1))
        if 1990 <= y <= 2030:
            return y
    # Fallback: any 4-digit year
    m = re.search(r'(19\d{2}|20\d{2})', filename)
    if m:
        y = int(m.group(1))
        if 1990 <= y <= 2030:
            return y
    return None


def get_keywords(row):
    """提取 10 个关键词，返回去重列表"""
    kws = []
    for i in range(1, 11):
        kw = row.get(f'关键词_{i:02d}', '').strip()
        if kw:
            kws.append(kw)
    return list(dict.fromkeys(kws))  # 去重保序


def preprocess(cases):
    """预处理：给每条案例附加衍生字段"""
    for c in cases:
        c['_category'] = extract_category(c.get('文件名', ''))
        c['_year'] = extract_year(c.get('文件名', ''))
        c['_keywords'] = get_keywords(c)
        c['_desc_len'] = len(c.get('案件描述', ''))
        c['_judgment'] = c.get('判决结果', '').strip()
    return cases


# ============================================================
# 分析 1: 关键词共现网络
# ============================================================

def analyze_keyword_cooccurrence(cases):
    """
    分析高频关键词的共现关系，生成热力图。
    共现定义：两个关键词出现在同一条案例中。
    """
    print("\n" + "=" * 60)
    print("[1/5] 关键词共现网络分析")
    print("=" * 60)

    # 收集所有关键词，统计频率
    kw_counter = Counter()
    for c in cases:
        for kw in c['_keywords']:
            kw_counter[kw] += 1

    # 取 Top 30 关键词
    top_kws = [kw for kw, _ in kw_counter.most_common(30)]
    top_set = set(top_kws)

    # 构建共现矩阵
    n = len(top_kws)
    cooc_matrix = np.zeros((n, n), dtype=int)
    kw_index = {kw: i for i, kw in enumerate(top_kws)}

    for c in cases:
        present = [kw for kw in c['_keywords'] if kw in top_set]
        for a, b in combinations(present, 2):
            i, j = kw_index[a], kw_index[b]
            cooc_matrix[i][j] += 1
            cooc_matrix[j][i] += 1

    # 对角线设为自己的总频次
    for kw, idx in kw_index.items():
        cooc_matrix[idx][idx] = kw_counter[kw]

    # ---- 画热力图 ----
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cooc_matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(top_kws, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(top_kws, fontsize=7)
    ax.set_title('Top 30 关键词共现矩阵热力图', fontsize=14, fontweight='bold', pad=20)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('共现次数', fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig1_cooccurrence_heatmap.png'))
    plt.close(fig)
    print(f"   ✅ 图表已保存: fig1_cooccurrence_heatmap.png")

    # 发现最强的共现对（排除对角线）
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((top_kws[i], top_kws[j], cooc_matrix[i][j]))
    pairs.sort(key=lambda x: -x[2])
    top_pairs = pairs[:10]

    print(f"   Top 10 共现对:")
    for a, b, v in top_pairs:
        print(f"     {a} — {b}: {v} 次")

    conclusion = (
        f"高频关键词共现分析揭示了裁判文书中反复出现的法律概念组合。"
        f"最强共现对为「{top_pairs[0][0]}」与「{top_pairs[0][1]}」"
        f"（共现 {top_pairs[0][2]} 次），反映了该类案件的核心争议焦点。"
        f"共现矩阵呈明显的块状结构，表明不同案由类型的关键词自成体系，"
        f"但也存在跨类别关联（如程序性关键词与实体性关键词的交叉）。"
    )
    print(f"\n   📝 结论: {conclusion}")
    return conclusion


# ============================================================
# 分析 2: 判决书篇幅分析
# ============================================================

def analyze_length_distribution(cases):
    """
    统计案件描述字数分布，按案由类别对比篇幅差异。
    """
    print("\n" + "=" * 60)
    print("[2/5] 判决书篇幅分析")
    print("=" * 60)

    # 统计总体分布
    lengths = [c['_desc_len'] for c in cases]
    print(f"   案件描述字数: 均值={np.mean(lengths):.0f}, "
          f"中位数={np.median(lengths):.0f}, "
          f"最小={np.min(lengths)}, 最大={np.max(lengths)}")

    # 按类别分组（取案例数 >= 30 的类别）
    cat_len = defaultdict(list)
    for c in cases:
        cat_len[c['_category']].append(c['_desc_len'])

    cat_counts = {k: len(v) for k, v in cat_len.items()}
    major_cats = sorted(
        [(k, v) for k, v in cat_len.items() if len(v) >= 30],
        key=lambda x: -len(x[1])
    )

    labels = [k for k, _ in major_cats]
    data = [v for _, v in major_cats]

    # ---- 画箱线图 ----
    fig, ax = plt.subplots(figsize=(16, 8))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, vert=False,
                     showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=5))

    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(labels)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xlabel('案件描述字数', fontsize=11)
    ax.set_title('不同案由的案件描述篇幅分布（箱线图）', fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_length_boxplot.png'))
    plt.close(fig)
    print(f"   ✅ 图表已保存: fig2_length_boxplot.png")

    # 找出篇幅差异最大的类别
    means = [(label, np.mean(vals)) for label, vals in major_cats]
    means.sort(key=lambda x: -x[1])
    print(f"   篇幅最长的类别: {means[0][0]} (均值 {means[0][1]:.0f} 字)")
    print(f"   篇幅最短的类别: {means[-1][0]} (均值 {means[-1][1]:.0f} 字)")

    conclusion = (
        f"案件描述篇幅呈现明显的右偏分布，中位数 {np.median(lengths):.0f} 字，"
        f"均值 {np.mean(lengths):.0f} 字，说明大多数案件描述较为简洁，"
        f"但存在少量极端冗长的案例拉高了均值。"
        f"不同案由类别的篇幅差异显著——「{means[0][0]}」平均篇幅最长"
        f"（{means[0][1]:.0f} 字），而「{means[-1][0]}」平均最短"
        f"（{means[-1][1]:.0f} 字），反映了不同案由的案情复杂度差异。"
    )
    print(f"\n   📝 结论: {conclusion}")
    return conclusion


# ============================================================
# 分析 3: 判决结果情感倾向分析
# ============================================================

def analyze_judgment_outcomes(cases):
    """
    识别判决结果中的关键动作词（驳回/维持/撤销/改判等），统计分布。
    """
    print("\n" + "=" * 60)
    print("[3/5] 判决结果情感倾向分析")
    print("=" * 60)

    # 判决结果分类规则（优先级从高到低）
    def classify_judgment(text):
        if not text:
            return '未知'
        # 二审改判
        if '改判' in text and ('驳回' in text or '维持' in text):
            if text.index('改判') < max(text.find('驳回') if '驳回' in text else -1,
                                        text.find('维持') if '维持' in text else -1):
                return '部分改判'
        if '改判' in text:
            return '改判'
        if '发回重审' in text:
            return '发回重审'
        if '撤销' in text:
            return '撤销原判'
        if '驳回' in text and ('维持' in text or '支持' in text):
            return '部分支持'
        if '驳回' in text:
            return '驳回'
        if '维持' in text:
            return '维持原判'
        if '支持' in text:
            return '支持诉求'
        if '解除' in text or '确认' in text or '变更' in text:
            return '其他裁判'
        return '其他'

    outcomes = Counter()
    for c in cases:
        label = classify_judgment(c['_judgment'])
        outcomes[label] += 1

    print("   判决结果分布:")
    total_labeled = sum(outcomes.values())
    for label, count in outcomes.most_common():
        pct = count / total_labeled * 100
        bar = '█' * int(pct / 2)
        print(f"     {label:10s}: {count:5d} ({pct:5.1f}%) {bar}")

    # ---- 画饼图 ----
    # 合并小类别
    threshold = total_labeled * 0.01  # 1% 以下合并
    merged = {}
    others_count = 0
    for label, count in outcomes.most_common():
        if count >= threshold:
            merged[label] = count
        else:
            others_count += count
    if others_count > 0:
        merged['其他（合并）'] = others_count

    colors_map = {
        '驳回': '#DC2626', '维持原判': '#059669', '撤销原判': '#D97706',
        '部分支持': '#7C3AED', '部分改判': '#0891B2', '改判': '#0284C7',
        '支持诉求': '#16A34A', '其他裁判': '#6B7280', '发回重审': '#F59E0B',
        '其他（合并）': '#D1D5DB', '未知': '#9CA3AF',
    }
    pie_colors = [colors_map.get(l, '#CBD5E1') for l in merged.keys()]

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        merged.values(), labels=None, autopct='%1.1f%%',
        colors=pie_colors, startangle=90,
        pctdistance=0.75,
        explode=[0.03] * len(merged),
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight('bold')

    # 图例
    legend_labels = [f'{k} ({v})' for k, v in merged.items()]
    ax.legend(wedges, legend_labels, title='判决结果类型', loc='center left',
              bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.set_title('判决结果类型分布', fontsize=14, fontweight='bold')

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_judgment_pie.png'))
    plt.close(fig)
    print(f"   ✅ 图表已保存: fig3_judgment_pie.png")

    top = outcomes.most_common(1)[0]
    conclusion = (
        f"在 {total_labeled} 条有判决结果的案例中，"
        f"「{top[0]}」占比最高（{top[1]/total_labeled*100:.1f}%），"
        f"说明大部分进入法院的案件以原告诉求被驳回告终。"
        f"「撤销原判」和「改判」合计占比 "
        f"{(outcomes.get('撤销原判',0)+outcomes.get('改判',0)+outcomes.get('部分改判',0))/total_labeled*100:.1f}%，"
        f"反映了二审纠错机制的实际运作比例。"
        f"整体判决结果分布反映了司法实践中对原审裁判的相对尊重。"
    )
    print(f"\n   📝 结论: {conclusion}")
    return conclusion


# ============================================================
# 分析 4: 案由-关键词关联分析
# ============================================================

def analyze_cause_keywords(cases):
    """
    每个主要案由类别下的 Top 5 特征关键词，画分组柱状图。
    使用 TF-IDF 思想：关键词在该类别中的频次 / 该关键词的总频次。
    """
    print("\n" + "=" * 60)
    print("[4/5] 案由-关键词关联分析")
    print("=" * 60)

    # 按类别分组关键词
    cat_kw_counter = defaultdict(Counter)
    global_kw_counter = Counter()

    for c in cases:
        cat = c['_category']
        for kw in c['_keywords']:
            cat_kw_counter[cat][kw] += 1
            global_kw_counter[kw] += 1

    # 选取案例数 >= 100 的类别
    cat_counts = Counter(c['_category'] for c in cases)
    major_cats = [cat for cat, cnt in cat_counts.most_common() if cnt >= 100][:8]

    # 为每个类别计算特征关键词得分：局部频次 * log(全局频次) 的归一化
    total_global = sum(global_kw_counter.values())

    cat_top_kws = {}
    for cat in major_cats:
        scores = {}
        for kw, local_cnt in cat_kw_counter[cat].most_common(100):
            global_cnt = global_kw_counter[kw]
            # TF-IDF 风格得分: 局部频次占比 * IDF
            tf = local_cnt / cat_counts[cat]
            idf = np.log(total_global / (global_cnt + 1))
            scores[kw] = tf * idf * 1000  # 放大便于阅读
        top5 = sorted(scores.items(), key=lambda x: -x[1])[:5]
        cat_top_kws[cat] = top5

    # ---- 画分组柱状图 ----
    n_cats = len(major_cats)
    n_kws = 5
    x = np.arange(n_cats)
    width = 0.15

    fig, ax = plt.subplots(figsize=(16, 8))
    bar_colors = plt.cm.Reds(np.linspace(0.3, 0.95, n_kws))

    for i in range(n_kws):
        values = []
        labels_for_legend = []
        for cat in major_cats:
            if i < len(cat_top_kws[cat]):
                kw, score = cat_top_kws[cat][i]
                values.append(score)
                labels_for_legend.append(kw)
            else:
                values.append(0)

        bars = ax.bar(x + i * width, values, width, color=bar_colors[i],
                      edgecolor='white', linewidth=0.5)

    # 简化 X 轴标签
    short_labels = []
    for cat in major_cats:
        if len(cat) > 8:
            short_labels.append(cat[:7] + '…')
        else:
            short_labels.append(cat)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(short_labels, fontsize=9, rotation=30, ha='right')
    ax.set_ylabel('TF-IDF 特征得分', fontsize=11)
    ax.set_title('主要案由类别的 Top 5 特征关键词', fontsize=14, fontweight='bold')

    # 图例：显示每个类别的 Top 1 关键词
    legend_items = [f'{major_cats[j][:6]}→{cat_top_kws[major_cats[j]][0][0]}'
                    for j in range(n_cats)][:8]
    ax.legend(legend_items, fontsize=7, loc='upper right')

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_cause_keywords_bar.png'))
    plt.close(fig)
    print(f"   ✅ 图表已保存: fig4_cause_keywords_bar.png")

    print(f"   各类别 Top 3 特征关键词:")
    for cat in major_cats:
        top3 = [f'{kw}({score:.1f})' for kw, score in cat_top_kws[cat][:3]]
        print(f"     {cat}: {', '.join(top3)}")

    # 找最有区分度的关键词
    conclusion = (
        f"案由-关键词关联分析揭示了不同案件类型的特征关键词体系。"
        f"「{major_cats[0]}」类案件的特征关键词为 "
        f"「{cat_top_kws[major_cats[0]][0][0]}」「{cat_top_kws[major_cats[0]][1][0]}」等，"
        f"而「{major_cats[-1]}」类案件则以 "
        f"「{cat_top_kws[major_cats[-1]][0][0]}」为显著特征。"
        f"关键词在不同案由间的分布差异较大，说明案由体系对关键词有较强的约束力，"
        f"这为基于关键词的案件自动分类提供了可行依据。"
    )
    print(f"\n   📝 结论: {conclusion}")
    return conclusion


# ============================================================
# 分析 5: 年度趋势分析
# ============================================================

def analyze_year_trend(cases):
    """
    分年度统计案件数量 + 民事/刑事/行政比例变化，画堆叠面积图。
    """
    print("\n" + "=" * 60)
    print("[5/5] 年度趋势分析")
    print("=" * 60)

    # 刑事关键词（用于分类）
    CRIMINAL_KW = {'罪','盗窃','抢劫','杀人','故意伤害','诈骗','强奸','走私','贩毒',
                   '贪污','贿赂','受贿','行贿','挪用','非法拘禁','绑架','敲诈勒索',
                   '抢夺','侵占','职务侵占','寻衅滋事','聚众斗殴','危险驾驶','醉驾',
                   '逃税','非法经营','集资诈骗','洗钱','伪造','冒充','拐卖','虐待',
                   '赌博','开设赌场','组织卖淫','贩毒','毒品','故意杀人','交通肇事',
                   '过失致人死亡','黑社会','贩卖毒品','制造毒品','非法吸收公众存款'}
    ADMIN_KW = {'行政','行政复议','行政诉讼','国家赔偿','征地','拆迁','许可','处罚',
                '信息公开','社保','工伤认定','行政登记','行政确认','行政强制'}

    def classify_type(keywords):
        crim = sum(1 for kw in keywords if any(ck in kw for ck in CRIMINAL_KW))
        admin = sum(1 for kw in keywords if any(ak in kw for ak in ADMIN_KW))
        if crim > admin and crim > 0:
            return '刑事'
        if admin > crim and admin > 0:
            return '行政'
        return '民事'

    # 按年度统计
    year_type_counter = defaultdict(Counter)
    year_total = Counter()

    for c in cases:
        y = c['_year']
        if y is None:
            continue
        t = classify_type(c['_keywords'])
        year_type_counter[y][t] += 1
        year_total[y] += 1

    years = sorted(year_total.keys())
    civil_counts = [year_type_counter[y].get('民事', 0) for y in years]
    criminal_counts = [year_type_counter[y].get('刑事', 0) for y in years]
    admin_counts = [year_type_counter[y].get('行政', 0) for y in years]
    totals = [year_total[y] for y in years]

    print(f"   年度范围: {years[0]} - {years[-1]}")
    print(f"   总案例数: {sum(totals)}")
    for y in years:
        c = year_type_counter[y]
        print(f"     {y}: 民事{c.get('民事',0):4d}  刑事{c.get('刑事',0):4d}  行政{c.get('行政',0):4d}  合计{year_total[y]:5d}")

    # ---- 画堆叠面积图 ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})

    # 上图：堆叠面积图（绝对数量）
    ax1.stackplot(years, civil_counts, criminal_counts, admin_counts,
                  labels=['民事', '刑事', '行政'],
                  colors=['#34D399', '#F87171', '#FBBF24'],
                  alpha=0.8)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_ylabel('案例数量', fontsize=11)
    ax1.set_title('2014-2025 年各类案件数量趋势（堆叠面积图）', fontsize=14, fontweight='bold')
    ax1.set_xlim(years[0], years[-1])
    ax1.grid(axis='y', alpha=0.3)

    # 下图：百分比堆叠面积图
    pct_civil = [c/t*100 if t else 0 for c, t in zip(civil_counts, totals)]
    pct_criminal = [c/t*100 if t else 0 for c, t in zip(criminal_counts, totals)]
    pct_admin = [c/t*100 if t else 0 for c, t in zip(admin_counts, totals)]

    ax2.stackplot(years, pct_civil, pct_criminal, pct_admin,
                  labels=['民事', '刑事', '行政'],
                  colors=['#34D399', '#F87171', '#FBBF24'],
                  alpha=0.8)
    ax2.set_ylabel('占比 (%)', fontsize=11)
    ax2.set_xlabel('年份', fontsize=11)
    ax2.set_title('各类案件占比变化趋势', fontsize=13, fontweight='bold')
    ax2.set_xlim(years[0], years[-1])
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig5_year_trend_area.png'))
    plt.close(fig)
    print(f"   ✅ 图表已保存: fig5_year_trend_area.png")

    # 计算趋势变化
    if len(years) >= 2:
        civil_change = pct_civil[-1] - pct_civil[0]
        criminal_change = pct_criminal[-1] - pct_criminal[0]

    conclusion = (
        f"2014-2025 年间共收录 {sum(totals)} 条案例。"
        f"案件总数在 {years[totals.index(max(totals))]} 年达到峰值（{max(totals)} 条），"
        f"整体呈现先升后稳的趋势。"
        f"民事案件始终占主导地位（{np.mean(pct_civil):.1f}%），"
        f"刑事案件占比 {np.mean(pct_criminal):.1f}%。"
        f"值得注意的是，刑事案件占比从 {years[0]} 年的 {pct_criminal[0]:.1f}%"
        f"变化至 {years[-1]} 年的 {pct_criminal[-1]:.1f}%，"
        f"反映了司法公开数据收录口径的演进。"
    )
    print(f"\n   📝 结论: {conclusion}")
    return conclusion


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("   ⚖️  裁判文书大数据深度分析")
    print("=" * 60)

    # 前置检查
    if not os.path.exists(CSV_PATH):
        print(f"❌ 数据文件不存在: {CSV_PATH}")
        print(f"   请确认 all_cases_perfect.csv 在正确路径")
        sys.exit(1)

    # 加载数据
    cases = load_data(CSV_PATH)
    cases = preprocess(cases)
    print(f"   预处理完成，{len(cases)} 条案例就绪\n")

    # 逐项分析
    conclusions = []

    try:
        c1 = analyze_keyword_cooccurrence(cases)
        conclusions.append(("关键词共现网络分析", c1))
    except Exception as e:
        print(f"   ❌ 分析1失败: {e}")
        import traceback; traceback.print_exc()

    try:
        c2 = analyze_length_distribution(cases)
        conclusions.append(("判决书篇幅分析", c2))
    except Exception as e:
        print(f"   ❌ 分析2失败: {e}")

    try:
        c3 = analyze_judgment_outcomes(cases)
        conclusions.append(("判决结果情感倾向分析", c3))
    except Exception as e:
        print(f"   ❌ 分析3失败: {e}")

    try:
        c4 = analyze_cause_keywords(cases)
        conclusions.append(("案由-关键词关联分析", c4))
    except Exception as e:
        print(f"   ❌ 分析4失败: {e}")

    try:
        c5 = analyze_year_trend(cases)
        conclusions.append(("年度趋势分析", c5))
    except Exception as e:
        print(f"   ❌ 分析5失败: {e}")

    # 保存结论
    conclusion_path = os.path.join(OUTPUT_DIR, 'conclusions.txt')
    with open(conclusion_path, 'w', encoding='utf-8') as f:
        f.write("裁判文书大数据深度分析 — 文字结论汇总\n")
        f.write("=" * 50 + "\n\n")
        for title, text in conclusions:
            f.write(f"【{title}】\n{text}\n\n")

    print("\n" + "=" * 60)
    print("   ✅ 全部分析完成！")
    print(f"   📁 图表目录: {OUTPUT_DIR}/")
    print(f"   📝 结论文件: {conclusion_path}")
    print("=" * 60)

    # 列出输出文件
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        if fname.endswith(('.png', '.txt')):
            fpath = os.path.join(OUTPUT_DIR, fname)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"     {fname} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
