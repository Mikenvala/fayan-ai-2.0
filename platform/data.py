#!/usr/bin/env python3
"""
法眼AI 2.0 · 仪表盘数据计算
============================
从 all_cases_perfect.csv 计算：
  - 概览统计（总数、类型分布）
  - 案由分布 Top 15
  - 关键词热力图
  - 案件关键词共现网络
  - 判决结果分类统计
"""

import csv
import json
import os
import re
from collections import Counter, defaultdict

ABS_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(ABS_DIR, "..", "..", "all_cases_perfect.csv")
OUTPUT_DIR = os.path.join(ABS_DIR, "static", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 案由关键词映射
CASE_TYPE_KEYWORDS = {
    "民间借贷": ["借贷", "借款", "借条", "欠款", "民间借贷", "利息", "本金"],
    "合同纠纷": ["合同", "违约", "协议", "履行", "解除合同", "定金", "货款"],
    "交通事故": ["交通", "肇事", "车祸", "事故", "撞", "车主", "保险理赔"],
    "劳动争议": ["劳动", "工资", "工伤", "解雇", "加班", "社保", "劳动合同", "劳动关系"],
    "婚姻家庭": ["离婚", "抚养", "继承", "婚姻", "夫妻", "财产分割", "赡养"],
    "公司纠纷": ["公司", "股东", "股权", "法人", "工商", "企业", "注册"],
    "知识产权": ["专利", "商标", "著作权", "版权", "侵权", "知识产权"],
    "物业纠纷": ["物业", "业主", "物业费", "小区", "房产"],
    "刑事犯罪": ["罪", "刑事", "盗窃", "诈骗", "故意伤害", "贪污", "受贿", "寻衅滋事"],
    "行政纠纷": ["行政", "政府", "处罚", "复议", "拆迁", "征收"],
    "房产纠纷": ["房产", "房屋", "买房", "卖房", "租赁", "租金", "房贷"],
    "金融纠纷": ["金融", "银行", "保险", "证券", "理财", "信用卡", "贷款"],
    "医疗纠纷": ["医疗", "医院", "医生", "手术", "医疗事故", "误诊"],
    "网络纠纷": ["网络", "电商", "网购", "直播", "平台", "互联网"],
}

def classify_case_type(desc: str) -> str:
    """基于关键词分类案由"""
    scores = {}
    for ctype, keywords in CASE_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in desc)
        if score > 0:
            scores[ctype] = score
    if scores:
        return max(scores, key=scores.get)
    return "其他"

def compute_dashboard_data():
    """计算所有仪表盘数据"""
    cases = []
    with open(CSV_PATH, "rb") as f:
        raw = f.read().replace(bytes([0]), b'')
    import io
    with io.TextIOWrapper(io.BytesIO(raw), encoding='utf-8') as tf:
        reader = csv.DictReader(tf)
        for row in reader:
            cases.append(row)

    total = len(cases)
    print(f"📂 共 {total} 条案例")

    # ============================================================
    # 1. 案由分布
    # ============================================================
    type_counter = Counter()
    for c in cases:
        desc = c.get("案件描述", "")
        ctype = classify_case_type(desc)
        type_counter[ctype] += 1

    case_type_dist = [
        {"name": k, "value": v}
        for k, v in type_counter.most_common(15)
    ]

    # ============================================================
    # 2. 关键词频率 Top 30
    # ============================================================
    keyword_counter = Counter()
    for c in cases:
        for i in range(1, 11):
            kw = c.get(f"关键词_{i:02d}", "").strip()
            if kw:
                keyword_counter[kw] += 1

    top_keywords = [
        {"name": k, "value": v}
        for k, v in keyword_counter.most_common(50)
    ]

    # ============================================================
    # 3. 判决结果关键词分析
    # ============================================================
    judgment_patterns = {
        "支持原告诉求": ["支持", "予以支持", "应予支持", "应当支持", "判令"],
        "部分支持": ["部分支持", "酌情", "适当", "部分驳回"],
        "驳回": ["驳回", "不予支持", "不予采纳", "不予认定"],
        "调解/和解": ["调解", "和解", "协商", "达成"],
        "撤销": ["撤销", "发回重审", "改判"],
        "维持原判": ["维持原判", "维持原裁判", "驳回上诉"],
    }

    judgment_counter = Counter()
    for c in cases:
        result = c.get("判决结果", "") + c.get("判别标准", "")
        matched = False
        for jtype, patterns in judgment_patterns.items():
            if any(p in result for p in patterns):
                judgment_counter[jtype] += 1
                matched = True
                break
        if not matched:
            judgment_counter["其他"] += 1

    judgment_dist = [
        {"name": k, "value": v}
        for k, v in judgment_counter.most_common()
    ]

    # ============================================================
    # 4. 民事 vs 刑事 vs 行政 大分类
    # ============================================================
    civil_keywords = ["民事", "民初", "合同", "借贷", "劳动", "离婚", "继承", "侵权", "交通事故"]
    criminal_keywords = ["刑事", "刑初", "公诉", "罪", "盗窃", "诈骗", "贪污"]
    admin_keywords = ["行政", "行初", "政府", "复议", "征收"]

    category_counter = Counter()
    for c in cases:
        fname = c.get("文件名", "")
        desc = c.get("案件描述", "")
        text = fname + desc
        if any(kw in text for kw in criminal_keywords):
            category_counter["刑事"] += 1
        elif any(kw in text for kw in admin_keywords):
            category_counter["行政"] += 1
        elif any(kw in text for kw in civil_keywords):
            category_counter["民事"] += 1
        else:
            category_counter["其他"] += 1

    category_dist = [
        {"name": k, "value": v}
        for k, v in category_counter.most_common()
    ]

    # ============================================================
    # 5. 关键词共现矩阵 (Top 20 关键词)
    # ============================================================
    top20_kw = [k for k, v in keyword_counter.most_common(20)]
    cooccurrence = defaultdict(lambda: defaultdict(int))
    for c in cases:
        case_kws = set()
        for i in range(1, 11):
            kw = c.get(f"关键词_{i:02d}", "").strip()
            if kw in top20_kw:
                case_kws.add(kw)
        case_kws = list(case_kws)
        for i in range(len(case_kws)):
            for j in range(i + 1, len(case_kws)):
                cooccurrence[case_kws[i]][case_kws[j]] += 1
                cooccurrence[case_kws[j]][case_kws[i]] += 1

    # 转成 ECharts 力导向图格式
    nodes = [{"name": kw, "symbolSize": min(60, max(20, keyword_counter[kw] / 50))}
             for kw in top20_kw]
    links = []
    for k1 in top20_kw:
        for k2 in top20_kw:
            if k1 < k2 and cooccurrence[k1][k2] > 0:
                links.append({
                    "source": k1,
                    "target": k2,
                    "value": cooccurrence[k1][k2]
                })

    # ============================================================
    # 6. 案例长度分布
    # ============================================================
    length_bins = {"短(<200字)": 0, "中(200-500字)": 0, "长(500-1000字)": 0, "超长(>1000字)": 0}
    for c in cases:
        desc_len = len(c.get("案件描述", ""))
        if desc_len < 200:
            length_bins["短(<200字)"] += 1
        elif desc_len < 500:
            length_bins["中(200-500字)"] += 1
        elif desc_len < 1000:
            length_bins["长(500-1000字)"] += 1
        else:
            length_bins["超长(>1000字)"] += 1

    length_dist = [{"name": k, "value": v} for k, v in length_bins.items()]

    # ============================================================
    # 汇总
    # ============================================================
    dashboard_data = {
        "overview": {
            "total_cases": total,
            "civil_count": category_counter.get("民事", 0),
            "criminal_count": category_counter.get("刑事", 0),
            "admin_count": category_counter.get("行政", 0),
            "case_types_count": len(type_counter),
            "unique_keywords": len(keyword_counter),
        },
        "case_type_dist": case_type_dist,
        "top_keywords": top_keywords,
        "judgment_dist": judgment_dist,
        "category_dist": category_dist,
        "length_dist": length_dist,
        "keyword_network": {
            "nodes": nodes,
            "links": links
        }
    }

    # 保存
    output_path = os.path.join(OUTPUT_DIR, "dashboard.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 仪表盘数据已保存: {output_path}")
    return dashboard_data


if __name__ == "__main__":
    compute_dashboard_data()
