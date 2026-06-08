#!/usr/bin/env python3
"""
法眼AI - 案例数据统计分析脚本
输出 JSON 格式统计结果，供前端可视化使用
"""

import csv
import json
import re
import os
import sys
from collections import Counter, defaultdict

CSV_PATH = os.path.join(os.path.dirname(__file__), "all_cases_perfect.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "static", "data", "stats.json")

# 民事关键词（用于分类统计）
CIVIL_KW = {"合同", "纠纷", "侵权", "赔偿", "借贷", "借款", "离婚", "继承",
            "婚姻", "房产", "房屋", "土地", "建设工程", "租赁", "劳动",
            "工伤", "仲裁", "公司", "股权", "股东", "合伙", "交通事故",
            "保险", "医疗", "担保", "债权", "著作权", "商标", "专利",
            "执行异议", "民间借贷", "物业服务", "买卖", "不当得利"}

CRIMINAL_KW = {"罪", "盗窃", "抢劫", "杀人", "故意伤害", "诈骗", "强奸",
               "走私", "贩毒", "贪污", "贿赂", "受贿", "行贿", "挪用",
               "非法拘禁", "绑架", "敲诈勒索", "抢夺", "侵占", "职务侵占",
               "寻衅滋事", "聚众斗殴", "危险驾驶", "醉驾", "逃税",
               "非法经营", "集资诈骗", "洗钱", "伪造", "冒充"}

ADMIN_KW = {"行政", "行政复议", "行政诉讼", "国家赔偿", "征地", "拆迁",
            "许可", "处罚", "信息公开", "社保"}


def detect_category(keywords):
    """根据关键词判断案件类别"""
    civil_score = sum(1 for kw in CIVIL_KW if any(kw in k for k in keywords))
    criminal_score = sum(1 for kw in CRIMINAL_KW if any(kw in k for k in keywords))
    admin_score = sum(1 for kw in ADMIN_KW if any(kw in k for k in keywords))

    max_score = max(civil_score, criminal_score, admin_score)
    if max_score == 0:
        return "其他"
    if criminal_score == max_score:
        return "刑事"
    if admin_score == max_score:
        return "行政"
    return "民事"


def extract_year(filename):
    """从文件名提取年份"""
    m = re.search(r'\[(\d{4})\]', filename)
    if m:
        return int(m.group(1))
    m = re.search(r'(\d{4})', filename)
    if m:
        y = int(m.group(1))
        if 1990 <= y <= 2030:
            return y
    return None


def main():
    if not os.path.exists(CSV_PATH):
        print(f"错误: 找不到数据文件 {CSV_PATH}")
        sys.exit(1)

    print(f"正在分析案例数据...")

    categories = Counter()        # 案件类别分布
    years = Counter()             # 年份分布
    all_keywords = Counter()      # 关键词频率
    cause_of_action = Counter()   # 案由统计（从文件名提取）
    total = 0

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            # 提取所有关键词
            keywords = []
            for i in range(1, 11):
                kw = row.get(f"关键词_{i:02d}", "").strip()
                if kw:
                    keywords.append(kw)
                    all_keywords[kw] += 1

            # 分类
            cat = detect_category(keywords)
            categories[cat] += 1

            # 年份
            filename = row.get("文件名", "")
            year = extract_year(filename)
            if year:
                years[year] += 1

            # 案由（从文件名路径提取目录名）
            parts = filename.replace("\\", "/").split("/")
            if len(parts) >= 2:
                folder = parts[-2]
                folder = re.sub(r'^\d{4}/', '', folder)
                folder = re.sub(r'^\d{4}年度案例', '', folder)
                folder = re.sub(r'_扫描版$', '', folder)
                folder = folder.strip('_')
                if folder:
                    cause_of_action[folder] += 1

    # 构建输出
    result = {
        "total_cases": total,
        "category_distribution": [
            {"name": k, "value": v} for k, v in categories.most_common()
        ],
        "year_distribution": [
            {"year": str(k), "count": v} for k, v in sorted(years.items())
        ],
        "top_keywords": [
            {"name": k, "value": v} for k, v in all_keywords.most_common(50)
        ],
        "top_cause_of_action": [
            {"name": k, "value": v} for k, v in cause_of_action.most_common(20)
        ],
    }

    # 输出
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"分析完成: {total} 条案例")
    print(f"  类别: {dict(categories.most_common())}")
    print(f"  年份跨度: {min(years.keys()) if years else 'N/A'} - {max(years.keys()) if years else 'N/A'}")
    print(f"  统计结果已保存至: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
