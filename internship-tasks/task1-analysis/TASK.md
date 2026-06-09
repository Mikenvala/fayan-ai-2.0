# Task 1 · 裁判文书大数据深度分析

## 任务目标

基于 10,241 条裁判文书数据，编写 Python 分析脚本，挖掘以下规律：

### 必须完成的分析维度（每个 20 分）

| # | 分析维度 | 具体要求 |
|---|---------|---------|
| 1 | 关键词共现网络 | 找出高频关键词之间的共现关系，输出共现矩阵，选取 Top 30 关键词画热力图 |
| 2 | 判决书篇幅分析 | 统计案件描述字数分布，按案由分类对比篇幅差异，画箱线图 |
| 3 | 判决结果情感倾向 | 识别"驳回"/"支持"/"维持"/"撤销"等判决关键词，统计各类占比 |
| 4 | 案由-关键词关联 | 每个案由类别下的 Top 5 特征关键词，画分组柱状图 |
| 5 | 年度趋势对比 | 分年度统计案件数量 + 民事/刑事比例变化，画堆叠面积图 |

### 评分标准

- **代码规范**（20%）：函数封装、注释清晰、PEP8 风格
- **分析深度**（30%）：不止于 count，有统计检验或发现规律
- **图表质量**（30%）：中文字体正确、配色美观、有标题和图例
- **结论输出**（20%）：脚本运行后在终端打印每项分析的文字结论

### 交付物

```
task1-analysis/
├── analysis_report.py    # 主分析脚本，一键运行输出所有图表+结论
├── output/
│   ├── fig1_cooccurrence_heatmap.png
│   ├── fig2_length_boxplot.png
│   ├── fig3_judgment_pie.png
│   ├── fig4_cause_keywords_bar.png
│   └── fig5_year_trend_area.png
└── output/conclusions.txt  # 文字结论汇总
```

### 提示

- 用 `font_manager` 设置中文字体，避免乱码
- 关键词共现：两个关键词出现在同一案例中就算一次共现
- 判决结果识别不需要太复杂，用关键词匹配即可（驳回/支持/维持/撤销/改判）
- 文字结论要简洁，每项 2-3 句话，说明发现了什么规律

### 参考代码框架

```python
#!/usr/bin/env python3
"""裁判文书数据分析脚本"""

import csv
import os
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# === 中文字体设置 ===
matplotlib.rcParams['font.sans-serif'] = ['STHeiti', 'PingFang SC', 'Heiti SC']
matplotlib.rcParams['axes.unicode_minus'] = False

CSV_PATH = "../all_cases_perfect.csv"      # 自己改成实际路径
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 数据加载 ===
def load_data(csv_path):
    """加载 CSV，返回 list[dict]"""
    cases = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append(row)
    return cases

# === 分析1: 关键词共现网络 ===
def analyze_keyword_cooccurrence(cases):
    """TODO: 实现"""
    pass

# === 分析2: 判决书篇幅分析 ===
def analyze_length_distribution(cases):
    """TODO: 实现"""
    pass

# === 分析3: 判决结果情感倾向 ===
def analyze_judgment_outcomes(cases):
    """TODO: 实现"""
    pass

# === 分析4: 案由-关键词关联 ===
def analyze_cause_keywords(cases):
    """TODO: 实现"""
    pass

# === 分析5: 年度趋势 ===
def analyze_year_trend(cases):
    """TODO: 实现"""
    pass

# === 主函数 ===
def main():
    print("=" * 60)
    print("裁判文书大数据深度分析")
    print("=" * 60)
    
    cases = load_data(CSV_PATH)
    print(f"\n加载案例: {len(cases)} 条")
    
    # 逐项分析
    conclusions = []
    
    print("\n[1/5] 关键词共现网络分析...")
    c1 = analyze_keyword_cooccurrence(cases)
    conclusions.append(c1)
    
    print("\n[2/5] 判决书篇幅分析...")
    c2 = analyze_length_distribution(cases)
    conclusions.append(c2)
    
    print("\n[3/5] 判决结果情感倾向分析...")
    c3 = analyze_judgment_outcomes(cases)
    conclusions.append(c3)
    
    print("\n[4/5] 案由-关键词关联分析...")
    c4 = analyze_cause_keywords(cases)
    conclusions.append(c4)
    
    print("\n[5/5] 年度趋势分析...")
    c5 = analyze_year_trend(cases)
    conclusions.append(c5)
    
    # 保存结论
    with open(f"{OUTPUT_DIR}/conclusions.txt", "w", encoding="utf-8") as f:
        for i, c in enumerate(conclusions, 1):
            f.write(f"分析{i}: {c}\n\n")
    
    print(f"\n✅ 所有图表已保存至 {OUTPUT_DIR}/")
    print(f"✅ 结论已保存至 {OUTPUT_DIR}/conclusions.txt")

if __name__ == "__main__":
    main()
```

开始填写代码吧，每完成一个分析函数就运行一次看看效果！
