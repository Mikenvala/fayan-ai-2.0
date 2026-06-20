#!/usr/bin/env python3
"""
阶段6：完美行导出
==================
筛选满足全部质量标准的行：10列关键词齐全 + 无符号 + 不重复。

输入: all_cases_judgment.csv
输出: all_cases_perfect.csv
"""

import csv
import os
import re

CSV_IN = os.path.join(os.path.dirname(__file__), "all_cases_judgment.csv")
CSV_OUT = os.path.join(os.path.dirname(__file__), "..", "all_cases_perfect.csv")

KW_COLS = [f"关键词_{n:02d}" for n in range(1, 11)]

def has_sym(v):
    return bool(re.search(r'[、，,．.。；;：:｜|/·？！?\n\r]', v))

def is_perfect(row):
    vals = [row[c].strip() for c in KW_COLS]
    # 全部有值
    if any(len(v) < 2 for v in vals):
        return False
    # 无符号
    if any(has_sym(v) for v in vals):
        return False
    # 不重复
    if len(set(vals)) != 10:
        return False
    return True

def process():
    with open(CSV_IN, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    total = len(rows)
    perfect = [r for r in rows if is_perfect(r)]
    imperfect = total - len(perfect)

    with open(CSV_OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(perfect)

    print(f"总行数: {total}")
    print(f"完美行: {len(perfect)} ({100*len(perfect)/total:.1f}%)")
    print(f"不完美: {imperfect}")
    print(f"输出: {CSV_OUT}")

if __name__ == "__main__":
    process()
