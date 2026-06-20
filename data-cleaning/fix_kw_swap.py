#!/usr/bin/env python3
"""
阶段1：Swap机制关键词去重
==========================
当LLM生成的关键词与另一列已有值冲突时，接受新值并清空被冲突列。

输入: all_cases.csv
输出: all_cases_swap.csv
"""

import csv
import re
import os
from dotenv import load_dotenv
load_dotenv()

CSV_IN = os.path.join(os.path.dirname(__file__), "..", "all_cases.csv")
CSV_OUT = os.path.join(os.path.dirname(__file__), "all_cases_swap.csv")
CKPT = os.path.join(os.path.dirname(__file__), "swap_checkpoint.json")

KW_COLS = [f"关键词_{n:02d}" for n in range(1, 11)]
FIELDNAMES = []

def has_sym(v):
    """检测是否含符号连接"""
    return bool(re.search(r'[、，,．.。；;：:｜|/·？！?\n\r]', v))

def clean_val(v):
    """去除非中文字符"""
    v = re.sub(r'[^一-鿿]', '', v.strip())
    return v if len(v) >= 2 else ''

def process():
    global FIELDNAMES
    with open(CSV_IN, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        FIELDNAMES = list(reader.fieldnames)
        rows = list(reader)

    total = len(rows)
    swaps = 0
    print(f"处理 {total} 行...")

    for idx, row in enumerate(rows):
        # 构建已有值映射
        existing_val_to_col = {}
        for col in KW_COLS:
            v = clean_val(row[col].strip())
            if v and not has_sym(v):
                existing_val_to_col[v] = col

        # 检查每个关键词列
        for col in KW_COLS:
            v = clean_val(row[col].strip())
            if not v:
                continue
            if v in existing_val_to_col:
                other_col = existing_val_to_col[v]
                if other_col != col:
                    # 接受新值，清空对方
                    row[other_col] = ''
                    del existing_val_to_col[v]
                    existing_val_to_col[v] = col
                    swaps += 1

        if (idx + 1) % 1000 == 0:
            print(f"  进度: {idx+1}/{total}, swaps={swaps}")

    # 保存
    with open(CSV_OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ 完成: {swaps} 次swap, 输出 {CSV_OUT}")

if __name__ == "__main__":
    process()
