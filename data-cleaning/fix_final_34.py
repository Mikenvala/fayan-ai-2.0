#!/usr/bin/env python3
"""
阶段3：顽固符号行逐行修复
==========================
对Swap多轮后仍残留的34个符号行，逐行调用LLM+TXT原文进行精准修复。

输入: all_cases_filled.csv
输出: all_cases_kw_clean.csv
"""

import csv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

CSV_IN = os.path.join(os.path.dirname(__file__), "all_cases_filled.csv")
CSV_OUT = os.path.join(os.path.dirname(__file__), "all_cases_kw_clean.csv")
CKPT = os.path.join(os.path.dirname(__file__), "final34_checkpoint.json")

KW_COLS = [f"关键词_{n:02d}" for n in range(1, 11)]
FIELDNAMES = []

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
)

def has_sym(v):
    return bool(re.search(r'[、，,．.。；;：:｜|/·？！?\n\r]', v))

def fix_single_row(idx, row):
    """逐行修复单个符号行"""
    kw_state = []
    existing = set()
    need_cols = []
    for col in KW_COLS:
        v = row[col].strip()
        if len(v) >= 2 and not has_sym(v):
            kw_state.append(f'{col}="{v}"')
            existing.add(v)
        else:
            kw_state.append(f'{col}=(含符号或空)')
            need_cols.append(col)

    if not need_cols:
        return {}

    prompt = f"""案件描述: {row.get('案件描述','')[:400]}
已知关键词: {'; '.join(kw_state)}
需修复列: {', '.join(need_cols)}

为每个需修复列生成一个纯中文关键词(2-8字)，不与已知值重复。
输出JSON: {{"关键词_XX": "值", ...}}"""

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, timeout=30
        )
        content = resp.choices[0].message.content
        # Extract JSON
        json_match = re.search(r'\{[^}]+\}', content)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"  行{idx}错误: {e}")
    return {}

def process():
    global FIELDNAMES
    with open(CSV_IN, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        FIELDNAMES = list(reader.fieldnames)
        rows = list(reader)

    # 找符号行
    sym_rows = []
    for idx, row in enumerate(rows):
        for col in KW_COLS:
            if has_sym(row[col].strip()):
                sym_rows.append(idx)
                break

    print(f"顽固符号行: {len(sym_rows)}")

    results = {}
    if os.path.exists(CKPT):
        with open(CKPT) as f:
            results = json.load(f)

    todo = [i for i in sym_rows if str(i) not in results]
    print(f"需处理: {len(todo)}")

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(fix_single_row, i, rows[i]): i for i in todo}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                patch = fut.result()
                if patch:
                    results[str(i)] = patch
            except Exception as e:
                print(f"  行{i}: {e}")

    # 应用
    for str_i, patch in results.items():
        for col, val in patch.items():
            if col in KW_COLS:
                rows[int(str_i)][col] = val

    with open(CKPT, "w") as f:
        json.dump(results, f, ensure_ascii=False)

    with open(CSV_OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)
    print(f"✅ 完成: 修复 {len(results)} 行, 输出 {CSV_OUT}")

if __name__ == "__main__":
    process()
