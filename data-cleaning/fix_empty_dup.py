#!/usr/bin/env python3
"""
阶段2：空值与重复关键词修复 (LLM批量处理)
===========================================
将每行10列关键词状态发送给DeepSeek，由LLM判断需填补的列。

输入: all_cases_swap.csv
输出: all_cases_filled.csv
"""

import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

CSV_IN = os.path.join(os.path.dirname(__file__), "all_cases_swap.csv")
CSV_OUT = os.path.join(os.path.dirname(__file__), "all_cases_filled.csv")
CKPT = os.path.join(os.path.dirname(__file__), "fill_checkpoint.json")

KW_COLS = [f"关键词_{n:02d}" for n in range(1, 11)]
FIELDNAMES = []
BATCH_SIZE = 10

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
)

SYSTEM = """你是法律关键词专家。为案件生成缺少的关键词。

【规则】
1. 每个关键词2-8个中文字符，纯名词短语，无任何标点
2. 绝对不能与已填好的关键词重复
3. 从案件描述中提取法律维度：证据、程序、实体、救济、法理等

输出JSON，只包含需要填/改的列：
[{"index":0,"keywords":{"关键词_03":"举证责任","关键词_05":"免责事由"}},...]"""

def has_sym(v):
    import re
    return bool(re.search(r'[、，,．.。；;：:｜|/·？！?\n\r]', v))

def process_batch(batch_rows):
    """处理一批行"""
    messages = [{"role": "system", "content": SYSTEM}]
    for row in batch_rows:
        # 构建关键词状态
        kw_state = []
        existing = set()
        for col in KW_COLS:
            v = row[col].strip()
            if len(v) >= 2 and not has_sym(v):
                kw_state.append(f'{col}="{v}"')
                existing.add(v)
            else:
                kw_state.append(f'{col}=(空)')
        context = f"案件描述: {row.get('案件描述','')[:300]}\n关键词状态: {'; '.join(kw_state)}"
        messages.append({"role": "user", "content": context})

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat", messages=messages, temperature=0.0, timeout=30
        )
        result = json.loads(resp.choices[0].message.content)
        return result
    except Exception as e:
        print(f"  LLM错误: {e}")
        return []

def process():
    global FIELDNAMES
    with open(CSV_IN, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        FIELDNAMES = list(reader.fieldnames)
        rows = list(reader)

    # 加载检查点
    results = {}
    if os.path.exists(CKPT):
        with open(CKPT, "r") as f:
            results = json.load(f)

    total = len(rows)
    filled = 0
    
    # 收集需要修复的行
    need_fix = []
    for idx, row in enumerate(rows):
        if str(idx) in results:
            continue
        has_issue = False
        vals = []
        for col in KW_COLS:
            v = row[col].strip()
            vals.append(v)
            if len(v) < 2 or has_sym(v):
                has_issue = True
        # 检查重复
        clean_vals = [v for v in vals if len(v) >= 2 and not has_sym(v)]
        if len(clean_vals) != len(set(clean_vals)):
            has_issue = True
        if has_issue:
            need_fix.append((idx, row))

    print(f"需修复: {len(need_fix)}/{total} 行")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        # 分批提交
        for batch_start in range(0, len(need_fix), BATCH_SIZE):
            batch = [r for _, r in need_fix[batch_start:batch_start+BATCH_SIZE]]
            fut = executor.submit(process_batch, batch)
            futures[fut] = batch_start

        for fut in as_completed(futures):
            batch_start = futures[fut]
            try:
                patches = fut.result()
                for patch in patches:
                    idx = need_fix[batch_start + patch["index"]][0]
                    results[str(idx)] = patch["keywords"]
                    filled += len(patch["keywords"])
            except Exception as e:
                print(f"  批次错误: {e}")

    # 应用修复
    for str_idx, kw_map in results.items():
        idx = int(str_idx)
        for col, val in kw_map.items():
            if col in KW_COLS:
                rows[idx][col] = val

    # 保存检查点
    with open(CKPT, "w") as f:
        json.dump(results, f, ensure_ascii=False)

    # 保存CSV
    with open(CSV_OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ 完成: 填补 {filled} 个关键词, 输出 {CSV_OUT}")

if __name__ == "__main__":
    process()
