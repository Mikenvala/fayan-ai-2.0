#!/usr/bin/env python3
"""
阶段5：判决结果精细修复
========================
对判决结果中的空值、过短、含杂质(如程序性裁定混入)等问题进行LLM修复。

输入: all_cases_claims.csv
输出: all_cases_judgment.csv
"""

import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

CSV_IN = os.path.join(os.path.dirname(__file__), "all_cases_claims.csv")
CSV_OUT = os.path.join(os.path.dirname(__file__), "all_cases_judgment.csv")
CKPT = os.path.join(os.path.dirname(__file__), "judgment_checkpoint.json")

FIELDNAMES = []

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
)

SYSTEM = """你是法律文书专家。从案件原文提取法院判决结果。

【规则】
1. 提取判决主文（法院最终裁定/判决内容）
2. 包含赔偿金额、刑期、驳回/支持等关键信息
3. 去除法官后语、附录、案件受理费负担等程序性内容
4. 长度30-300字，保留原文关键表述

仅输出判决结果。"""

def fix_judgment(idx, row):
    """修复单行判决结果"""
    filepath = row.get("文件名", "")
    txt_dir = os.path.join(os.path.dirname(__file__), "..", "_cases_cleaned")
    txt_path = os.path.join(txt_dir, filepath) if filepath else ""
    
    case_text = row.get("案件描述", "")[:500]
    if txt_path and os.path.exists(txt_path):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                full = f.read()
                # 判决在文末60-70%
                split_point = int(len(full) * 0.6)
                case_text = full[split_point:split_point+2000]
        except:
            pass

    current = row.get("判决结果", "")
    if len(current) > 30:
        return current  # Already good

    prompt = f"案件后段原文:\n{case_text}\n\n当前判决结果({len(current)}字): {current}\n\n请提取完整判决结果:"
    
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0, timeout=30
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  行{idx}错误: {e}")
        return current

def process():
    global FIELDNAMES
    with open(CSV_IN, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        FIELDNAMES = list(reader.fieldnames)
        rows = list(reader)

    results = {}
    if os.path.exists(CKPT):
        with open(CKPT) as f:
            results = json.load(f)

    # 找短判决
    todo = []
    for idx, row in enumerate(rows):
        if str(idx) in results:
            continue
        j = row.get("判决结果", "")
        if len(j) < 30:
            todo.append(idx)

    print(f"短判决: {len(todo)}/{len(rows)}")

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(fix_judgment, i, rows[i]): i for i in todo}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                new_j = fut.result()
                if new_j and len(new_j) > len(rows[i].get("判决结果", "")):
                    results[str(i)] = new_j
            except Exception as e:
                print(f"  行{i}: {e}")

    for str_i, j in results.items():
        rows[int(str_i)]["判决结果"] = j

    with open(CKPT, "w") as f:
        json.dump(results, f, ensure_ascii=False)

    with open(CSV_OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)
    print(f"✅ 完成: 修复 {len(results)} 条, 输出 {CSV_OUT}")

if __name__ == "__main__":
    process()
