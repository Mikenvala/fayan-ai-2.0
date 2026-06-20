#!/usr/bin/env python3
"""
阶段4：全量原告诉求重写
========================
逐行调用LLM结合TXT原文，重写原告诉求字段。
去除判决语言，确保纯原告/公诉机关诉讼请求表述。

输入: all_cases_kw_clean.csv + 原始TXT文件
输出: all_cases_claims.csv
"""

import csv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

CSV_IN = os.path.join(os.path.dirname(__file__), "all_cases_kw_clean.csv")
CSV_OUT = os.path.join(os.path.dirname(__file__), "all_cases_claims.csv")
CKPT = os.path.join(os.path.dirname(__file__), "claims_checkpoint.json")

FIELDNAMES = []

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
)

SYSTEM = """你是法律文书专家。从案件原文提取原告/公诉机关的诉讼请求。

【规则】
1. 纯原告/公诉机关诉求，不含被告答辩、法院认定或判决
2. 使用"原告请求""公诉机关指控"等表述
3. 长度50-200字，精炼完整
4. 不要出现"法院认为""本院""判决如下"等判决语言

仅输出原告/公诉机关的诉讼请求原文或摘要。"""

def fix_claim(idx, row):
    """重写单行原告诉求"""
    # 读取TXT原文
    filepath = row.get("文件名", "")
    txt_dir = os.path.join(os.path.dirname(__file__), "..", "_cases_cleaned")
    txt_path = os.path.join(txt_dir, filepath) if filepath else ""
    
    case_text = row.get("案件描述", "")[:300]
    if txt_path and os.path.exists(txt_path):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                case_text = f.read()[:3000]
        except:
            pass

    prompt = f"案件原文(前3000字):\n{case_text}\n\n请提取原告/公诉机关的诉讼请求:"
    
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, timeout=30
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  行{idx}错误: {e}")
        return row.get("原告诉求", "")

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

    total = len(rows)
    todo = [i for i in range(total) if str(i) not in results]
    print(f"需处理: {len(todo)}/{total} 行")

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(fix_claim, i, rows[i]): i for i in todo}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                new_claim = fut.result()
                if new_claim:
                    results[str(i)] = new_claim
            except Exception as e:
                print(f"  行{i}: {e}")

    for str_i, claim in results.items():
        rows[int(str_i)]["原告诉求"] = claim

    with open(CKPT, "w") as f:
        json.dump(results, f, ensure_ascii=False)

    with open(CSV_OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)

    avg_len = sum(len(r["原告诉求"]) for r in rows) / total
    print(f"✅ 完成: 修复 {len(results)} 条, 均长 {avg_len:.0f}字, 输出 {CSV_OUT}")

if __name__ == "__main__":
    process()
