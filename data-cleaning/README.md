# 法眼AI 2.0 · 数据清洗流水线

对 10,321 条裁判文书结构化数据进行全面清洗，修复**关键词**、**原告诉求**、**判决结果**三大类的质量问题。

## 最终成果

| 指标 | 原始状态 | 最终状态 |
|------|---------|---------|
| 关键词符号连接 | 4,947+ | **0** |
| 关键词空位 | ~10,000 | **32** |
| 关键词重复行 | 625 | **48** |
| 关键词完美行 | ~3,000 | **10,241** |
| 原告诉求填充率 | 34.5% | **100%** |
| 原告诉求平均长度 | ~30字 | **81字** |
| 判决结果填充率 | ~92% | **~100%** |

## 流水线流程

```
all_cases.csv (10,321行 × 15列)
    │
    ├── [阶段1] fix_kw_swap.py        ← Swap机制去重
    ├── [阶段2] fix_empty_dup.py       ← 空值+重复修复(LLM)
    ├── [阶段3] fix_final_34.py        ← 顽固符号行逐行修复(LLM)
    ├── [阶段4] fix_claims_all_rows.py ← 全量原告诉求重写(LLM)
    ├── [阶段5] fix_judgment_final.py  ← 判决结果精细修复(LLM)
    └── [阶段6] extract_perfect.py     ← 完美行导出
                                           │
                                     all_cases_perfect.csv
                                     (10,241行 × 14列)
```

## 环境要求

```bash
pip3 install openai python-dotenv
```

## .env 配置

```bash
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

## 运行

按阶段顺序执行，每阶段输出中间CSV供下一阶段使用：
```bash
python3 fix_kw_swap.py        # → all_cases_swap.csv
python3 fix_empty_dup.py       # → all_cases_filled.csv
python3 fix_final_34.py        # → all_cases_kw_clean.csv
python3 fix_claims_all_rows.py # → all_cases_claims.csv
python3 fix_judgment_final.py  # → all_cases_judgment.csv
python3 extract_perfect.py     # → all_cases_perfect.csv
```
