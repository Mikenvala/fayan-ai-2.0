# 法眼AI 2.0 · 任务模块

| 目录 | 内容 | 技术栈 |
|------|------|--------|
| `task1-analysis/` | 数据清洗与统计分析 | Pandas, Matplotlib, scikit-learn |
| `task2-agent/` | 法律案件检索 Agent | LangChain, BM25, TF-IDF |
| `task3-report/` | 因果推断分析报告 | Python, LaTeX |
| `platform/` | 统一 Web 平台 | FastAPI, LangGraph, ECharts |
| `task5-benchmark/` | 多模型法律能力评测 | 六款国产大模型 API |

## 数据

```bash
DATA=../all_cases_perfect.csv   # 10,241 条清洗后裁判文书
```

## 环境变量

```bash
source ../.env   # MINIMAX_API_KEY, DEEPSEEK_API_KEY 等
```
