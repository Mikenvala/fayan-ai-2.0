# ⚖️ 法眼AI 2.0 — 智能法律案例分析统一平台

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.6-ff6b6b.svg)](https://langchain-ai.github.io/langgraph/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

基于 **LangGraph 多Agent协作 + FastAPI + ECharts + MiniMax LLM** 的智能法律案例分析平台，覆盖大数据分析、智能体开发、智能应用、学术报告四大模块。

![Dashboard](internship-tasks/task4-unified-platform/screenshots/dashboard.png)

---

## 🏗 架构

```
┌────────────────────────────────────────┐
│         前端 SPA (platform.html)         │
│   📊 数据仪表盘  💬 智能问答  📄 报告    │
└────────────────────────────────────────┘
                    │ REST / SSE
                    ▼
┌────────────────────────────────────────┐
│        FastAPI (backend.py)             │
│   /api/dashboard  /api/agent  /api/report│
└────────────────────────────────────────┘
         │              │            │
         ▼              ▼            ▼
┌────────────┐ ┌────────────┐ ┌──────────┐
│ LangGraph   │ │ 数据分析    │ │ PDF报告   │
│ 多Agent     │ │ 10241条     │ │ Chrome    │
│ 协作系统     │ │ 裁判文书    │ │ 无头渲染   │
└────────────┘ └────────────┘ └──────────┘
```

## ✨ 四大模块

### 📊 Task 1 — 大数据分析
- 描述统计 → Logistic/LASSO → LDA → Mann-Kendall/STL
- **因果推断四层框架**: PSM + IPW + DID + 因果森林 + 中介分析 + Rosenbaum 敏感性检验
- 产出 20+ 张专业图表

### 🤖 Task 2 — 智能体开发
- LangChain Agent + BM25 + TF-IDF char-ngram 混合检索
- MiniMax LLM 驱动，防幻觉机制
- 意图识别 → 类案检索 (Top-3 准确率 70%+)

### 🧠 Task 4 — 多Agent协作平台
- **LangGraph 三Agent 协作**: 检索Agent → 分析Agent → 校验Agent
- **SSE 流式输出** + Markdown 实时渲染
- **ChatGPT 风格 UI**: 玻璃拟态、高斯模糊、打字光标
- 7 张 ECharts 交互图表 (含词云、关键词共现网络)

### 📝 Task 3 — CSSCI 论文
- LaTeX 格式学术论文 (15页, 25篇参考文献)
- 中英文结构化摘要、因果推断完整框架
- 6 张三线表 + 8 张图

---

## 🚀 快速启动

```bash
# 1. 安装依赖
pip install -r internship-tasks/task4-unified-platform/requirements.txt
pip install markdown

# 2. 设置 API Key
cp .env.example .env
# 编辑 .env 填入 MINIMAX_API_KEY

# 3. 预计算数据
cd internship-tasks/task4-unified-platform
python dashboard_data.py

# 4. 启动
python -m uvicorn backend:app --host 0.0.0.0 --port 8800
# 访问 http://localhost:8800
```

### Docker 一键部署

```bash
docker-compose up -d
# 访问 http://localhost:8800
```

---

## 📂 项目结构

```
├── internship-tasks/
│   ├── task1-analysis/      # 大数据分析 (因果推断)
│   ├── task2-agent/         # Agent 开发 (LangChain)
│   ├── task3-report/        # CSSCI 论文 (LaTeX)
│   └── task4-unified-platform/  # 统一平台
│       ├── backend.py       # FastAPI 后端
│       ├── multi_agent.py   # LangGraph 多Agent
│       ├── dashboard_data.py # 仪表盘数据
│       └── templates/
│           └── platform.html # 三合一前端
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🛠 技术栈

| 层级 | 技术 |
|------|------|
| **LLM** | MiniMax M2.7, LangChain, LangGraph |
| **检索** | BM25, TF-IDF char-ngram, jieba |
| **后端** | FastAPI, Uvicorn, SSE Streaming |
| **前端** | ECharts, marked.js, 原生 HTML/CSS/JS |
| **分析** | scikit-learn, NumPy, Pandas, 因果推断 |
| **工程** | Docker, Git, Chrome Headless PDF |

---

## 📄 License

MIT © [Mikenvala](https://github.com/Mikenvala)
