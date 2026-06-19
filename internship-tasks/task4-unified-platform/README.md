# ⚖️ 法眼AI 2.0 统一平台

基于 LangGraph 多Agent协作 + FastAPI + ECharts 的智能法律案例分析系统。

## 架构

```
┌─────────────────────────────────────────────────────┐
│           前端 (platform.html · SPA)                 │
│   📊 数据仪表盘    💬 智能问答    📄 报告生成          │
└─────────────────────────────────────────────────────┘
                         │ REST API
                         ▼
┌─────────────────────────────────────────────────────┐
│           FastAPI Backend (backend.py)               │
│   /api/dashboard/*  /api/agent/*  /api/report/*      │
└─────────────────────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ LangGraph    │ │ dashboard   │ │ report_gen  │
│ 多Agent协作   │ │ _data.py    │ │ (HTML/PDF)  │
│              │ │ (10,241条)   │ │             │
│ ┌─────────┐  │ └─────────────┘ └─────────────┘
│ │检索Agent │  │
│ │分析Agent │  │
│ │校验Agent │  │
│ └─────────┘  │
└─────────────┘
```

## 快速启动

```bash
# 1. 安装依赖
pip3 install fastapi uvicorn langgraph langchain langchain-openai \
  scikit-learn numpy jieba rank_bm25 python-dotenv

# 2. 进入项目目录
cd internship-tasks/task4-unified-platform

# 3. 生成仪表盘数据（首次运行需要）
python3 dashboard_data.py

# 4. 启动服务
python3 backend.py

# 5. 浏览器访问
open http://localhost:8800
```

> 如需后台持续运行：`screen -dmS fayan python3 backend.py`

## 环境变量

在项目根目录创建 `.env` 文件：

```bash
MINIMAX_API_KEY=your_key_here
MINIMAX_BASE_URL=https://api.minimax.chat/v1
```

## 三大模块

### 📊 数据仪表盘
- 基于 10,241 条裁判文书数据
- 7 个 ECharts 交互图表
- 案由分布、大类分布、判决结果、关键词共现网络

### 💬 智能问答（多Agent协作）
- **检索Agent**: BM25 + TF-IDF 混合检索 Top 3 相似案例
- **分析Agent**: 基于检索结果进行法律分析
- **校验Agent**: 事实核查，防幻觉，确保引用准确
- LangGraph 状态机编排: 意图→检索→分析→校验→格式化

### ⚖️ 模拟辩论
- 三Agent辩论：原告Agent vs 被告Agent vs 法官Agent
- SSE 流式推送，实时展示辩论过程
- 支持自定义辩题和辩论轮数

### 📄 报告生成
- 一键生成 HTML 分析报告
- 包含：数据概览、案由分布、判决统计、关键词分析
- 支持在线预览和下载

## 技术亮点

- 多Agent协作 LangGraph 工作流，支持自动修订
- BM25(权重0.6) + TF-IDF char-ngram(权重0.4) 混合检索
- ECharts 7 张交互式可视化图表
- FastAPI 异步高性能后端
- 校验Agent 自动检测 LLM 幻觉
