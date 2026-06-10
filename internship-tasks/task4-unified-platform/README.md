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
# 安装依赖
pip3 install fastapi uvicorn langgraph langchain langchain-openai \
  scikit-learn numpy jieba rank_bm25 python-dotenv

# 生成仪表盘数据
python3 dashboard_data.py

# 启动服务
python3 -m uvicorn backend:app --host 0.0.0.0 --port 8800

# 浏览器访问
open http://localhost:8800
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
