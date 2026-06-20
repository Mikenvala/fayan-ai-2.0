# ⚖️ 法眼AI 2.0 统一平台

基于 LangGraph 多Agent协作 + FastAPI + ECharts 的智能法律案例分析 Web 系统。

## 架构

```
┌─────────────────────────────────────────────────────┐
│           前端 (index.html · SPA)                 │
│   📊 数据仪表盘    💬 智能问答    📄 报告生成          │
└─────────────────────────────────────────────────────┘
                         │ REST API
                         ▼
┌─────────────────────────────────────────────────────┐
│           FastAPI Backend (app.py)               │
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
  scikit-learn numpy jieba rank_bm25 python-dotenv markdown

# 2. 生成仪表盘数据（首次运行需要）
python3 data.py

# 3. 启动服务
python3 app.py

# 4. 浏览器访问
open http://localhost:8800
```

> 后台持续运行: `screen -dmS fayan python3 app.py`

## 环境变量

```bash
MINIMAX_API_KEY=your_key_here
MINIMAX_BASE_URL=https://api.minimax.chat/v1
```

## 功能模块

### 📊 数据仪表盘
- 基于 10,241 条裁判文书数据
- 7 张 ECharts 交互图表
- 案由分布、大类分布、判决结果、关键词共现网络、词云图

### 💬 智能问答（多Agent协作）
- **检索Agent**: BM25 + TF-IDF 混合检索 Top 3 相似案例
- **分析Agent**: 基于检索结果进行法律分析
- **校验Agent**: 事实核查，防幻觉，自动修订循环
- LangGraph 状态机编排: 意图→检索→分析→校验→格式化

### ⚖️ 模拟辩论
- 三Agent辩论：原告 Agent vs 被告 Agent vs 法官 Agent
- SSE 流式推送，实时展示辩论过程
- 支持自定义辩题和 1~5 轮辩论深度

### 📄 报告生成
- 一键生成 HTML 分析报告
- 支持在线预览和 PDF 下载
- AI 输出智能美化

## 技术亮点

- 多Agent协作 LangGraph 工作流，自动修订循环
- BM25(权重0.6) + TF-IDF char-ngram(权重0.4) 混合检索
- ECharts 7 张交互式可视化图表
- FastAPI 异步高性能后端 + SSE 流式输出
- 校验Agent 自动检测 LLM 幻觉
- Docker Compose 一键部署
