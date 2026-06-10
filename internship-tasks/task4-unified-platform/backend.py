#!/usr/bin/env python3
"""
法眼AI 2.0 · 统一后端 API
=========================
FastAPI 后端，提供三大模块：
  - /api/dashboard/*    数据仪表盘
  - /api/agent/*        多Agent智能问答
  - /api/report/*       报告生成

启动: python backend.py
访问: http://localhost:8800
"""

import os
import sys
import json
import time
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ABS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ABS_DIR)

# ============================================================
# 生命周期：启动时初始化
# ============================================================
agent_instance = None
dashboard_cache = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_instance, dashboard_cache
    print("🚀 法眼AI 2.0 启动中...")

    # 加载仪表盘数据（如果存在缓存就用缓存，否则计算）
    cache_path = os.path.join(ABS_DIR, "static", "data", "dashboard.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            dashboard_cache = json.load(f)
        print("   📊 仪表盘数据已缓存")
    else:
        from dashboard_data import compute_dashboard_data
        dashboard_cache = compute_dashboard_data()
        print("   📊 仪表盘数据已计算")

    # 初始化多Agent
    from multi_agent import FaYanMultiAgent
    agent_instance = FaYanMultiAgent()
    print("   🤖 多Agent系统已就绪")
    print(f"✅ 服务启动: http://localhost:8800")
    yield
    print("🛑 服务关闭")

# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(
    title="法眼AI 2.0 统一平台",
    description="智能法律案例分析系统 - 多Agent协作 + 数据仪表盘 + 报告生成",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件
static_dir = os.path.join(ABS_DIR, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ============================================================
# 请求模型
# ============================================================
class ChatRequest(BaseModel):
    query: str
    history: Optional[List[dict]] = []

class ChatResponse(BaseModel):
    answer: str
    intent: str
    cases: list
    verification: dict
    agent_trace: str

class ReportRequest(BaseModel):
    title: Optional[str] = "法眼AI 案例分析报告"
    queries: Optional[List[str]] = []

# ============================================================
# 页面路由
# ============================================================
@app.get("/")
async def index():
    """主页 - 统一平台"""
    return FileResponse(os.path.join(ABS_DIR, "templates", "platform.html"))

# ============================================================
# 仪表盘 API
# ============================================================
@app.get("/api/dashboard/overview")
async def dashboard_overview():
    """仪表盘概览数据"""
    return JSONResponse(dashboard_cache.get("overview", {}))

@app.get("/api/dashboard/case-types")
async def dashboard_case_types():
    """案由分布"""
    return JSONResponse(dashboard_cache.get("case_type_dist", []))

@app.get("/api/dashboard/keywords")
async def dashboard_keywords():
    """关键词频率"""
    return JSONResponse(dashboard_cache.get("top_keywords", []))

@app.get("/api/dashboard/judgments")
async def dashboard_judgments():
    """判决结果分布"""
    return JSONResponse(dashboard_cache.get("judgment_dist", []))

@app.get("/api/dashboard/categories")
async def dashboard_categories():
    """民事/刑事/行政分布"""
    return JSONResponse(dashboard_cache.get("category_dist", []))

@app.get("/api/dashboard/lengths")
async def dashboard_lengths():
    """案例长度分布"""
    return JSONResponse(dashboard_cache.get("length_dist", []))

@app.get("/api/dashboard/keyword-network")
async def dashboard_keyword_network():
    """关键词共现网络"""
    return JSONResponse(dashboard_cache.get("keyword_network", {}))

@app.get("/api/dashboard/all")
async def dashboard_all():
    """全部仪表盘数据"""
    return JSONResponse(dashboard_cache)

# ============================================================
# 多Agent 对话 API
# ============================================================
@app.post("/api/agent/chat", response_model=ChatResponse)
async def agent_chat(req: ChatRequest):
    """多Agent对话"""
    if agent_instance is None:
        raise HTTPException(503, "Agent系统未就绪")

    try:
        result = agent_instance.chat(req.query, req.history)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(500, f"Agent错误: {str(e)}")

@app.get("/api/agent/health")
async def agent_health():
    """Agent健康检查"""
    return {"status": "ok" if agent_instance else "not_ready"}


@app.post("/api/agent/chat/stream")
async def agent_chat_stream(req: ChatRequest):
    """多Agent流式对话 (SSE)"""
    if agent_instance is None:
        raise HTTPException(503, "Agent系统未就绪")

    import json as _json
    async def generate():
        try:
            for event in agent_instance.chat_stream(req.query, req.history):
                yield f"data: {_json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {_json.dumps({'event': 'error', 'data': {'message': str(e)}}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# ============================================================
# 报告生成 API
# ============================================================
@app.post("/api/report/generate")
async def generate_report(req: ReportRequest):
    """生成分析报告（HTML格式）"""
    report_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{req.title}</title>
<style>
body{{font-family:"PingFang SC","Microsoft YaHei",sans-serif;max-width:900px;margin:40px auto;padding:20px;color:#333;line-height:1.8}}
h1{{text-align:center;color:#7F1D1D;border-bottom:2px solid #7F1D1D;padding-bottom:10px}}
h2{{color:#5B0E0E;margin-top:30px}}
.section{{margin:20px 0}}
table{{width:100%;border-collapse:collapse;margin:15px 0}}
th,td{{border:1px solid #ddd;padding:10px 12px;text-align:left}}
th{{background:#7F1D1D;color:#fff}}
.stats{{display:flex;gap:20px;flex-wrap:wrap;margin:20px 0}}
.stat-box{{flex:1;min-width:150px;background:#FEF2F2;border-radius:10px;padding:20px;text-align:center}}
.stat-box .num{{font-size:2rem;color:#7F1D1D;font-weight:700}}
.stat-box .label{{color:#666;margin-top:5px}}
.footer{{margin-top:40px;text-align:center;color:#999;font-size:.85rem;border-top:1px solid #eee;padding-top:20px}}
</style>
</head>
<body>
<h1>⚖️ {req.title}</h1>
<p style="text-align:center;color:#666">生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")} | 数据来源: 法眼AI 裁判文书库</p>

<div class="section">
<h2>一、数据概览</h2>
<div class="stats">
<div class="stat-box"><div class="num">{dashboard_cache['overview']['total_cases']:,}</div><div class="label">案例总数</div></div>
<div class="stat-box"><div class="num">{dashboard_cache['overview']['civil_count']:,}</div><div class="label">民事案件</div></div>
<div class="stat-box"><div class="num">{dashboard_cache['overview']['criminal_count']:,}</div><div class="label">刑事案件</div></div>
<div class="stat-box"><div class="num">{dashboard_cache['overview']['case_types_count']}</div><div class="label">案由类型</div></div>
</div>
</div>

<div class="section">
<h2>二、案由分布 Top 10</h2>
<table>
<tr><th>排名</th><th>案由</th><th>数量</th><th>占比</th></tr>
"""
    type_dist = dashboard_cache.get("case_type_dist", [])[:10]
    for i, item in enumerate(type_dist, 1):
        pct = item["value"] / dashboard_cache["overview"]["total_cases"] * 100
        report_html += f"<tr><td>{i}</td><td>{item['name']}</td><td>{item['value']:,}</td><td>{pct:.1f}%</td></tr>\n"

    report_html += """
</table>
</div>

<div class="section">
<h2>三、判决结果分布</h2>
<table>
<tr><th>判决类型</th><th>数量</th><th>占比</th></tr>
"""
    jud_dist = dashboard_cache.get("judgment_dist", [])
    for item in jud_dist:
        pct = item["value"] / dashboard_cache["overview"]["total_cases"] * 100
        report_html += f"<tr><td>{item['name']}</td><td>{item['value']:,}</td><td>{pct:.1f}%</td></tr>\n"

    report_html += f"""
</table>
</div>

<div class="section">
<h2>四、高频关键词 Top 15</h2>
<table>
<tr><th>排名</th><th>关键词</th><th>频次</th></tr>
"""
    kw_list = dashboard_cache.get("top_keywords", [])[:15]
    for i, item in enumerate(kw_list, 1):
        report_html += f"<tr><td>{i}</td><td>{item['name']}</td><td>{item['value']:,}</td></tr>\n"

    report_html += f"""
</table>
</div>

<div class="footer">
<p>法眼AI 2.0 · 智能法律案例分析系统 | 报告自动生成</p>
<p>本报告基于 {dashboard_cache['overview']['total_cases']:,} 条裁判文书数据自动生成</p>
</div>
</body>
</html>"""

    return Response(
        content=report_html,
        media_type="text/html",
        headers={"Content-Disposition": "attachment; filename=report.html"}
    )


# ============================================================
# 案例搜索 API（供前端快速预览）
# ============================================================
@app.get("/api/search")
async def search_cases(q: str = "", top_k: int = 5):
    """快速案例搜索"""
    if not q:
        return JSONResponse([])
    from multi_agent import get_retriever
    retriever = get_retriever()
    cases = retriever.search(q, top_k=top_k)
    # 精简返回字段
    simplified = []
    for c in cases:
        simplified.append({
            "filename": c["filename"],
            "desc": c["desc"][:200],
            "judgment_result": c["judgment_result"][:150],
            "score": c["score"]
        })
    return JSONResponse(simplified)


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8800, log_level="info")
