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
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse, Response
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
    chat_history: Optional[List[dict]] = []
    format: Optional[str] = "html"  # html or pdf
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
# 报告生成 API (v4 - 纯字符串拼接，不转义Markdown)
# ============================================================
@app.post("/api/report/generate")
async def generate_report(req: ReportRequest):
    import subprocess, tempfile, os, re as _re
    total = dashboard_cache['overview']['total_cases']

    # 构建报告 HTML（用列表拼接，不用 f-string）
    parts = []
    
    # CSS & 头部
    parts.append('''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>''')
    parts.append(req.title)
    parts.append('''</title>
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.0/marked.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter','PingFang SC','Microsoft YaHei',sans-serif;background:#f8f9fa;color:#1a1a2e;line-height:1.8;padding:0}
.cover{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);color:#fff;padding:60px 40px;text-align:center}
.cover h1{font-size:2rem;font-weight:700;letter-spacing:2px;margin-bottom:8px}
.cover .sub{font-size:.9rem;opacity:.7}
.container{max-width:860px;margin:0 auto;padding:32px 24px}
h2{font-size:1.15rem;font-weight:700;color:#0f3460;margin:32px 0 16px;padding-left:14px;border-left:4px solid #e94560}
.stats-row{display:flex;gap:16px;flex-wrap:wrap;margin:16px 0}
.stat-card{flex:1;min-width:140px;background:#fff;border-radius:12px;padding:24px 20px;text-align:center;box-shadow:0 2px 12px rgba(0,0,0,.04);border:1px solid #f0f0f0}
.stat-card .num{font-size:2rem;font-weight:700;color:#e94560}
.stat-card .label{font-size:.8rem;color:#666;margin-top:4px}
.md-content{line-height:1.9;font-size:.9rem}
.md-content h1{font-size:1.3rem;color:#0f3460;margin:16px 0 8px}
.md-content h2{font-size:1.1rem;color:#1a1a2e;margin:14px 0 6px;border:none;padding:0}
.md-content h3{font-size:1rem;color:#333;margin:10px 0 4px}
.md-content table{width:100%;border-collapse:collapse;margin:10px 0;font-size:.85rem;box-shadow:0 2px 8px rgba(0,0,0,.04);border-radius:8px;overflow:hidden}
.md-content th{background:#1a1a2e;color:#fff;padding:10px 14px;text-align:left;font-size:.82rem}
.md-content td{padding:8px 14px;border-bottom:1px solid #eee;background:#fff}
.md-content tr:last-child td{border:none}
.md-content blockquote{border-left:3px solid #e94560;margin:8px 0;padding:6px 16px;background:#fff5f5;border-radius:0 8px 8px 0}
.md-content code{background:#f0f0f0;padding:2px 6px;border-radius:4px;font-size:.85rem}
.md-content pre{background:#1a1a2e;color:#e0e0e0;padding:16px;border-radius:10px;overflow-x:auto;font-size:.82rem}
.md-content strong{color:#0f3460}
.md-content ul,.md-content ol{padding-left:22px;margin:8px 0}
.md-content p{margin:6px 0}
.md-content hr{border:none;border-top:1px solid #ddd;margin:16px 0}
/* Chat items */
.chat-block{margin:20px 0}
.chat-item{display:flex;gap:12px;margin:12px 0;align-items:flex-start}
.chat-item.user{flex-direction:row-reverse}
.chat-role{font-size:.75rem;font-weight:700;padding:4px 14px;border-radius:20px;flex-shrink:0}
.user .chat-role{background:#e94560;color:#fff}
.ai-msg .chat-role{background:#0f3460;color:#fff}
.chat-content{background:#fff;border-radius:12px;padding:14px 20px;max-width:700px;box-shadow:0 2px 8px rgba(0,0,0,.04);font-size:.88rem}
.ai-msg .chat-content{border-top-left-radius:4px}
.user .chat-content{border-top-right-radius:4px;background:#f0f4ff}
.footer{margin-top:48px;padding:24px;text-align:center;color:#999;font-size:.78rem;border-top:1px solid #eee;background:#fff;border-radius:16px}
</style>
</head>
<body>
<div class="cover">
  <h1>⚖️ ''')
    parts.append(req.title)
    parts.append('''</h1>
  <div class="sub">生成时间: ''')
    parts.append(time.strftime("%Y-%m-%d %H:%M:%S"))
    parts.append(' · 数据来源: 法眼AI 裁判文书库 · ')
    parts.append(str(total))
    parts.append(' 条</div></div><div class="container">')

    # 一、数据概览
    parts.append('<div class="section"><h2>一、数据概览</h2><div class="stats-row">')
    parts.append('<div class="stat-card"><div class="num">' + f'{total:,}' + '</div><div class="label">案例总数</div></div>')
    parts.append('<div class="stat-card"><div class="num">' + f'{dashboard_cache["overview"]["civil_count"]:,}' + '</div><div class="label">民事案件</div></div>')
    parts.append('<div class="stat-card"><div class="num">' + f'{dashboard_cache["overview"]["criminal_count"]:,}' + '</div><div class="label">刑事案件</div></div>')
    parts.append('<div class="stat-card"><div class="num">' + str(dashboard_cache['overview']['case_types_count']) + '</div><div class="label">案由类型</div></div>')
    parts.append('</div></div>')

    # 二、案由分布
    parts.append('<div class="section"><h2>二、案由分布 Top 10</h2><table><tr><th>排名</th><th>案由</th><th>数量</th><th>占比</th></tr>')
    for i, item in enumerate(dashboard_cache.get("case_type_dist", [])[:10], 1):
        pct = item["value"] / total * 100
        parts.append(f'<tr><td>{i}</td><td>{item["name"]}</td><td>{item["value"]:,}</td><td>{pct:.1f}%</td></tr>')
    parts.append('</table></div>')

    # 三、判决结果分布
    parts.append('<div class="section"><h2>三、判决结果分布</h2><table><tr><th>判决类型</th><th>数量</th><th>占比</th></tr>')
    for item in dashboard_cache.get("judgment_dist", []):
        pct = item["value"] / total * 100
        parts.append(f'<tr><td>{item["name"]}</td><td>{item["value"]:,}</td><td>{pct:.1f}%</td></tr>')
    parts.append('</table></div>')

    # 四、高频关键词
    parts.append('<div class="section"><h2>四、高频关键词 Top 15</h2><table><tr><th>排名</th><th>关键词</th><th>频次</th></tr>')
    for i, item in enumerate(dashboard_cache.get("top_keywords", [])[:15], 1):
        parts.append(f'<tr><td>{i}</td><td>{item["name"]}</td><td>{item["value"]:,}</td></tr>')
    parts.append('</table></div>')

    # 五、对话分析（Markdown 原始内容不转义）
    if req.chat_history:
        parts.append('<div class="section"><h2>五、咨询对话与AI分析</h2>')
        for msg in req.chat_history[-20:]:
            role_class = "user" if msg.get("role") == "user" else "ai-msg"
            role_label = "👤 用户" if msg.get("role") == "user" else "🤖 AI"
            text = str(msg.get("content", ""))
            # 移除 <think> 标签
            text = _re.sub(r'<think>.*?</think>', '', text, flags=_re.DOTALL)
            # 不做 HTML 转义！保留原始 Markdown
            parts.append(f'<div class="chat-item {role_class}"><div class="chat-role">{role_label}</div><div class="chat-content md-content">')
            parts.append(text)
            parts.append('</div></div>')
        parts.append('</div>')

    # Footer
    parts.append(f'<div class="footer"><p>⚖️ 法眼AI 2.0 · 智能法律案例分析系统</p><p style="margin-top:4px">基于 {total:,} 条裁判文书数据 · GitHub: Mikenvala/fayan-ai-2.0</p></div>')
    parts.append('</div>')

    # Markdown 渲染脚本
    parts.append('''<script>
if(typeof marked !== 'undefined') {
  marked.setOptions({breaks:true,gfm:true});
  document.querySelectorAll('.md-content').forEach(function(el) {
    var raw = el.textContent || el.innerText || '';
    if(raw.trim()) {
      try { el.innerHTML = marked.parse(raw); } catch(e) {}
    }
  });
}
</script></body></html>''')

    report_html = ''.join(parts)

    import urllib.parse
    if req.format == "pdf":
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
                f.write(report_html); html_path = f.name
            pdf_path = html_path.replace('.html', '.pdf')
            chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
            subprocess.run([chrome,'--headless','--disable-gpu','--no-sandbox',
                '--print-to-pdf='+pdf_path,'--no-pdf-header-footer',
                'file://'+html_path], capture_output=True, timeout=30)
            with open(pdf_path, 'rb') as f: pdf_bytes = f.read()
            os.unlink(html_path); os.unlink(pdf_path)
            filename = urllib.parse.quote(req.title + ".pdf")
            return Response(content=pdf_bytes, media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"})
        except Exception as e:
            report_html = f"<p style='color:red'>PDF生成失败({e})</p>" + report_html

    filename = urllib.parse.quote(req.title + ".html")
    return HTMLResponse(content=report_html,
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"})


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
