#!/usr/bin/env python3
"""
法眼AI 2.0 · 统一后端 API
=========================
FastAPI 后端，提供三大模块：
  - /api/dashboard/*    数据仪表盘
  - /api/agent/*        多Agent智能问答
  - /api/report/*       报告生成

启动: python app.py
访问: http://localhost:8800
"""

import os
import sys
import json
import time
import asyncio
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
        from data import compute_dashboard_data
        dashboard_cache = compute_dashboard_data()
        print("   📊 仪表盘数据已计算")

    # 初始化多Agent
    from agents import FaYanMultiAgent
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
    debate: Optional[dict] = None
    format: Optional[str] = "html"  # html or pdf
    queries: Optional[List[str]] = []

# ============================================================
# 页面路由
# ============================================================
@app.get("/")
async def index():
    """主页 - 统一平台"""
    return FileResponse(os.path.join(ABS_DIR, "templates", "index.html"))

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
    import subprocess, tempfile, os, re as _re, markdown as _md
    import subprocess, tempfile, os, re as _re, markdown as _md

    def _wrap_box_drawing_blocks(text):
        """将包含 Unicode 框线字符的连续行转换为精美的 HTML 信息卡片"""
        BOX_CHARS = set('┌┐└┘├┤┬┴┼│─═╔╗╚╝╠╣╦╩╬║╭╮╯╰')
        # Strip only outer borders, keep inner structure
        import re as _re2
        lines = text.split('\n')
        result = []
        i = 0
        while i < len(lines):
            line = lines[i]
            has_box = any(ch in BOX_CHARS for ch in line)
            if has_box and line.strip():
                # Collect consecutive box-drawing lines
                box_lines = [line]
                j = i + 1
                while j < len(lines) and any(ch in BOX_CHARS for ch in lines[j]) and lines[j].strip():
                    box_lines.append(lines[j])
                    j += 1

                # Parse box content: clean borders, split key:value on first colon
                content_rows = []
                for bl in box_lines:
                    cleaned = bl
                    for ch in BOX_CHARS:
                        cleaned = cleaned.replace(ch, '')
                    cleaned = cleaned.strip()
                    if not cleaned:
                        continue
                    # Try to split on first ：or : as key-value
                    kv = _re2.split(r'[：:]', cleaned, maxsplit=1)
                    if len(kv) == 2 and kv[0].strip() and kv[1].strip():
                        content_rows.append(('kv', kv[0].strip(), kv[1].strip()))
                    else:
                        # No colon - whole line as a statement
                        content_rows.append(('text', cleaned, ''))

                if not content_rows:
                    i = j
                    continue

                # Build a beautiful HTML card
                card_parts = ['<div class="info-card">']
                # Card header icon
                card_parts.append('<div class="info-card-header">')
                card_parts.append('<span class="info-card-icon">⚖️</span>')
                card_parts.append('<span class="info-card-label">裁判要点</span>')
                card_parts.append('</div>')
                card_parts.append('<div class="info-card-body">')
                for idx_row, row in enumerate(content_rows):
                    row_type = row[0]
                    if row_type == 'kv':
                        card_parts.append('<div class="info-card-row">')
                        card_parts.append(f'<span class="info-card-key">{row[1]}</span>')
                        card_parts.append(f'<span class="info-card-val">{row[2]}</span>')
                        card_parts.append('</div>')
                    else:
                        card_parts.append(f'<div class="info-card-stmt"><span class="info-card-dot"></span>{row[1]}</div>')
                card_parts.append('</div>')
                card_parts.append('</div>')
                result.append('\n'.join(card_parts))
                i = j
            else:
                result.append(line)
                i += 1
        return '\n'.join(result)
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
<!-- Markdown rendered server-side -->
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
.md-content code{background:#f0f0f0;padding:2px 6px;border-radius:4px;font-size:.85rem;font-family:'SF Mono','Monaco','Menlo','Consolas',monospace}
.md-content pre{background:#1a1a2e;color:#e0e0e0;padding:16px;border-radius:10px;overflow-x:auto;font-size:.82rem;line-height:1.3;white-space:pre;font-family:'SF Mono','Monaco','Menlo','Consolas',monospace}
.md-content pre code{background:transparent;padding:0;font-size:inherit;color:inherit}
.info-card{
  background:#fff;border:1px solid #e8e8e8;border-radius:16px;
  margin:20px 0;overflow:hidden;
  box-shadow:0 4px 16px rgba(0,0,0,.06),0 1px 3px rgba(0,0,0,.04)
}
.info-card-header{
  background:linear-gradient(135deg,#7F1D1D,#991B1B);
  padding:14px 24px;display:flex;align-items:center;gap:10px
}
.info-card-icon{font-size:1.1rem}
.info-card-label{color:#fff;font-size:.88rem;font-weight:700;letter-spacing:.5px}
.info-card-body{padding:16px 24px}
.info-card-row{
  display:flex;align-items:flex-start;padding:12px 0;
  border-bottom:1px solid #f5f5f5
}
.info-card-row:last-child{border-bottom:none}
.info-card-key{
  font-size:.8rem;font-weight:600;color:var(--brand,#7F1D1D);
  min-width:90px;flex-shrink:0;padding-right:12px;
  background:#FEF2F2;padding:3px 10px;border-radius:6px;
  text-align:center;margin-right:12px
}
.info-card-val{
  font-size:.85rem;color:#1a1a2e;font-weight:500;line-height:1.6;flex:1
}
.info-card-stmt{
  font-size:.85rem;color:#1a1a2e;padding:8px 0 8px 24px;
  position:relative;line-height:1.6
}
.info-card-stmt .info-card-dot{
  position:absolute;left:8px;top:15px;
  width:6px;height:6px;border-radius:50%;background:#7F1D1D;opacity:.6
}
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
.debate-plaintiff-card{border-left:4px solid #dc2626;background:#fef2f2;padding:14px 18px;margin:12px 0;border-radius:0 8px 8px 0}
.debate-defendant-card{border-left:4px solid #2563eb;background:#eff6ff;padding:14px 18px;margin:12px 0;border-radius:0 8px 8px 0}
.debate-judge-card{background:linear-gradient(135deg,#f5f3ff,#faf5ff);border-left:4px solid #7c3aed;padding:18px 20px;margin:16px 0;border-radius:0 8px 8px 0}
.debate-role-label{font-weight:700;font-size:.82rem;margin-bottom:8px;display:flex;align-items:center;gap:6px}
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
            # 预处理：将 ASCII/Unicode 框线图包裹为代码块，保证等宽渲染
            text = _wrap_box_drawing_blocks(text)
            # 不做 HTML 转义！保留原始 Markdown
            parts.append(f'<div class="chat-item {role_class}"><div class="chat-role">{role_label}</div><div class="chat-content md-content">')
            rendered = _md.markdown(text, extensions=['tables', 'fenced_code', 'codehilite'])
            parts.append(rendered)
            parts.append('</div></div>')
        parts.append('</div>')

    # Footer
    # 六、模拟法庭辩论（如果有）
    if req.debate:
        parts.append('<div class="section"><h2>六、模拟法庭辩论</h2>')
        facts_text = req.debate.get("facts","")
        if facts_text:
            parts.append(f'<p style="color:#555;font-size:.85rem;margin-bottom:6px"><strong>📋 案件：</strong>{facts_text}</p>')
        focus_text = req.debate.get("focus","")
        if focus_text:
            parts.append(f'<p style="color:#555;font-size:.85rem;margin-bottom:18px"><strong>🎯 争议焦点：</strong>{focus_text}</p>')

        transcript = req.debate.get("transcript", [])
        if transcript:
            parts.append('<div style="margin:16px 0"><h3 style="color:#0f3460;font-size:1rem;margin-bottom:12px">🗣 辩论过程</h3>')
            for i, t in enumerate(transcript):
                role = t.get("role","")
                content = t.get("content","")
                round_num = t.get("round",0)
                if role == "plaintiff":
                    label = "👤 原告律师"
                    card_class = "debate-plaintiff-card"
                elif role == "defendant":
                    label = "👤 被告律师"
                    card_class = "debate-defendant-card"
                else:
                    label = "👤 发言人"
                    card_class = "debate-plaintiff-card"
                rendered_t = _md.markdown(content, extensions=['tables','fenced_code'])
                parts.append(f'<div class="{card_class}">')
                parts.append(f'<div class="debate-role-label">{label} · 第{round_num}轮</div>')
                parts.append(f'<div class="md-content" style="font-size:.85rem">{rendered_t}</div>')
                parts.append('</div>')
            parts.append('</div>')

        verdict = req.debate.get("verdict","")
        if verdict:
            parts.append('<div class="debate-judge-card">')
            parts.append('<div class="debate-role-label" style="color:#7c3aed;font-size:.88rem">⚖️ 法官终裁</div>')
            rendered_v = _md.markdown(verdict, extensions=['tables','fenced_code'])
            parts.append(f'<div class="md-content" style="font-size:.85rem">{rendered_v}</div>')
            parts.append('</div>')
        parts.append('</div>')

    # Footer
    parts.append(f'<div class="footer"><p>⚖️ 法眼AI 2.0 · 智能法律案例分析系统</p><p style="margin-top:4px">基于 {total:,} 条裁判文书数据 · GitHub: Mikenvala/fayan-ai-2.0</p></div>')
    parts.append('</div>')

    # Markdown 渲染脚本
    parts.append('''<!-- Markdown rendered server-side by Python markdown library --></body></html>''')

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
    from agents import get_retriever
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

# ============================================================
# 模拟法庭辩论 API (SSE 流式)
# ============================================================
DEBATE_CASES = [
    {
        "id": "p2p",
        "title": "P2P平台爆雷案",
        "facts": "某P2P平台虚构34个借款人信息，发布虚假标的，以20%年化收益为诱饵，向1586人吸收资金10.3亿元。所募资金未进入公司账户，由平台实控人周某个人掌控，用于购买房产、豪车、首饰等个人消费。案发后3.56亿元无法归还。",
        "focus": "周某的行为构成非法吸收公众存款罪还是集资诈骗罪？"
    },
    {
        "id": "ai_copyright",
        "title": "AI生成内容侵权案",
        "facts": "某AI公司使用大量网络文章训练模型，生成的AI内容与某知名博主的多篇文章高度相似。博主起诉AI公司侵犯著作权，要求赔偿100万元。AI公司辩称训练数据的使用属于'合理使用'，AI生成内容是'转换性使用'。",
        "focus": "AI公司使用网络文章训练模型是否构成著作权侵权？AI生成内容与原文相似是否侵权？"
    },
    {
        "id": "delivery_worker",
        "title": "外卖骑手工伤案",
        "facts": "外卖骑手张某在配送途中闯红灯被撞重伤，要求平台赔偿医疗费及伤残赔偿金80万元。平台认为张某是'个体工商户'，双方签订的是合作协议而非劳动合同，不构成劳动关系。张某每天工作12小时，接受平台派单管理，收入为唯一生活来源。",
        "focus": "张某与外卖平台是否构成劳动关系？平台是否应承担工伤赔偿责任？"
    }
]

PLAINTIFF_PROMPT = """你是一名资深原告律师。你的目标是为原告争取最大利益。

案件事实：{case_facts}
争议焦点：{dispute_focus}

对方观点（如有）：{opponent_argument}

请以原告律师身份发表辩论意见，要求：
1. 引用相关法条支持原告主张
2. 指出对方观点的漏洞
3. 提出具体的诉讼请求
4. 控制在300字以内，专业且有力"""

DEFENDANT_PROMPT = """你是一名资深被告律师。你的目标是为被告辩护，减轻责任。

案件事实：{case_facts}
争议焦点：{dispute_focus}

对方观点（如有）：{opponent_argument}

请以被告律师身份发表辩论意见，要求：
1. 引用法条和案例支持被告立场
2. 反驳对方观点
3. 提出减轻责任的理由
4. 控制在300字以内，专业且有力"""

JUDGE_PROMPT = """你是一名资深法官。请根据以下辩论记录做出裁判摘要。

案件事实：{case_facts}

辩论记录：
{debate_transcript}

请给出：
1. 案件定性（适用什么法律）
2. 双方的合理主张
3. 裁判倾向（更支持哪方，为什么）
4. 建议的和解/判决方案"""


class DebateOrchestrator:
    """多Agent辩论编排器"""
    def __init__(self):
        from langchain_openai import ChatOpenAI
        self.plaintiff_llm = ChatOpenAI(
            api_key=os.environ.get("MINIMAX_API_KEY", ""),
            base_url="https://api.minimax.chat/v1",
            model="MiniMax-M2.7", temperature=0.6, timeout=60, max_retries=1
        )
        self.defendant_llm = ChatOpenAI(
            api_key=os.environ.get("MINIMAX_API_KEY", ""),
            base_url="https://api.minimax.chat/v1",
            model="MiniMax-M2.7", temperature=0.6, timeout=60, max_retries=1
        )
        self.judge_llm = ChatOpenAI(
            api_key=os.environ.get("MINIMAX_API_KEY", ""),
            base_url="https://api.minimax.chat/v1",
            model="MiniMax-M2.7", temperature=0.3, timeout=60, max_retries=1
        )

    async def run_stream(self, facts: str, focus: str, rounds: int = 3):
        """流式运行辩论，逐轮 yield JSON 事件"""
        plaintiff_arg = ""
        defendant_arg = ""
        transcript = []

        yield json.dumps({"type": "start", "facts": facts, "focus": focus, "rounds": rounds}) + "\n"

        for rnd in range(1, rounds + 1):
            # 原告发言
            yield json.dumps({"type": "status", "round": rnd, "speaker": "plaintiff", "stage": "thinking"}) + "\n"
            p_prompt = PLAINTIFF_PROMPT.format(
                case_facts=facts, dispute_focus=focus,
                opponent_argument=defendant_arg if defendant_arg else "（首轮发言）"
            )
            try:
                p_resp = self.plaintiff_llm.invoke(p_prompt)
                plaintiff_arg = p_resp.content
                # Strip <think> tags
                import re as _re
                plaintiff_arg = _re.sub(r'<think>.*?</think>', '', plaintiff_arg, flags=_re.DOTALL).strip()
            except Exception as e:
                plaintiff_arg = f"[发言失败: {e}]"

            transcript.append({"round": rnd, "role": "plaintiff", "content": plaintiff_arg})
            yield json.dumps({"type": "speech", "round": rnd, "role": "plaintiff", "label": "👤 原告律师", "content": plaintiff_arg}) + "\n"
            await asyncio.sleep(0.3)

            # 被告发言
            yield json.dumps({"type": "status", "round": rnd, "speaker": "defendant", "stage": "thinking"}) + "\n"
            d_prompt = DEFENDANT_PROMPT.format(
                case_facts=facts, dispute_focus=focus,
                opponent_argument=plaintiff_arg
            )
            try:
                d_resp = self.defendant_llm.invoke(d_prompt)
                defendant_arg = d_resp.content
                import re as _re
                defendant_arg = _re.sub(r'<think>.*?</think>', '', defendant_arg, flags=_re.DOTALL).strip()
            except Exception as e:
                defendant_arg = f"[发言失败: {e}]"

            transcript.append({"round": rnd, "role": "defendant", "content": defendant_arg})
            yield json.dumps({"type": "speech", "round": rnd, "role": "defendant", "label": "👤 被告律师", "content": defendant_arg}) + "\n"
            await asyncio.sleep(0.3)

        # 裁判总结
        yield json.dumps({"type": "status", "round": 0, "speaker": "judge", "stage": "deliberating"}) + "\n"
        transcript_text = "\n\n".join([
            f"第{t['round']}轮 - {'原告律师' if t['role'] == 'plaintiff' else '被告律师'}:\n{t['content']}"
            for t in transcript
        ])
        j_prompt = JUDGE_PROMPT.format(case_facts=facts, debate_transcript=transcript_text)
        try:
            j_resp = self.judge_llm.invoke(j_prompt)
            verdict = j_resp.content
            import re as _re
            verdict = _re.sub(r'<think>.*?</think>', '', verdict, flags=_re.DOTALL).strip()
        except Exception as e:
            verdict = f"[裁判失败: {e}]"

        yield json.dumps({"type": "verdict", "label": "⚖️ 法官裁判", "content": verdict, "transcript": transcript}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"


@app.get("/api/debate/cases")
async def get_debate_cases():
    """获取预设辩论案例列表"""
    return JSONResponse([{"id": c["id"], "title": c["title"], "focus": c["focus"]} for c in DEBATE_CASES])


class DebateRequest(BaseModel):
    case_id: str = "p2p"
    facts: str = ""
    focus: str = ""
    rounds: int = 3

@app.post("/api/debate/run")
async def run_debate(req: DebateRequest):
    """运行多Agent辩论 (SSE流式)"""
    case_id = req.case_id
    custom_facts = req.facts
    custom_focus = req.focus
    rounds = min(req.rounds, 5)

    # 查找预设案例或使用自定义
    case = next((c for c in DEBATE_CASES if c["id"] == case_id), None)
    if custom_facts and custom_focus:
        facts = custom_facts
        focus = custom_focus
    elif case:
        facts = case["facts"]
        focus = case["focus"]
    else:
        facts = DEBATE_CASES[0]["facts"]
        focus = DEBATE_CASES[0]["focus"]

    orchestrator = DebateOrchestrator()

    async def event_stream():
        async for line in orchestrator.run_stream(facts, focus, rounds):
            yield f"data: {line}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8800, log_level="info")

