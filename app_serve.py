"""
法眼AI - 轻量启动（仅前端页面 + API 路由）
启动: python app_serve.py
"""
import os, sys, json

# 尝试导入重模块（可能慢）
_heavy_loaded = False
_fayan = None
_init_error = "正在初始化..."

def _lazy_load():
    global _heavy_loaded, _fayan, _init_error
    if _heavy_loaded:
        return
    try:
        from fayan_api import FaYanLegal, MINIMAX_BASE_URL, LLM_MODEL, CASES_JSON, CRIMINAL_CASES_JSON
        api_key = os.environ.get("MINIMAX_API_KEY", "")
        if not api_key:
            _init_error = "MINIMAX_API_KEY 未设置"
            _heavy_loaded = True
            return
        _fayan = FaYanLegal(
            api_key=api_key, base_url=MINIMAX_BASE_URL, model=LLM_MODEL,
            cases_json=CASES_JSON, criminal_json=CRIMINAL_CASES_JSON
        )
        _init_error = None
    except Exception as e:
        _init_error = str(e)
    _heavy_loaded = True

from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/status")
def status():
    _lazy_load()
    return jsonify({
        "ok": _init_error is None,
        "error": _init_error,
        "cases_count": _fayan.retriever.civil_count if _fayan else 0,
        "criminal_count": _fayan.retriever.criminal_count if _fayan else 0,
        "model": "MiniMax-M2.7",
    })

@app.route("/api/analyze", methods=["POST"])
def analyze():
    _lazy_load()
    if _init_error:
        return jsonify({"error": _init_error}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "请求体为空"}), 400

    case_text = data.get("case_text", "").strip()
    if not case_text or len(case_text) < 10:
        return jsonify({"error": "案情描述需至少10字"}), 400
    if len(case_text) > 5000:
        return jsonify({"error": "案情描述不超过5000字"}), 400

    try:
        result = _fayan.analyze(
            case_text=case_text,
            amount=float(data.get("amount", 0) or 0),
            party_count=int(data.get("party_count", 2) or 2),
            has_evidence_gap=bool(data.get("has_evidence_gap", False)),
            has_criminal_cross=bool(data.get("has_criminal_cross", False)),
        )
        return jsonify(_fayan.to_dict(result, result.case_kind))
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"分析失败: {str(e)}"}), 500

@app.route("/api/retrieve", methods=["POST"])
def retrieve():
    _lazy_load()
    if _init_error:
        return jsonify({"error": _init_error}), 500

    data = request.get_json()
    query = data.get("query", "").strip()
    top_k = min(int(data.get("top_k", 5) or 5), 10)

    if not query:
        return jsonify({"error": "query 不能为空"}), 400

    try:
        results = _fayan.retriever.retrieve(query, k=top_k)
        return jsonify({
            "query": query, "total": len(results),
            "results": [{
                "case_number": r["case"].get("case_number", r["case"].get("id", "")),
                "title": r["case"].get("title", ""),
                "court": r["case"].get("court", "N/A"),
                "cause_of_action": r["case"].get("cause_of_action", "N/A"),
                "ruling_points": r["case"].get("metadata", {}).get("ruling_points", "")[:300],
                "score": round(r["score"], 3),
            } for r in results]
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/classify", methods=["POST"])
def classify():
    data = request.get_json()
    if not data:
        return jsonify({"error": "请求体为空"}), 400
    case_text = data.get("case_text", "").strip()
    if len(case_text) < 5:
        return jsonify({"error": "案情描述过短"}), 400

    _lazy_load()
    from fayan_api import CaseClassifier
    try:
        result = CaseClassifier.classify(case_text)
        return jsonify(result)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return Response("ok", mimetype="text/plain")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"法眼AI 启动中...")
    print(f"访问地址: http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
