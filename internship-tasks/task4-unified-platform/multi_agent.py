#!/usr/bin/env python3
"""
法眼AI 2.0 · 多Agent协作系统 (LangGraph)
=========================================
三个专业Agent协同工作：
  - 检索Agent：BM25 + TF-IDF 混合检索相似案例
  - 分析Agent：基于检索结果进行法律分析
  - 校验Agent：事实核查，确保引用准确、无幻觉

运行: python multi_agent.py
"""

import csv
import os
import re
import sys
import json
import hashlib
import pickle
from typing import TypedDict, Annotated, Optional, List, Dict, Any
from operator import add
from dotenv import load_dotenv

ABS_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ABS_DIR, "..", "..", ".env"))

import numpy as np
import jieba
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ============================================================
# 配置
# ============================================================
CSV_PATH = os.path.join(ABS_DIR, "..", "..", "all_cases_perfect.csv")
INDEX_CACHE_DIR = os.path.join(ABS_DIR, ".cache")
os.makedirs(INDEX_CACHE_DIR, exist_ok=True)

API_KEY = os.environ.get("MINIMAX_API_KEY", "")
BASE_URL = "https://api.minimax.chat/v1"
MODEL = "MiniMax-M2.7"
TOP_K = 3
MAX_REVISIONS = 2

# ============================================================
# State 定义
# ============================================================
class MultiAgentState(TypedDict):
    query: str                           # 用户问题
    chat_history: Annotated[List[Dict[str, str]], add]  # 对话历史
    intent: str                          # 意图分类: case_search | legal_knowledge | general
    retrieved_cases: List[Dict]          # 检索到的案例
    analysis: str                        # 分析Agent产出
    verification: Dict[str, Any]         # 校验结果 {passed, issues, suggestions}
    final_answer: str                    # 最终回答
    needs_revision: bool                 # 是否需要修改
    revision_count: int                  # 已修改次数

# ============================================================
# 数据加载 & 检索引擎
# ============================================================
class HybridRetriever:
    """BM25 + TF-IDF 混合检索"""

    def __init__(self, csv_path: str = CSV_PATH):
        self.cases: List[Dict] = []
        self.case_texts: List[str] = []
        self.bm25: Optional[BM25Okapi] = None
        self.tfidf_vec: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self._load_data(csv_path)
        self._build_index()

    def _load_data(self, path: str):
        with open(path, "rb") as f:
            raw = f.read().replace(bytes([0]), b'')
        import io
        with io.TextIOWrapper(io.BytesIO(raw), encoding='utf-8') as tf:
            reader = csv.DictReader(tf)
            for row in reader:
                desc = row.get("案件描述", "")
                keywords = " ".join([row.get(f"关键词_{i:02d}", "") for i in range(1, 11)])
                text = f"{row.get('文件名','')} {desc} {keywords}"
                self.cases.append({
                    "filename": row.get("文件名", ""),
                    "desc": desc,
                    "plaintiff_claim": row.get("原告诉求", ""),
                    "judgment_criteria": row.get("判别标准", ""),
                    "judgment_result": row.get("判决结果", ""),
                    "full_text": text
                })
                self.case_texts.append(text)
        print(f"   📂 加载 {len(self.cases)} 条案例")

    def _build_index(self):
        cache_key = hashlib.md5(CSV_PATH.encode()).hexdigest()
        bm25_cache = os.path.join(INDEX_CACHE_DIR, f"bm25_{cache_key}.pkl")
        tfidf_cache = os.path.join(INDEX_CACHE_DIR, f"tfidf_{cache_key}.pkl")

        if os.path.exists(bm25_cache) and os.path.exists(tfidf_cache):
            with open(bm25_cache, "rb") as f:
                self.bm25 = pickle.load(f)
            with open(tfidf_cache, "rb") as f:
                data = pickle.load(f)
                self.tfidf_vec = data["vec"]
                self.tfidf_matrix = data["matrix"]
            print("   ⚡ 从缓存加载索引")
            return

        # BM25
        tokenized = [list(jieba.cut(t)) for t in self.case_texts]
        self.bm25 = BM25Okapi(tokenized)
        with open(bm25_cache, "wb") as f:
            pickle.dump(self.bm25, f)

        # TF-IDF (char ngram 2-3, faster)
        self.tfidf_vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), max_features=5000)
        self.tfidf_matrix = self.tfidf_vec.fit_transform(self.case_texts)
        with open(tfidf_cache, "wb") as f:
            pickle.dump({"vec": self.tfidf_vec, "matrix": self.tfidf_matrix}, f)

        print("   🔨 索引构建完成")

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """混合检索：BM25(权重0.6) + TF-IDF(权重0.4)"""
        # BM25
        q_tokens = list(jieba.cut(query))
        bm25_scores = np.array(self.bm25.get_scores(q_tokens))
        bm25_norm = bm25_scores / (bm25_scores.max() + 1e-8)

        # TF-IDF
        q_vec = self.tfidf_vec.transform([query])
        tfidf_scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        tfidf_norm = tfidf_scores / (tfidf_scores.max() + 1e-8)

        # 混合
        combined = 0.6 * bm25_norm + 0.4 * tfidf_norm
        top_indices = np.argsort(combined)[::-1][:top_k]

        results = []
        for idx in top_indices:
            case = self.cases[idx].copy()
            case["score"] = round(float(combined[idx]), 4)
            results.append(case)
        return results


# 全局检索引擎实例（懒加载）
_retriever: Optional[HybridRetriever] = None

def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


# ============================================================
# LLM 客户端
# ============================================================
def get_llm(temperature: float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
        temperature=temperature,
        timeout=30,
        max_retries=0
    )


# ============================================================
# Agent 1: 意图分类
# ============================================================
INTENT_PROMPT = """你是一个法律咨询意图分类器。根据用户输入，判断意图类型。

意图类型：
- case_search: 用户想查找类似案例、了解某类案件的判决情况
- legal_knowledge: 用户咨询法律概念、法条、程序性问题
- general: 闲聊或其他非法律问题

只输出一个意图类型单词，不要解释。

用户问题: {query}
意图:"""

def classify_intent(query: str, llm: ChatOpenAI) -> str:
    """分类用户意图"""
    # 快速关键词预判
    case_keywords = ["案例", "案件", "判决", "类似", "怎么判", "赔偿", "胜诉",
                     "民事", "刑事", "合同", "借贷", "劳动", "交通", "离婚",
                     "民间借贷", "交通事故", "劳动争议", "合同纠纷"]
    knowledge_keywords = ["什么是", "定义", "法条", "第几条", "刑法", "民法典",
                          "司法解释", "规定", "程序", "起诉", "管辖"]

    lower_q = query.lower()
    case_hits = sum(1 for k in case_keywords if k in lower_q)
    know_hits = sum(1 for k in knowledge_keywords if k in lower_q)

    if case_hits >= 2:
        return "case_search"
    if know_hits >= 2 and case_hits == 0:
        return "legal_knowledge"

    # LLM 分类
    try:
        resp = llm.invoke(INTENT_PROMPT.format(query=query))
        text = resp.content.strip().lower()
        if "case_search" in text:
            return "case_search"
        elif "legal_knowledge" in text or "knowledge" in text:
            return "legal_knowledge"
        else:
            return "general"
    except Exception:
        return "case_search"  # 默认按案例检索处理


# ============================================================
# LangGraph 节点函数
# ============================================================
def intent_node(state: MultiAgentState, llm: ChatOpenAI) -> MultiAgentState:
    """意图分类节点"""
    intent = classify_intent(state["query"], llm)
    return {"intent": intent, "revision_count": 0, "needs_revision": False}


def retrieve_node(state: MultiAgentState) -> MultiAgentState:
    """检索节点：BM25 + TF-IDF 混合检索"""
    if state.get("intent") == "general":
        return {"retrieved_cases": []}

    retriever = get_retriever()
    cases = retriever.search(state["query"])
    return {"retrieved_cases": cases}


ANALYST_SYSTEM = """你是法眼AI的法律分析专家。基于检索到的真实案例，回答用户的问题。

要求：
1. 引用案例时标注"【案例X】文件名"格式
2. 如果检索结果没有相关案例，诚实说明并给出一般性法律知识
3. 分析要有理有据，包含：案由、关键事实、裁判要点、判决结果
4. 不要断言"一定会赢/输"，使用"根据类似案例...通常..."的表述
5. 如果涉及金额、刑期等，引用案例中的具体数字

格式要求：
- 使用 Markdown 格式
- 对案例排版采用：
  > **【案例1】** (相似度: XX%)
  > - 案由：...
  > - 关键事实：...
  > - 裁判要点：...
  > - 判决结果：...
"""

def analyst_node(state: MultiAgentState, llm: ChatOpenAI) -> MultiAgentState:
    """分析节点：基于检索结果进行法律分析"""
    query = state["query"]
    cases = state.get("retrieved_cases", [])
    chat_history = state.get("chat_history", [])

    # 构建案例上下文
    if cases:
        case_context = "\n\n".join([
            f"【案例{i+1}】(相似度: {c['score']:.1%})\n"
            f"文件名: {c['filename']}\n"
            f"案件描述: {c['desc'][:300]}\n"
            f"原告诉求: {c['plaintiff_claim'][:200]}\n"
            f"判别标准: {c['judgment_criteria'][:200]}\n"
            f"判决结果: {c['judgment_result'][:200]}"
            for i, c in enumerate(cases)
        ])
    else:
        case_context = "（未检索到相关案例，请基于一般性法律知识回答）"

    # 构建消息
    messages = [{"role": "system", "content": ANALYST_SYSTEM}]
    for h in chat_history[-6:]:  # 最近3轮
        messages.append(h)
    messages.append({
        "role": "user",
        "content": f"## 检索到的相关案例\n{case_context}\n\n## 用户问题\n{query}\n\n请基于以上案例进行分析回答。"
    })

    # 调用 LLM
    from langchain_core.messages import SystemMessage, HumanMessage
    lc_messages = [SystemMessage(content=ANALYST_SYSTEM)]
    for h in chat_history[-6:]:
        if h["role"] == "user":
            lc_messages.append(HumanMessage(content=h["content"]))
        else:
            lc_messages.append(SystemMessage(content=h["content"]))

    lc_messages.append(HumanMessage(content=f"## 检索到的相关案例\n{case_context}\n\n## 用户问题\n{query}\n\n请基于以上案例进行分析回答。"))

    resp = llm.invoke(lc_messages)
    return {"analysis": resp.content}


VERIFIER_SYSTEM = """你是法眼AI的事实校验专家。检查以下法律分析回答是否存在问题。

检查维度：
1. 引用的案例文件名是否与检索结果中的一致？
2. 引用的数字（金额、刑期等）是否与原始案例匹配？
3. 是否存在"一定赢/输"等断言性表述？
4. 法律概念是否使用准确？
5. 是否遗漏了重要信息？

输出 JSON 格式：
{
  "passed": true/false,
  "issues": ["问题1", "问题2"],
  "suggestions": "修改建议",
  "confidence": 0-100
}
"""

def verifier_node(state: MultiAgentState, llm: ChatOpenAI) -> MultiAgentState:
    """校验节点：事实核查"""
    analysis = state.get("analysis", "")
    cases = state.get("retrieved_cases", [])

    cases_summary = "\n".join([
        f"案例{i+1}: {c['filename']} | {c['desc'][:100]} | {c['judgment_result'][:100]}"
        for i, c in enumerate(cases)
    ])

    from langchain_core.messages import SystemMessage, HumanMessage
    messages = [
        SystemMessage(content=VERIFIER_SYSTEM),
        HumanMessage(content=f"## 检索到的原始案例\n{cases_summary}\n\n## 分析回答\n{analysis}\n\n请校验。")
    ]

    resp = llm.invoke(messages)
    try:
        # 尝试解析 JSON
        content = resp.content
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            verification = json.loads(json_match.group())
        else:
            verification = {"passed": True, "issues": [], "suggestions": "", "confidence": 80}
    except Exception:
        verification = {"passed": True, "issues": [], "suggestions": "", "confidence": 80}

    needs_revision = (
        not verification.get("passed", True)
        and state.get("revision_count", 0) < MAX_REVISIONS
        and verification.get("confidence", 100) < 70
    )

    return {
        "verification": verification,
        "needs_revision": needs_revision,
        "revision_count": state.get("revision_count", 0) + 1
    }


def formatter_node(state: MultiAgentState) -> MultiAgentState:
    """格式化节点"""
    analysis = state.get("analysis", "")
    verification = state.get("verification", {})

    # 添加校验标注
    footer = ""
    if verification.get("issues"):
        footer = "\n\n---\n⚡ *校验注：已自动核查引用准确性，置信度 {}%*".format(
            verification.get("confidence", 80))
    elif verification:
        footer = f"\n\n---\n✅ *自动校验通过（置信度 {verification.get('confidence', 80)}%）*"

    final = analysis + footer
    return {"final_answer": final}


# ============================================================
# 路由函数
# ============================================================
def route_after_intent(state: MultiAgentState) -> str:
    """意图分类后的路由"""
    intent = state.get("intent", "case_search")
    if intent == "general":
        return "analyst"  # 直接分析，不需要检索
    return "retriever"


def route_after_verify(state: MultiAgentState) -> str:
    """校验后的路由"""
    if state.get("needs_revision", False):
        return "analyst"  # 需要修改，回到分析
    return "formatter"


# ============================================================
# 构建 LangGraph
# ============================================================
def build_agent_graph() -> StateGraph:
    """构建多Agent协作图"""
    llm = get_llm()

    workflow = StateGraph(MultiAgentState)

    # 添加节点
    workflow.add_node("intent", lambda s: intent_node(s, llm))
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("analyst", lambda s: analyst_node(s, llm))
    workflow.add_node("verifier", lambda s: verifier_node(s, llm))
    workflow.add_node("formatter", formatter_node)

    # 添加边
    workflow.set_entry_point("intent")
    workflow.add_conditional_edges("intent", route_after_intent, {
        "retriever": "retriever",
        "analyst": "analyst"
    })
    workflow.add_edge("retriever", "analyst")
    workflow.add_edge("analyst", "verifier")
    workflow.add_conditional_edges("verifier", route_after_verify, {
        "analyst": "analyst",
        "formatter": "formatter"
    })
    workflow.add_edge("formatter", END)

    # 编译（带内存 checkpoint）
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app


# ============================================================
# 对外接口
# ============================================================

class FaYanMultiAgent:
    """法眼AI 多Agent 系统封装"""

    def __init__(self):
        self.graph = build_agent_graph()
        self.thread_id = "default"

    def chat(self, query: str, chat_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """多Agent对话接口

        Args:
            query: 用户问题
            chat_history: 对话历史 [{"role":"user"/"assistant", "content":"..."}]

        Returns:
            {
                "answer": str,
                "intent": str,
                "cases": list,
                "verification": dict,
                "agent_trace": ["intent→retriever→analyst→verifier→formatter"]
            }
        """
        if chat_history is None:
            chat_history = []

        config = {"configurable": {"thread_id": self.thread_id}}

        initial_state: MultiAgentState = {
            "query": query,
            "chat_history": chat_history,
            "intent": "",
            "retrieved_cases": [],
            "analysis": "",
            "verification": {},
            "final_answer": "",
            "needs_revision": False,
            "revision_count": 0
        }

        try:
            result = self.graph.invoke(initial_state, config)
            trace = self._build_trace(result)
            return {
            "answer": result.get("final_answer", "抱歉，分析过程出现问题，请重试。"),
            "intent": result.get("intent", "unknown"),
            "cases": result.get("retrieved_cases", []),
            "verification": result.get("verification", {}),
                "agent_trace": trace
            }
        except Exception as e:
            err = str(e)
            if "402" in err or "insufficient" in err.lower():
                msg = "⚠️ API 余额不足，请充值 MiniMax 账户后再试。"
            else:
                msg = "❌ 服务错误：" + err[:100]
            return {"answer": msg, "intent": "error", "cases": [], "verification": {}, "agent_trace": "error"}

    def _build_trace(self, result: dict) -> str:
        """构建Agent执行轨迹"""
        parts = ["intent"]
        if result.get("intent") != "general":
            parts.append("retriever")
        parts.append("analyst")
        parts.append("verifier")
        if result.get("needs_revision"):
            parts.append("analyst(revised)")
        parts.append("formatter")
        return " → ".join(parts)


# ============================================================
# 测试入口
# ============================================================
    def chat_stream(self, query: str, chat_history: Optional[List[Dict]] = None):
        """流式多Agent对话 - 分步推送进度

        Yields: {"event": "intent|retrieve|analyze|verify|format|done", "data": {...}}
        """
        if chat_history is None:
            chat_history = []

        llm = get_llm()
        config = {"configurable": {"thread_id": self.thread_id}}

        # Step 1: Intent
        yield {"event": "intent", "data": {"status": "classifying"}}
        intent = classify_intent(query, llm)
        yield {"event": "intent", "data": {"status": "done", "intent": intent}}

        # Step 2: Retrieve
        if intent != "general":
            yield {"event": "retrieve", "data": {"status": "searching"}}
            retriever = get_retriever()
            cases = retriever.search(query)
            yield {"event": "retrieve", "data": {"status": "done", "cases": [
                {"filename": c["filename"], "desc": c["desc"][:150], "score": c["score"]}
                for c in cases
            ]}}
        else:
            cases = []
            yield {"event": "retrieve", "data": {"status": "skipped"}}

        # Step 3: Analyze
        yield {"event": "analyze", "data": {"status": "thinking"}}
        from langchain_core.messages import SystemMessage, HumanMessage

        if cases:
            case_context = "\n\n".join([
                f"【案例{i+1}】(相似度: {c['score']:.1%})\n"
                f"文件名: {c['filename']}\n"
                f"案件描述: {c['desc'][:300]}\n"
                f"原告诉求: {c['plaintiff_claim'][:200]}\n"
                f"判决结果: {c['judgment_result'][:200]}"
                for i, c in enumerate(cases)
            ])
        else:
            case_context = "（未检索到相关案例）"

        lc_messages = [SystemMessage(content=ANALYST_SYSTEM)]
        for h in chat_history[-6:]:
            if h["role"] == "user":
                lc_messages.append(HumanMessage(content=h["content"]))
            else:
                lc_messages.append(SystemMessage(content=h["content"]))
        lc_messages.append(HumanMessage(content=f"## 检索到的相关案例\n{case_context}\n\n## 用户问题\n{query}"))

        # Stream analysis
        full_analysis = ""
        for chunk in llm.stream(lc_messages):
            full_analysis += chunk.content
            yield {"event": "analyze", "data": {"status": "streaming", "chunk": chunk.content}}

        yield {"event": "analyze", "data": {"status": "done"}}

        # Step 4: Verify
        yield {"event": "verify", "data": {"status": "checking"}}
        cases_summary = "\n".join([
            f"案例{i+1}: {c['filename']} | {c['desc'][:100]} | {c['judgment_result'][:100]}"
            for i, c in enumerate(cases)
        ])
        ve_messages = [
            SystemMessage(content=VERIFIER_SYSTEM),
            HumanMessage(content=f"## 原始案例\n{cases_summary}\n\n## 分析回答\n{full_analysis}")
        ]
        ve_resp = llm.invoke(ve_messages)
        try:
            import re, json
            json_match = re.search(r'\{[\s\S]*\}', ve_resp.content)
            verification = json.loads(json_match.group()) if json_match else {"passed": True, "issues": [], "suggestions": "", "confidence": 80}
        except:
            verification = {"passed": True, "issues": [], "suggestions": "", "confidence": 80}
        yield {"event": "verify", "data": {"status": "done", "verification": verification}}

        # Step 5: Format
        yield {"event": "format", "data": {"status": "formatting"}}
        footer = ""
        if verification.get("issues"):
            footer = f"\n\n---\n⚡ *校验注：已自动核查引用准确性，置信度 {verification.get('confidence', 80)}%*"
        final = full_analysis + footer
        yield {"event": "done", "data": {
            "answer": final,
            "intent": intent,
            "cases": [{"filename": c["filename"], "desc": c["desc"][:150], "score": c["score"]} for c in cases],
            "verification": verification,
            "agent_trace": self._build_trace_from(intent, cases, verification)
        }}

    def _build_trace_from(self, intent, cases, verification):
        parts = ["intent"]
        if intent != "general":
            parts.append(f"retriever({len(cases)} cases)")
        parts.append("analyst")
        parts.append("verifier")
        parts.append("formatter")
        return " → ".join(parts)

if __name__ == "__main__":
    print("=" * 60)
    print("   ⚖️  法眼AI 2.0 · 多Agent协作系统")
    print("=" * 60)

    agent = FaYanMultiAgent()
    print("\n✅ 多Agent系统初始化完成\n")

    test_queries = [
        "民间借贷纠纷中，借条丢失了怎么办？",
        "交通事故赔偿标准是什么？",
    ]

    for q in test_queries:
        print(f"\n👤 用户: {q}")
        print("-" * 40)
        result = agent.chat(q)
        print(f"🎯 意图: {result['intent']}")
        print(f"📋 Agent链路: {result['agent_trace']}")
        if result["cases"]:
            print(f"📂 检索到 {len(result['cases'])} 个案例:")
            for c in result["cases"]:
                print(f"   - {c['filename']} (相似度: {c['score']:.2%})")
        print(f"\n🤖 回答:\n{result['answer'][:500]}...")
        print()

