#!/usr/bin/env python3
"""
法眼AI · 法律案件智能推荐 Agent
================================
基于 LangChain + BM25/TF-IDF 检索 + MiniMax LLM 的多轮对话 Agent。

功能:
- 意图识别: 区分"案情检索"和"法律知识咨询"
- 案例检索: BM25 + TF-IDF 混合检索 Top 3 相似案例
- 格式化输出: 案由/关键事实/裁判要点/判决结果/相似度
- 多轮对话: 支持追问和上下文记忆
- 防幻觉: 检索不到时诚实告知

运行: python case_agent.py
"""

import csv, os, re, sys, json
from collections import defaultdict
from typing import Optional
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))

import numpy as np
import jieba
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ============================================================
# 配置
# ============================================================
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "all_cases_perfect.csv")
INDEX_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
TOP_K = 3
MAX_CASE_DISPLAY_CHARS = 250

API_KEY = os.environ.get("MINIMAX_API_KEY", "")
BASE_URL = "https://api.minimax.chat/v1"
MODEL = "MiniMax-M2.7"

# ============================================================
# 1. 案例数据加载
# ============================================================

def load_cases(csv_path: str) -> list[dict]:
    """加载 CSV 案例库，返回结构化案例列表"""
    print(f"📂 加载案例数据: {csv_path}")
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read().replace('\x00', '')
    reader = csv.DictReader(content.splitlines())

    cases = []
    for row in reader:
        fn = row.get('文件名', '')
        # 提取案号
        case_num_match = re.search(r'(\d{4})\]\s*(\d+)_', fn)
        year = case_num_match.group(1) if case_num_match else ''
        num  = case_num_match.group(2) if case_num_match else ''
        case_id = f"({year}){num}"

        # 提取案由（从文件名路径的目录名）
        parts = fn.replace('\\', '/').split('/')
        folder = parts[-2] if len(parts) >= 2 else ''
        folder = re.sub(r'^\(NEW\).*?[：:]', '', folder)
        folder = re.sub(r'^\d+\s*[、\s]*', '', folder)
        folder = re.sub(r'\s*超清[对照]*$|超高清$|_\s*扫描版$|^\d{4}年度案例', '', folder)
        cause = folder.strip('_ ')

        # 从文件名取标题
        title_match = re.search(r'(\d+_\s*)?(.+?)\.txt$', fn.split('/')[-1])
        title = title_match.group(2) if title_match else fn.split('/')[-1].replace('.txt','')

        # 关键词
        kws = []
        for i in range(1, 11):
            kw = row.get(f'关键词_{i:02d}', '').strip()
            if kw: kws.append(kw)

        # 判决结果摘要（取前100字）
        judgment = row.get('判决结果', '').strip()

        cases.append({
            'id': case_id,
            'title': title,
            'cause': cause,
            'description': row.get('案件描述', '')[:500],
            'ruling_points': row.get('判别标准', '')[:500],
            'judgment': judgment[:150],
            'keywords': kws,
            '_search_text': '',
        })

    # 构建检索文本
    for c in cases:
        c['_search_text'] = (
            f"{c['cause']} {c['cause']} "  # 案由加权
            f"{' '.join(c['keywords'][:5])} "
            f"{c['description'][:300]} "
            f"{c['ruling_points'][:200]}"
        )

    print(f"   共加载 {len(cases)} 条案例")
    return cases


# ============================================================
# 2. 检索引擎
# ============================================================

def jieba_tokenizer(text: str) -> list[str]:
    """中文分词"""
    text = re.sub(r'[^\u4e00-\u9fff\w]', ' ', text)
    return [w for w in jieba.lcut(text) if len(w.strip()) >= 1]


class CaseRetriever:
    """BM25 + TF-IDF 混合检索器"""

    def __init__(self, cases: list[dict]):
        self.cases = cases
        corpus = [c['_search_text'] for c in cases]

        # BM25
        tokenized = [jieba_tokenizer(t) for t in corpus]
        self.bm25 = BM25Okapi(tokenized)

        # TF-IDF
        self.tfidf_vec = TfidfVectorizer(
            tokenizer=jieba_tokenizer, max_features=2000,
            token_pattern=None, ngram_range=(1, 1)
        )
        print("   TF-IDF构建中...")
        self.tfidf_matrix = self.tfidf_vec.fit_transform(corpus)

        print(f"   ✅ 索引就绪: BM25 + TF-IDF({self.tfidf_matrix.shape[1]}维, {self.tfidf_matrix.shape[0]}条)")

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """BM25 + TF-IDF 混合检索，返回 Top-K 案例"""
        # BM25 得分
        q_toks = jieba_tokenizer(query)
        bm25_scores = np.array(self.bm25.get_scores(q_toks))
        # 归一化
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()

        # TF-IDF cosine 相似度
        q_vec = self.tfidf_vec.transform([query])
        tfidf_scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        # 混合得分（BM25权重0.6, TF-IDF权重0.4）
        combined = 0.6 * bm25_scores + 0.4 * tfidf_scores

        top_idx = np.argsort(combined)[-top_k:][::-1]
        results = []
        for i in top_idx:
            if combined[i] < 0.01:
                break
            results.append({**self.cases[i], 'score': round(float(combined[i]), 4)})
        return results


# ============================================================
# 3. LangChain 工具定义
# ============================================================

# 全局变量：Agent 初始化后设置
retriever: Optional[CaseRetriever] = None

@tool
def search_similar_cases(query: str) -> str:
    """
    根据用户描述的案情，检索最相似的历史案例。
    适用于：用户描述了某个具体案件情况，想找类似案例参考。
    输入：一段案情描述文本（中文），可以包含案由、事实、诉求等。
    输出：Top 3 最相似案例的详细信息，包括案由、关键事实、裁判要点、判决结果和相似度评分。
    """
    global retriever
    if retriever is None:
        return "⚠️ 案例库尚未初始化，请稍后重试。"

    results = retriever.search(query, top_k=TOP_K)
    if not results:
        return "🔍 未检索到高度相似的案例。建议：\n1. 简化案情描述，突出核心争议\n2. 尝试用更通用的法律术语描述\n3. 说明案件是民事还是刑事"

    lines = [f"🔍 检索到 {len(results)} 个相似案例：\n"]
    for i, c in enumerate(results, 1):
        lines.append(f"{'─'*50}")
        lines.append(f"[{i}] 相似度: {c['score']:.2%}")
        lines.append(f"    案由: {c['cause']}")
        lines.append(f"    关键事实: {c['description'][:MAX_CASE_DISPLAY_CHARS]}...")
        lines.append(f"    裁判要点: {c['ruling_points'][:MAX_CASE_DISPLAY_CHARS]}...")
        lines.append(f"    判决结果: {c['judgment']}")
        lines.append(f"    关键词: {', '.join(c['keywords'][:5])}")
    lines.append(f"{'─'*50}")
    return '\n'.join(lines)


@tool
def query_legal_knowledge(question: str) -> str:
    """
    回答法律知识问题，例如法条内容、法律概念解释、诉讼程序等。
    适用于：用户询问某个法律术语的含义、某条法律的规定、诉讼流程等。
    输入：一个法律知识问题。
    输出：相关知识解答（由LLM根据法律知识回答）。
    注意：此工具不检索具体案例，仅回答法律知识问题。
    """
    # 此工具让LLM用自己的知识回答，返回空让LLM自行发挥
    return f"[法律知识查询] {question}\n\n请根据你的法律知识，用专业但易懂的语言回答上述问题。如涉及具体法条，请注明出处。"


# ============================================================
# 4. Agent 构建
# ============================================================

SYSTEM_PROMPT = """你是一个专业的法律案例智能推荐助手，名为"法眼AI"。

## 你的能力
你有两个工具：
1. `search_similar_cases`: 检索与用户描述的案情最相似的历史案例
2. `query_legal_knowledge`: 回答法律知识问题（法条、概念、程序等）

## 工作流程
1. **理解用户意图**：用户是在描述案情找类似案例，还是在询问法律知识？
2. **案情检索模式**：如果用户在描述案情，调用 search_similar_cases，然后：
   - 用通俗语言总结每个相似案例的关键点
   - 指出这些案例对用户有什么参考价值
   - 如果有多个案例共同指向某个结论，明确指出
3. **法律咨询模式**：如果用户在问法律知识，调用 query_legal_knowledge，然后给出专业回答
4. **混合模式**：如果用户既描述了案情又问了法律问题，先检索案例，再补充法律知识

## 严格规则
- ⚠️ **不要编造案例**：如果检索不到，诚实告知，不要编造不存在的结果
- ⚠️ **不要给出胜率/必然性判断**：不说"你会赢""一定胜诉"等
- ⚠️ **每次推荐案例都要引用具体信息**：案由、裁判要点、判决结果
- ⚠️ **用专业但易懂的语言**：避免过于学术化的术语堆砌
- ✅ 如果用户追问细节，结合对话历史给出更精准的回答
- ✅ 可以给出建议，但必须在结尾附上"建议咨询专业律师"的提示
"""


def create_agent():
    """构建 LangChain Agent"""
    global retriever

    # 加载数据
    cases = load_cases(CSV_PATH)
    retriever = CaseRetriever(cases)

    # LLM
    if not API_KEY or API_KEY == "your-api-key":
        print("\n⚠️  MINIMAX_API_KEY 未设置！")
        print("   请将 API Key 写入 .env 文件: MINIMAX_API_KEY=sk-xxx")
        print("   然后重新运行。\n")
        print("   进入 Debug 模式（仅测试检索功能，不调用 LLM）\n")
        return None, retriever

    llm = ChatOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
        temperature=0.3,
        max_tokens=2048,
    )

    tools = [search_similar_cases, query_legal_knowledge]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent, tools=tools,
        verbose=False,  # 改为 True 可看到思考过程
        max_iterations=5,
        handle_parsing_errors=True,
    )
    return executor, retriever


# ============================================================
# 5. 交互式对话
# ============================================================

def interactive_chat():
    """命令行交互式对话"""
    executor, ret = create_agent()

    if executor is None:
        # Debug 模式：仅测试检索
        print("=" * 60)
        print("🔧 Debug 模式：输入案情测试检索（不调用 LLM）")
        print("   输入 'quit' 退出")
        print("=" * 60)
        while True:
            q = input("\n👤 案情描述: ").strip()
            if q.lower() == 'quit': break
            if not q: continue
            results = ret.search(q)
            print(search_similar_cases.invoke(q))
        return

    print("\n" + "=" * 60)
    print("⚖️  法眼AI · 法律案例智能推荐 Agent")
    print("=" * 60)
    print("输入案情描述或法律问题，我会为您检索相似案例并提供建议。")
    print("输入 'quit' 退出 | 'reset' 重置对话 | 'verbose' 切换思考过程")
    print("=" * 60)

    chat_history = []
    verbose = False

    while True:
        try:
            user_input = input("\n👤 您: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() == 'quit':
            print("再见！")
            break
        if user_input.lower() == 'reset':
            chat_history = []
            print("🔄 对话已重置")
            continue
        if user_input.lower() == 'verbose':
            verbose = not verbose
            executor.verbose = verbose
            print(f"🔧 Verbose={'开' if verbose else '关'}")
            continue

        print("\n🤖 法眼AI 思考中...")
        try:
            result = executor.invoke({
                "input": user_input,
                "chat_history": chat_history,
            })
            response = result['output']
            print(f"\n🤖 法眼AI: {response}")
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response))
        except Exception as e:
            print(f"\n❌ 出错了: {e}")
            # 回退：不记入历史，让用户重试
            print("   请重试或输入 'reset' 重置对话")


if __name__ == "__main__":
    interactive_chat()
