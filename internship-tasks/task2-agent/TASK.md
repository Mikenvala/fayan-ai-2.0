# Task 2 · 法律案件智能推荐 Agent

## 任务目标

基于法眼AI 的案例库，用 LangChain 构建一个智能案件推荐 Agent，实现多轮对话式的案例检索体验。

### 核心功能要求

| # | 功能 | 描述 |
|---|------|------|
| 1 | 意图识别 | Agent 能理解用户是在"描述案情找相似案例"，还是在"询问法律知识" |
| 2 | 案例检索 | 根据用户描述的案情，用 BM25/TF-IDF 检索 Top 3 最相似案例 |
| 3 | 案例推荐 | 格式化输出推荐结果：案号、案由、关键事实、判决要点、相似度评分 |
| 4 | 多轮对话 | 支持追问："有没有刑法的类似案例？""金额更大的呢？" |
| 5 | 防幻觉 | 检索不到相关案例时，诚实告知而非编造结果 |

### Agent 架构建议

```
用户输入 → [Router] → 意图分类
              ├── "检索案例" → Retriever Tool → 返回案例列表 → LLM 总结
              └── "法律咨询" → Knowledge Tool → 返回法条知识 → LLM 回答
```

### 技术选型

- **框架**: `langchain` + `langchain-openai`
- **LLM**: MiniMax API（已有）或其他兼容 OpenAI 接口的模型
- **检索**: BM25（rank-bm25）+ TF-IDF（scikit-learn）
- **向量存储**（加分项）: FAISS
- **工具**: `@tool` 装饰器定义 Retriever / Knowledge 两个工具

### 交付物

```
task2-agent/
├── case_agent.py          # Agent 主程序
├── tools/
│   ├── retriever.py       # 案例检索工具
│   └── knowledge.py       # 法律知识工具
└── demo_conversation.txt  # 示例对话记录（3-5 轮）
```

### 示例交互

```
用户: 我借给朋友 50 万，没写借条，现在他不还了怎么办？

Agent: 🔍 检索到 3 个相似案例：

  [1] 民间借贷纠纷案 (相似度: 0.87)
      案由: 民间借贷纠纷
      关键事实: 原告主张口头借贷 80 万元，无书面借条，
               但有银行转账记录和微信聊天记录佐证
      裁判要点: 自然人之间借款合同不以书面形式为要件，
               转账凭证+聊天记录可形成证据链证明借贷关系存在
      判决: 部分支持，判令被告返还本金 80 万及利息

  [2] 借贷纠纷案 (相似度: 0.79)
      ...

  综合建议：虽然没有书面借条，但如果你有银行转账记录、
  微信聊天记录等证据，仍有可能被法院认定为借贷关系成立。
  建议收集转账凭证和沟通记录后咨询专业律师。

用户: 那要是对方说这钱是我送给他的呢？

Agent: 🔍 针对"赠与抗辩"检索到 2 个相关案例：
      ...
```

### 评分标准

| 项目 | 占比 |
|------|------|
| Agent 能正确检索到相关案例 | 30% |
| 输出格式规范（案号+要点+建议） | 20% |
| 多轮对话上下文记忆 | 20% |
| 检索不到时诚实告知 | 15% |
| 代码结构清晰 | 15% |

### 脚手架代码

```python
#!/usr/bin/env python3
"""法律案件智能推荐 Agent"""

import os, json, csv
import re
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import jieba
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ========== 配置 ==========
LLM = ChatOpenAI(
    api_key=os.environ.get("MINIMAX_API_KEY"),
    base_url="https://api.minimax.chat/v1",
    model="MiniMax-M2.7",
    temperature=0.3,
)

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "all_cases_perfect.csv")

# ========== 加载案例库 ==========
# TODO: 实现案例加载 + 索引构建
# 思路：
# 1. 读取 CSV，为每个案例拼接检索文本（案件描述 + 关键词）
# 2. 构建 BM25 索引
# 3. 构建 TF-IDF 矩阵（用于计算相似度分数）
# 每个案例保留：案号(文件名)、案由、关键事实(前200字)、判决要点(判别标准)、判决结果

# ========== 工具定义 ==========
@tool
def search_similar_cases(query: str) -> str:
    """
    根据用户描述的案情，检索最相似的历史案例。
    输入: 案情描述文本
    输出: Top 3 相似案例的详细信息（案由、关键事实、裁判要点、判决结果、相似度）
    """
    # TODO: 实现 BM25 + TF-IDF 检索
    pass

@tool
def query_legal_knowledge(question: str) -> str:
    """
    回答法律知识问题，例如法条内容、法律概念解释等。
    输入: 法律问题
    输出: 相关知识解答
    """
    # TODO: 可以用 LLM 直接回答，也可以对接法条库
    pass

# ========== Agent 系统提示词 ==========
SYSTEM_PROMPT = """你是一个法律案例智能推荐助手。你有以下工具可用：
- search_similar_cases: 检索相似案例
- query_legal_knowledge: 回答法律知识问题

工作流程：
1. 先理解用户意图：是在描述案情找案例，还是在问法律知识？
2. 如果是找案例，调用 search_similar_cases 检索，然后基于检索结果给出建议
3. 如果是法律知识，调用 query_legal_knowledge 回答
4. 如果用户追问，结合对话历史给出更精准的回答

重要规则：
- 每次案例推荐都要引用案号和裁判要点
- 如果检索不到，诚实告知，不要编造案例
- 不要给出"一定会赢/输"之类的不合规表述
- 用专业但易懂的语言回答
"""

# ========== Agent 构建 ==========
def create_agent():
    """构建 Agent"""
    tools = [search_similar_cases, query_legal_knowledge]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_tools_agent(LLM, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
    )
    return executor


# ========== 交互式对话 ==========
def interactive_chat():
    """命令行交互式对话"""
    executor = create_agent()
    print("=" * 60)
    print("⚖️  法律案例智能推荐 Agent")
    print("输入 'quit' 退出, 'reset' 重置对话")
    print("=" * 60)
    
    chat_history = []
    while True:
        user_input = input("\n👤 你: ").strip()
        if not user_input:
            continue
        if user_input.lower() == 'quit':
            print("再见！")
            break
        if user_input.lower() == 'reset':
            chat_history = []
            print("对话已重置")
            continue
        
        # TODO: 调用 agent，传入 chat_history
        # 提示: AgentExecutor.invoke({"input": user_input, "chat_history": chat_history})
        
        # 临时占位
        print("🤖 Agent: (请实现 search_similar_cases 和 query_legal_knowledge 工具)")


if __name__ == "__main__":
    interactive_chat()
```

### 提示

- 案例检索文本建议拼接：`案件描述[:300] + 判别标准[:200] + 关键词列表`
- 相似度可以用 TF-IDF 的 cosine_similarity，也可以用 BM25 的分数归一化
- 多轮对话上下文通过 `chat_history` 参数传递
- 先用爬架代码跑通流程，再优化检索效果
- 如果 MiniMax API 调用有问题，可以先用 `debug=True` 模式跳过 LLM 直接测试检索

开始动手吧！
