# ⚖️ 法眼AI —— 智能法律案例分析系统

基于大语言模型（LLM）与检索增强生成（RAG）技术的智能法律案例分析平台，面向公众提供民事/刑事案件的智能分析与类案检索服务。

## 🚀 功能特性

- **案件自动分类**：基于规则引擎 + 关键词匹配，支持民事、刑事、刑民交叉三类自动判别
- **RAG 智能检索**：TF-IDF + BM25 混合检索，双库架构（民事 + 刑事），精准召回相似案例
- **LLM 法律分析**：集成 MiniMax LLM，自动生成案件分析结论
- **合规检测**：内置禁止性用语检测引擎（RuleEngine），确保输出合法合规
- **复杂度评估**：自动评估案件复杂度（低/中/高/极高），必要时建议律师介入
- **涉案信息提取**：自动识别涉案金额、当事人数量等关键信息
- **全栈 Web 应用**：响应式 Web 界面，支持桌面端一键运行

## 🛠 技术栈

| 层级 | 技术 |
|------|------|
| 后端 | Python Flask / Node.js |
| AI/ML | LangChain、MiniMax API、scikit-learn、jieba |
| 检索 | TF-IDF、BM25、FAISS 向量存储 |
| 前端 | HTML5、CSS3、JavaScript（原生） |
| 数据 | CSV/JSON 处理、Pandas、NumPy |
| 部署 | PyInstaller 桌面打包、Docker |

## 📁 项目结构

```
法眼AI/
├── app(1).py              # Flask Web 服务入口
├── fayan_api.py           # 核心 API 层（RAG + LLM + 规则引擎）
├── fayan_rag_chain.py     # 法律知识库构建与 RAG 检索
├── fayan_legal_rag.py     # LangChain 版 RAG 实现
├── rag_qa.py              # 轻量级 RAG 问答系统
├── fayan_main(1).py       # 桌面端打包入口
├── server_full.js         # Node.js 全功能服务器
├── server.js              # Node.js 服务器（精简版）
├── server_v2.js           # Node.js 服务器 v2
├── 法眼AI.html            # 独立前端页面
├── templates/
│   └── index.html         # Flask 模板
├── static/
│   ├── css/style.css      # 样式表
│   └── js/app.js          # 前端逻辑
├── all_cases_perfect.csv  # 案例数据集（10,000+ 条）
├── requirements.txt       # Python 依赖
├── Dockerfile             # Docker 构建文件
└── README.md
```

## 🔧 快速开始

### 环境要求

- Python 3.9+
- Node.js 18+
- MiniMax API Key

### 安装与运行

```bash
# 1. 克隆仓库
git clone <repo-url>
cd 法眼AI

# 2. 安装 Python 依赖
pip install -r requirements.txt

# 3. 配置 API Key
echo "MINIMAX_API_KEY=your-api-key" > .env

# 4. 启动 Flask 服务
python app\(1\).py
# 访问 http://localhost:5000

# 或使用 Node.js 服务器
node server_full.js
# 访问 http://localhost:5099
```

### Docker 部署

```bash
# 构建镜像
docker build -t fayan-ai .

# 运行容器
docker run -p 5099:5099 -e MINIMAX_API_KEY=your-key fayan-ai
```

## 📊 数据说明

系统加载了 10,000+ 条真实裁判文书数据，按民事/刑事分类，每条案例包含：
- 案件描述、原告诉求、判别标准
- 判决结果、关键词（10个维度）
- 结构化元数据（案由、裁判要点、相关法条）

## 📝 License

个人学习与研究项目

---

*Made with ❤️ by 王志达*
