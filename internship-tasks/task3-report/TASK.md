# Task 3 · 数据分析报告撰写

## 任务目标

基于 Task 1 的分析结果，用 LaTeX 撰写一份规范的学术数据分析报告，格式参照《情报学报》。

### 报告结构

| 章节 | 内容要求 | 建议字数 |
|------|---------|---------|
| 标题 | 《基于大规模裁判文书的案件特征分析与规律挖掘》 | - |
| 摘要 | 简述数据来源、分析方法、核心发现（200字） | 200 |
| 1. 引言 | 研究背景、意义、数据概况 | 400 |
| 2. 数据与方法 | 数据集描述（10241条）、分析工具、5个分析维度 | 300 |
| 3. 分析结果 | 对应 Task 1 的 5 个分析，每项一个子节，配图+文字 | 1500 |
| 4. 讨论 | 发现的规律及其意义 | 300 |
| 5. 结论 | 总结 + 不足 + 展望 | 200 |
| 参考文献 | 至少引用 3 篇相关文献 | - |

### 格式要求

```
- 纸张: A4
- 字号: 正文五号(10.5pt)，标题三号/四号
- 行距: 1.5 倍
- 图表: 每个图/表要有编号和标题，图下标，表上标
- 引用: GB/T 7714 格式
- 页眉: 研究报告
- 页脚: 页码居中
```

### 交付物

```
task3-report/
├── report.tex             # LaTeX 主文件
├── figures/               # 图表（从 Task1 复制或引用）
│   ├── fig1_cooccurrence_heatmap.png
│   ├── fig2_length_boxplot.png
│   ├── fig3_judgment_pie.png
│   ├── fig4_cause_keywords_bar.png
│   └── fig5_year_trend_area.png
├── refs.bib               # 参考文献
└── report.pdf             # 编译后的 PDF
```

### 评分标准

| 项目 | 占比 |
|------|------|
| 结构完整（6章不缺） | 20% |
| 图表嵌入正确、编号规范 | 25% |
| 分析与图表匹配、有文字解读 | 25% |
| LaTeX 排版美观（字体/行距/页眉页脚） | 15% |
| 参考文献格式规范 | 15% |

### LaTeX 脚手架

```latex
\documentclass[12pt,a4paper]{ctexart}

% === 页面设置 ===
\usepackage[top=2.5cm, bottom=2.5cm, left=3cm, right=3cm]{geometry}
\usepackage{graphicx}
\usepackage{float}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{fancyhdr}
\usepackage[sort&compress]{gbt7714}

% 页眉页脚
\pagestyle{fancy}
\fancyhf{}
\fancyhead[C]{裁判文书案件特征分析报告}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}

% 图表路径
\graphicspath{{./figures/}}

\title{\heiti\fontsize{16pt}{20pt}\selectfont 基于大规模裁判文书的案件特征分析与规律挖掘}
\author{王志达}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
本文基于 10,241 条裁判文书数据，综合利用关键词共现分析、
文本篇幅统计、判决结果识别、案由特征关联及年度趋势分析等方法，
系统挖掘了裁判文书中的案件特征与潜在规律。
（—— 请根据 Task 1 的实际分析结果补充完整 ——）
\end{abstract}

\section{引言}
裁判文书是司法实践的重要记录，蕴含着丰富的案件特征信息\cite{ref1}。
（—— 请补充研究背景和意义 ——）

\section{数据与方法}
\subsection{数据来源}
本研究使用的数据集包含 10,241 条裁判文书，涵盖民事、刑事和行政案件类型，
时间跨度从 2014 年至 2025 年。

每条案例包含：案件描述、原告诉求、判别标准、判决结果以及 10 个关键词字段。

\subsection{分析方法}
（—— 请描述 Task 1 中使用的 5 种分析方法 ——）

\section{分析结果}
（—— 请逐项描述分析结果，每项配图 ——）

\subsection{关键词共现网络分析}
（—— 插入 fig1 并解读 ——）

\subsection{判决书篇幅分析}
（—— 插入 fig2 并解读 ——）

\subsection{判决结果情感倾向分析}
（—— 插入 fig3 并解读 ——）

\subsection{案由-关键词关联分析}
（—— 插入 fig4 并解读 ——）

\subsection{年度趋势分析}
（—— 插入 fig5 并解读 ——）

\section{讨论}
（—— 请总结发现的规律及其意义 ——）

\section{结论}
（—— 总结全文，指出不足与未来方向 ——）

% === 参考文献 ===
\bibliographystyle{gbt7714-numerical}
\bibliography{refs}

\end{document}
```

### 参考文献脚手架 (refs.bib)

```bibtex
@article{ref1,
  author  = {王禄生},
  title   = {司法大数据与人工智能技术应用的风险及伦理规制},
  journal = {法商研究},
  year    = {2019},
  volume  = {36},
  number  = {2},
  pages   = {101--112},
}

@article{ref2,
  author  = {左卫民},
  title   = {关于法律人工智能在中国运用前景的若干思考},
  journal = {清华法学},
  year    = {2018},
  volume  = {12},
  number  = {2},
  pages   = {108--124},
}

@article{ref3,
  author  = {华宇元典法律人工智能研究院},
  title   = {让法律人读懂人工智能},
  journal = {法律出版社},
  year    = {2019},
}
```

### 编译方法

```bash
# 如果你的机器安装了 texlive：
cd task3-report
xelatex report.tex
bibtex report
xelatex report.tex
xelatex report.tex   # 两次编译确保交叉引用正确
```

### 提示

- 先完成 Task 1 再做 Task 3，图表直接从 Task1 的 output 目录复制过来
- 每个分析结果的文字解读要回答三个问题：看到了什么？可能的原因？有什么意义？
- 如果本地没有 LaTeX 环境，先写 .tex 文件，需要编译时告诉我
- 参考文献如果找不到原文，可以用标题合理推测作者和期刊，不要求 100% 精确

开始动笔吧！
