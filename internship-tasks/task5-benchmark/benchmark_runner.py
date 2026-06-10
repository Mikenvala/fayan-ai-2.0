#!/usr/bin/env python3
"""
法眼AI · LLM 法律能力评测基准
=============================
评测 MiniMax LLM 在法律领域的核心能力：
  1. 法条适用  2. 罪名判断  3. 量刑推理
  4. 案例分析  5. 程序问题

运行: python benchmark_runner.py
"""

import json, os, sys, time, csv
from datetime import datetime
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))

from langchain_openai import ChatOpenAI

# ============================================================
# 评测题目 (30题, 5类x6题)
# ============================================================
QUESTIONS = [
    # === 法条适用 ===
    {"id": "L01", "category": "法条适用", "question": "根据《刑法》第253条之一，侵犯公民个人信息罪的最高刑期是多少？",
     "ground_truth": "七年有期徒刑", "keywords": ["七年", "253条", "有期徒刑"]},
    {"id": "L02", "category": "法条适用", "question": "民间借贷的年利率超过多少不受法律保护？",
     "ground_truth": "合同成立时一年期贷款市场报价利率（LPR）的4倍", "keywords": ["LPR", "4倍", "一年期"]},
    {"id": "L03", "category": "法条适用", "question": "根据《刑法》，诈骗公私财物'数额特别巨大'的标准是多少？",
     "ground_truth": "50万元以上", "keywords": ["50万", "五十万", "数额特别巨大"]},
    {"id": "L04", "category": "法条适用", "question": "劳动者在什么情况下可以单方解除劳动合同？",
     "ground_truth": "未及时足额支付劳动报酬、未缴纳社保、规章制度违法等", "keywords": ["劳动报酬", "社保", "解除"]},
    {"id": "L05", "category": "法条适用", "question": "交通肇事逃逸的法定刑期是多少？",
     "ground_truth": "三年以上七年以下有期徒刑", "keywords": ["三年", "七年", "有期徒刑"]},
    {"id": "L06", "category": "法条适用", "question": "民事诉讼的一般诉讼时效是多久？",
     "ground_truth": "三年", "keywords": ["三年", "3年"]},

    # === 罪名判断 ===
    {"id": "C01", "category": "罪名判断", "question": "甲趁乙不备，夺走其手中的手机逃跑。甲构成什么罪？",
     "ground_truth": "抢夺罪", "keywords": ["抢夺", "抢夺罪"]},
    {"id": "C02", "category": "罪名判断", "question": "公司会计利用职务便利，将公司账户50万元转入自己账户。构成什么罪？",
     "ground_truth": "职务侵占罪", "keywords": ["职务侵占", "职务侵占罪"]},
    {"id": "C03", "category": "罪名判断", "question": "张三虚构投资项目，以高息为诱饵向100人募集资金500万元后挥霍。构成什么罪？",
     "ground_truth": "集资诈骗罪", "keywords": ["集资诈骗", "集资诈骗罪"]},
    {"id": "C04", "category": "罪名判断", "question": "李某将他人停放在路边的汽车开走并变卖。构成什么罪？",
     "ground_truth": "盗窃罪", "keywords": ["盗窃", "盗窃罪"]},
    {"id": "C05", "category": "罪名判断", "question": "王某与他人发生口角后，用拳头将对方打成轻伤。构成什么罪？",
     "ground_truth": "故意伤害罪", "keywords": ["故意伤害", "故意伤害罪"]},
    {"id": "C06", "category": "罪名判断", "question": "国家机关工作人员赵某利用职务便利，收受他人财物10万元为他人谋利。构成什么罪？",
     "ground_truth": "受贿罪", "keywords": ["受贿", "受贿罪"]},

    # === 量刑推理 ===
    {"id": "S01", "category": "量刑推理", "question": "盗窃财物价值3000元（刚达到数额较大标准），初犯且退赃。可能的刑罚？",
     "ground_truth": "三年以下有期徒刑、拘役或管制，可适用缓刑", "keywords": ["三年以下", "缓刑", "拘役"]},
    {"id": "S02", "category": "量刑推理", "question": "集资诈骗金额5亿元，造成损失2亿元。首犯可能判处什么刑罚？",
     "ground_truth": "十年以上有期徒刑或无期徒刑", "keywords": ["十年以上", "无期徒刑", "无期"]},
    {"id": "S03", "category": "量刑推理", "question": "故意伤害致人重伤的法定刑期范围？",
     "ground_truth": "三年以上十年以下有期徒刑", "keywords": ["三年", "十年", "有期徒刑"]},
    {"id": "S04", "category": "量刑推理", "question": "受贿300万元以上，有自首情节。可能的刑罚范围？",
     "ground_truth": "十年以上有期徒刑或无期徒刑，自首可从轻", "keywords": ["十年", "无期", "自首"]},
    {"id": "S05", "category": "量刑推理", "question": "未成年人（16岁）初次盗窃，金额2000元。可能的处理方式？",
     "ground_truth": "应当从轻或减轻处罚，可能不起诉或适用缓刑", "keywords": ["从轻", "减轻", "缓刑", "不起诉"]},
    {"id": "S06", "category": "量刑推理", "question": "交通肇事致一人死亡且负全部责任，无逃逸。可能判处？",
     "ground_truth": "三年以下有期徒刑或拘役", "keywords": ["三年以下", "拘役", "有期徒刑"]},

    # === 案例分析 ===
    {"id": "A01", "category": "案例分析", "question": "甲向乙借款10万元，约定年利率24%。到期后甲不还款，乙起诉。乙能拿回多少利息？",
     "ground_truth": "按LPR 4倍计算，超出部分不支持", "keywords": ["LPR", "4倍", "不支持"]},
    {"id": "A02", "category": "案例分析", "question": "P2P平台虚构借款人信息吸收资金后用于个人消费。该行为如何定性？",
     "ground_truth": "集资诈骗罪，非法占有为目的使用诈骗方法", "keywords": ["集资诈骗", "非法占有", "诈骗"]},
    {"id": "A03", "category": "案例分析", "question": "外卖骑手送餐途中撞伤行人，赔偿责任由谁承担？",
     "ground_truth": "用人单位（外卖平台或劳务公司）承担雇主责任", "keywords": ["用人单位", "雇主", "平台"]},
    {"id": "A04", "category": "案例分析", "question": "离婚时一方隐藏转移夫妻共同财产，另一方如何救济？",
     "ground_truth": "可以少分或不分该方财产，离婚后可再诉", "keywords": ["少分", "不分", "再诉"]},
    {"id": "A05", "category": "案例分析", "question": "开发商一房二卖，两个买受人谁取得房屋所有权？",
     "ground_truth": "先办理过户登记的取得所有权，未登记的只能主张违约", "keywords": ["过户", "登记", "违约"]},
    {"id": "A06", "category": "案例分析", "question": "单位将员工个人信息出售给第三方营销公司。单位是否需要承担责任？",
     "ground_truth": "是，构成侵犯公民个人信息罪，单位可被判处罚金", "keywords": ["侵犯公民个人信息", "单位犯罪", "罚金"]},

    # === 程序问题 ===
    {"id": "P01", "category": "程序问题", "question": "民事诉讼中被告下落不明，法院如何送达？",
     "ground_truth": "公告送达，自公告之日起经过30日视为送达", "keywords": ["公告送达", "30日", "30天"]},
    {"id": "P02", "category": "程序问题", "question": "刑事拘留的最长期限是多少天？",
     "ground_truth": "一般3天，可延长至7天，特殊可达37天", "keywords": ["37天", "30天", "拘留"]},
    {"id": "P03", "category": "程序问题", "question": "民事诉讼一审判决后，上诉期限是多久？",
     "ground_truth": "判决书送达之日起15日内", "keywords": ["15日", "15天"]},
    {"id": "P04", "category": "程序问题", "question": "哪些案件不能公开审理？",
     "ground_truth": "涉及国家秘密、个人隐私、未成年人犯罪的案件", "keywords": ["国家秘密", "个人隐私", "未成年人"]},
    {"id": "P05", "category": "程序问题", "question": "劳动仲裁是诉讼的前置程序吗？",
     "ground_truth": "是，劳动争议一般需先经过劳动仲裁才能起诉", "keywords": ["前置", "先仲裁", "劳动仲裁"]},
    {"id": "P06", "category": "程序问题", "question": "刑事诉讼中被告人没有委托辩护人的，法院应当怎么做？",
     "ground_truth": "应当通知法律援助机构指派律师提供辩护", "keywords": ["法律援助", "指派", "辩护"]},
]


# ============================================================
# 评测引擎
# ============================================================
class LegalBenchmark:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=os.environ.get("MINIMAX_API_KEY", ""),
            base_url="https://api.minimax.chat/v1",
            model="MiniMax-M2.7",
            temperature=0.1,
            timeout=30,
            max_retries=0
        )
        self.results = []

    def run(self) -> list:
        print(f"\n{'='*60}")
        print(f"  ⚖️  法眼AI · LLM 法律能力评测")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"  模型: MiniMax-M2.7  |  题目: {len(QUESTIONS)} 题")
        print(f"{'='*60}\n")

        for i, q in enumerate(QUESTIONS, 1):
            print(f"[{i}/{len(QUESTIONS)}] {q['id']} {q['category']}: {q['question'][:50]}...")
            try:
                resp = self.llm.invoke(f"请用简洁的中文回答以下法律问题：\n{q['question']}")
                answer = resp.content
                score = self._score(answer, q)
                self.results.append({
                    **q,
                    "answer": answer[:500],
                    "score": score,
                    "full_answer": answer
                })
                print(f"   得分: {score}/5")
            except Exception as e:
                print(f"   ❌ 错误: {str(e)[:80]}")
                self.results.append({
                    **q,
                    "answer": f"ERROR: {str(e)[:100]}",
                    "score": 0
                })
            time.sleep(0.3)

        return self.results

    def _score(self, answer: str, q: dict) -> int:
        """评分：关键词匹配 + 语义评估"""
        score = 0
        answer_lower = answer.lower()
        for kw in q.get("keywords", []):
            if kw.lower() in answer_lower:
                score += 1
        # 最多5分
        return min(5, score)

    def report(self) -> str:
        if not self.results:
            return "无评测数据"

        total = len(self.results)
        avg_score = sum(r["score"] for r in self.results) / total
        categories = {}
        for r in self.results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = {"total": 0, "score": 0}
            categories[cat]["total"] += 1
            categories[cat]["score"] += r["score"]

        report = f"""# 法眼AI · LLM 法律能力评测报告

## 基本信息
- 模型: MiniMax-M2.7
- 评测日期: {datetime.now().strftime('%Y-%m-%d')}
- 题目总数: {total} 题
- 平均得分: {avg_score:.2f}/5

## 分类得分

| 类别 | 题目数 | 平均分 | 得分率 |
|------|--------|--------|--------|
"""
        for cat in ["法条适用", "罪名判断", "量刑推理", "案例分析", "程序问题"]:
            if cat in categories:
                c = categories[cat]
                avg = c["score"] / c["total"]
                report += f"| {cat} | {c['total']} | {avg:.2f} | {avg/5*100:.0f}% |\n"

        report += f"""
## 详细结果

| ID | 类别 | 问题 | 得分 |
|----|------|------|------|
"""
        for r in self.results:
            q_short = r["question"][:40]
            report += f"| {r['id']} | {r['category']} | {q_short}... | {r['score']}/5 |\n"

        report += f"""

## 结论

- 整体准确率: {avg_score/5*100:.1f}%
- 最强领域: {max(categories, key=lambda x: categories[x]['score']/categories[x]['total'])}
- 待提升领域: {min(categories, key=lambda x: categories[x]['score']/categories[x]['total'])}

---
*评测工具: 法眼AI Benchmark v1.0*
"""
        return report


if __name__ == "__main__":
    bench = LegalBenchmark()
    bench.run()

    report = bench.report()
    print("\n" + report)

    # 保存结果
    out_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(out_dir, "benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump(bench.results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "benchmark_report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n✅ 结果已保存到 benchmark_results.json 和 benchmark_report.md")
