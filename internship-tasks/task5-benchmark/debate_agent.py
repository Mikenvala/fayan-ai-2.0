#!/usr/bin/env python3
"""
法眼AI · 多Agent法律辩论系统
=============================
模拟法庭辩论：原告Agent vs 被告Agent，裁判Agent总结

运行: python debate_agent.py
"""

import os, sys, json, time
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))

from langchain_openai import ChatOpenAI

# ============================================================
# 配置
# ============================================================
DEBATE_ROUNDS = 3
API_KEY = os.environ.get("MINIMAX_API_KEY", "")
BASE_URL = "https://api.minimax.chat/v1"
MODEL = "MiniMax-M2.7"

def get_llm(temp=0.5):
    return ChatOpenAI(api_key=API_KEY, base_url=BASE_URL, model=MODEL, temperature=temp, timeout=30, max_retries=0)

# ============================================================
# Agent 提示词
# ============================================================
PLAINTIFF_PROMPT = """你是一名资深原告律师。你的目标是为原告争取最大利益。

案件事实：{case_facts}
争议焦点：{dispute_focus}

对方观点（如有）：{opponent_argument}

请以原告律师身份发表辩论意见，要求：
1. 引用相关法条支持原告主张
2. 指出对方观点的漏洞
3. 提出具体的诉讼请求
4. 控制在200字以内，专业且有力"""

DEFENDANT_PROMPT = """你是一名资深被告律师。你的目标是为被告辩护，减轻责任。

案件事实：{case_facts}
争议焦点：{dispute_focus}

对方观点（如有）：{opponent_argument}

请以被告律师身份发表辩论意见，要求：
1. 引用法条和案例支持被告立场
2. 反驳对方观点
3. 提出减轻责任的理由
4. 控制在200字以内，专业且有力"""

JUDGE_PROMPT = """你是一名资深法官。请根据以下辩论记录做出裁判摘要。

案件事实：{case_facts}

辩论记录：
{debate_transcript}

请给出：
1. 案件定性（适用什么法律）
2. 双方的合理主张
3. 裁判倾向（更支持哪方，为什么）
4. 建议的和解/判决方案
"""

# ============================================================
# 辩论案例
# ============================================================
DEBATE_CASES = [
    {
        "title": "P2P平台爆雷案",
        "facts": "某P2P平台虚构34个借款人信息，发布虚假标的，以20%年化收益为诱饵，向1586人吸收资金10.3亿元。所募资金未进入公司账户，由平台实控人周某个人掌控，用于购买房产、豪车、首饰等个人消费。案发后3.56亿元无法归还。",
        "focus": "周某的行为构成非法吸收公众存款罪还是集资诈骗罪？"
    },
    {
        "title": "AI生成内容侵权案",
        "facts": "某AI公司使用大量网络文章训练模型，生成的AI内容与某知名博主的多篇文章高度相似。博主起诉AI公司侵犯著作权，要求赔偿100万元。AI公司辩称训练数据的使用属于'合理使用'，AI生成内容是'转换性使用'。",
        "focus": "AI公司使用网络文章训练模型是否构成著作权侵权？AI生成内容与原文相似是否侵权？"
    },
    {
        "title": "外卖骑手工伤案",
        "facts": "外卖骑手张某在配送途中闯红灯被撞重伤，要求平台赔偿医疗费及伤残赔偿金80万元。平台认为张某是'个体工商户'，双方签订的是合作协议而非劳动合同，不构成劳动关系。张某每天工作12小时，接受平台派单管理，收入为唯一生活来源。",
        "focus": "张某与外卖平台是否构成劳动关系？平台是否应承担工伤赔偿责任？"
    }
]


# ============================================================
# 辩论引擎
# ============================================================
class DebateOrchestrator:
    def __init__(self):
        self.plaintiff_llm = get_llm(0.6)
        self.defendant_llm = get_llm(0.6)
        self.judge_llm = get_llm(0.3)
        self.transcript = []

    def run(self, case: dict) -> dict:
        facts = case["facts"]
        focus = case["focus"]
        plaintiff_arg = ""
        defendant_arg = ""

        print(f"\n{'='*60}")
        print(f"  ⚖️  法眼AI · 多Agent法律辩论")
        print(f"  案件: {case['title']}")
        print(f"{'='*60}")

        # 辩论回合
        for round_num in range(1, DEBATE_ROUNDS + 1):
            print(f"\n--- 第 {round_num} 轮 ---")

            # 原告发言
            p_prompt = PLAINTIFF_PROMPT.format(
                case_facts=facts, dispute_focus=focus,
                opponent_argument=defendant_arg if defendant_arg else "（首轮发言）"
            )
            try:
                p_resp = self.plaintiff_llm.invoke(p_prompt)
                plaintiff_arg = p_resp.content
                print(f"👤 原告律师: {plaintiff_arg[:120]}...")
            except Exception as e:
                plaintiff_arg = f"[发言失败: {e}]"
                print(f"❌ 原告发言失败")
            self.transcript.append({"round": round_num, "role": "plaintiff", "content": plaintiff_arg})
            time.sleep(0.3)

            # 被告发言
            d_prompt = DEFENDANT_PROMPT.format(
                case_facts=facts, dispute_focus=focus,
                opponent_argument=plaintiff_arg
            )
            try:
                d_resp = self.defendant_llm.invoke(d_prompt)
                defendant_arg = d_resp.content
                print(f"👤 被告律师: {defendant_arg[:120]}...")
            except Exception as e:
                defendant_arg = f"[发言失败: {e}]"
                print(f"❌ 被告发言失败")
            self.transcript.append({"round": round_num, "role": "defendant", "content": defendant_arg})
            time.sleep(0.3)

        # 裁判总结
        print(f"\n--- 裁判评议 ---")
        transcript_text = "\n\n".join([
            f"第{t['round']}轮 - {'原告律师' if t['role'] == 'plaintiff' else '被告律师'}:\n{t['content']}"
            for t in self.transcript
        ])
        j_prompt = JUDGE_PROMPT.format(case_facts=facts, debate_transcript=transcript_text)
        try:
            j_resp = self.judge_llm.invoke(j_prompt)
            verdict = j_resp.content
            print(f"⚖️ 裁判意见:\n{verdict}")
        except Exception as e:
            verdict = f"[裁判失败: {e}]"

        return {
            "case": case["title"],
            "facts": facts,
            "focus": focus,
            "transcript": self.transcript,
            "verdict": verdict
        }


if __name__ == "__main__":
    orchestrator = DebateOrchestrator()

    for i, case in enumerate(DEBATE_CASES, 1):
        print(f"\n\n{'#'*60}")
        print(f"# Case {i}: {case['title']}")
        print(f"{'#'*60}")

        result = orchestrator.run(case)

        # 保存
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                f"debate_{i:02d}_{case['title'].replace(' ', '_')}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 辩论结果已保存: {out_path}")

    print(f"\n{'='*60}")
    print(f"  全部 {len(DEBATE_CASES)} 场辩论完成！")
    print(f"{'='*60}")
