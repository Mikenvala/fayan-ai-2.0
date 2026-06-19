#!/usr/bin/env python3
"""单独评测 Mimo 模型 + 合并到已有结果"""
import json, os, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))
from langchain_openai import ChatOpenAI

# 复用30题
QUESTIONS = [
    {"id":"L01","category":"法条适用","question":"根据《刑法》第253条之一，侵犯公民个人信息罪的最高刑期是多少？","ground_truth":"七年有期徒刑","keywords":["七年","253条","有期徒刑"]},
    {"id":"L02","category":"法条适用","question":"民间借贷的年利率超过多少不受法律保护？","ground_truth":"合同成立时一年期贷款市场报价利率（LPR）的4倍","keywords":["LPR","4倍","一年期"]},
    {"id":"L03","category":"法条适用","question":"根据《刑法》，诈骗公私财物'数额特别巨大'的标准是多少？","ground_truth":"50万元以上","keywords":["50万","五十万","数额特别巨大"]},
    {"id":"L04","category":"法条适用","question":"劳动者在什么情况下可以单方解除劳动合同？","ground_truth":"未及时足额支付劳动报酬、未缴纳社保、规章制度违法等","keywords":["劳动报酬","社保","解除"]},
    {"id":"L05","category":"法条适用","question":"交通肇事逃逸的法定刑期是多少？","ground_truth":"三年以上七年以下有期徒刑","keywords":["三年","七年","有期徒刑"]},
    {"id":"L06","category":"法条适用","question":"民事诉讼的一般诉讼时效是多久？","ground_truth":"三年","keywords":["三年","3年"]},
    {"id":"C01","category":"罪名判断","question":"甲趁乙不备，夺走其手中的手机逃跑。甲构成什么罪？","ground_truth":"抢夺罪","keywords":["抢夺","抢夺罪"]},
    {"id":"C02","category":"罪名判断","question":"公司会计利用职务便利，将公司账户50万元转入自己账户。构成什么罪？","ground_truth":"职务侵占罪","keywords":["职务侵占","职务侵占罪"]},
    {"id":"C03","category":"罪名判断","question":"张三虚构投资项目，以高息为诱饵向100人募集资金500万元后挥霍。构成什么罪？","ground_truth":"集资诈骗罪","keywords":["集资诈骗","集资诈骗罪"]},
    {"id":"C04","category":"罪名判断","question":"李某将他人停放在路边的汽车开走并变卖。构成什么罪？","ground_truth":"盗窃罪","keywords":["盗窃","盗窃罪"]},
    {"id":"C05","category":"罪名判断","question":"王某与他人发生口角后，用拳头将对方打成轻伤。构成什么罪？","ground_truth":"故意伤害罪","keywords":["故意伤害","故意伤害罪"]},
    {"id":"C06","category":"罪名判断","question":"国家机关工作人员赵某利用职务便利，收受他人财物10万元为他人谋利。构成什么罪？","ground_truth":"受贿罪","keywords":["受贿","受贿罪"]},
    {"id":"S01","category":"量刑推理","question":"盗窃财物价值3000元（刚达到数额较大标准），初犯且退赃。可能的刑罚？","ground_truth":"三年以下有期徒刑、拘役或管制，可适用缓刑","keywords":["三年以下","缓刑","拘役"]},
    {"id":"S02","category":"量刑推理","question":"集资诈骗金额5亿元，造成损失2亿元。首犯可能判处什么刑罚？","ground_truth":"十年以上有期徒刑或无期徒刑","keywords":["十年以上","无期徒刑","无期"]},
    {"id":"S03","category":"量刑推理","question":"故意伤害致人重伤的法定刑期范围？","ground_truth":"三年以上十年以下有期徒刑","keywords":["三年","十年","有期徒刑"]},
    {"id":"S04","category":"量刑推理","question":"受贿300万元以上，有自首情节。可能的刑罚范围？","ground_truth":"十年以上有期徒刑或无期徒刑，自首可从轻","keywords":["十年","无期","自首"]},
    {"id":"S05","category":"量刑推理","question":"未成年人（16岁）初次盗窃，金额2000元。可能的处理方式？","ground_truth":"应当从轻或减轻处罚，可能不起诉或适用缓刑","keywords":["从轻","减轻","缓刑","不起诉"]},
    {"id":"S06","category":"量刑推理","question":"交通肇事致一人死亡且负全部责任，无逃逸。法定刑期？","ground_truth":"三年以下有期徒刑或者拘役","keywords":["三年以下","拘役","有期徒刑"]},
    {"id":"A01","category":"案例分析","question":"案情：甲以20%年化收益为饵，虚构借款人信息，通过P2P平台向1586人募集10.3亿元，用于购买房产和豪车。分析甲可能构成的罪名。","ground_truth":"集资诈骗罪","keywords":["集资诈骗","集资诈骗罪","非法占有"]},
    {"id":"A02","category":"案例分析","question":"案情：乙与丙有仇，某日持刀刺中丙胸部致其死亡。乙构成什么罪？可能判什么刑罚？","ground_truth":"故意杀人罪，死刑、无期徒刑或十年以上有期徒刑","keywords":["故意杀人","死刑","无期","十年"]},
    {"id":"A03","category":"案例分析","question":"案情：丁趁深夜撬锁进入商店，盗走价值2万元商品。丁构成什么罪？","ground_truth":"盗窃罪","keywords":["盗窃","盗窃罪"]},
    {"id":"A04","category":"案例分析","question":"案情：戊是公司高管，将公司客户名单和交易数据出售给竞争对手获利500万元。戊构成什么罪？","ground_truth":"侵犯商业秘密罪","keywords":["侵犯商业秘密","商业秘密罪"]},
    {"id":"A05","category":"案例分析","question":"案情：己未取得医生执业资格，擅自为他人进行节育复通手术，造成就诊人重伤。己构成什么罪？","ground_truth":"非法行医罪","keywords":["非法行医","非法行医罪"]},
    {"id":"A06","category":"案例分析","question":"案情：庚受国家机关委托从事公务，利用职务便利侵吞公款20万元。庚构成什么罪？","ground_truth":"贪污罪","keywords":["贪污","贪污罪"]},
    {"id":"P01","category":"程序问题","question":"民事诉讼中，被告所在地和原告所在地不一致时，通常由哪个法院管辖？","ground_truth":"被告住所地人民法院","keywords":["被告住所地","被告所在地"]},
    {"id":"P02","category":"程序问题","question":"不服一审判决，上诉期限是多少天？","ground_truth":"判决15日，裁定10日","keywords":["15日","15天","判决"]},
    {"id":"P03","category":"程序问题","question":"刑事拘留的最长期限是多少？","ground_truth":"37天","keywords":["37天","37日"]},
    {"id":"P04","category":"程序问题","question":"哪些案件不能公开审理？","ground_truth":"涉及国家秘密、个人隐私、未成年人犯罪的案件","keywords":["国家秘密","个人隐私","未成年人"]},
    {"id":"P05","category":"程序问题","question":"劳动仲裁是诉讼的前置程序吗？","ground_truth":"是，劳动争议一般需先经过劳动仲裁才能起诉","keywords":["前置","先仲裁","劳动仲裁"]},
    {"id":"P06","category":"程序问题","question":"刑事诉讼中被告人没有委托辩护人的，法院应当怎么做？","ground_truth":"应当通知法律援助机构指派律师提供辩护","keywords":["法律援助","指派","辩护"]},
]

def _score(answer, keywords):
    s = 0
    for kw in keywords:
        if kw.lower() in answer.lower():
            s += 1
    return min(5, s)

llm = ChatOpenAI(
    api_key=os.environ["MIMO_API_KEY"],
    base_url="https://api.xiaomimimo.com/v1",
    model="mimo-v2.5-pro",
    temperature=0.1, timeout=60, max_retries=1,
)

results = []
print(f"{'='*50}")
print(f"  🔬 Mimo (mimo-v2.5-pro)")
print(f"{'='*50}")

for i, q in enumerate(QUESTIONS, 1):
    prompt = f"请用简洁的中文回答以下法律问题：\n{q['question']}"
    try:
        resp = llm.invoke(prompt)
        score = _score(resp.content, q["keywords"])
        results.append({**q, "answer": resp.content[:500], "score": score})
        print(f"  [{i:2d}/30] {q['id']} {q['category']}: {score}/5")
    except Exception as e:
        print(f"  [{i:2d}/30] {q['id']} ❌ {str(e)[:60]}")
        results.append({**q, "answer": f"ERROR: {e}", "score": 0})
    time.sleep(0.1)

avg = sum(r["score"] for r in results) / len(results)
cats = {}
for r in results:
    cats.setdefault(r["category"], []).append(r["score"])

print(f"\n  >>> 平均分: {avg:.2f}/5 ({avg/5*100:.0f}%)")
for c in ["法条适用","罪名判断","量刑推理","案例分析","程序问题"]:
    v = cats.get(c, [0])
    print(f"  >>> {c}: {sum(v)/len(v):.1f}/5 ({sum(v)/len(v)/5*100:.0f}%)")

# Merge into existing results
with open("multi_benchmark_results.json") as f:
    all_res = json.load(f)

all_res["Mimo"] = {
    "model": "Mimo",
    "model_id": "mimo-v2.5-pro",
    "avg_score": avg,
    "accuracy": avg/5*100,
    "categories": {c: {"avg": sum(v)/len(v), "accuracy": sum(v)/len(v)/5*100} for c,v in cats.items()},
    "results": results,
}

with open("multi_benchmark_results.json", "w") as f:
    json.dump(all_res, f, ensure_ascii=False, indent=2)

print(f"\n  ✅ 已合并：共 {len(all_res)} 个模型")
for name in all_res:
    d = all_res[name]
    print(f"     {name}: {d['accuracy']:.0f}%")
