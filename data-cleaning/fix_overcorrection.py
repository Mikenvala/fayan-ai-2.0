#!/usr/bin/env python3
"""Fix overcorrections from previous passes."""
import json

with open('/Users/wangzhida/Downloads/chapters_fixed.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# The biggest issue is 之 was over-applied. Revert specific overcorrections.
fixes = {
    '本之': '本章',
    '第一之': '第一章',
    '第之': '第一章',
    '本之主': '本章主',
    '本之概': '本章概',
    '宪之': '宪章',
    '规之': '规章',
    '文之': '文章',
    '乐之': '乐章',
    '之一': '之一',  # keep correct

    # Also fix remaining issues
    '一。2年': '2002年',
    '2。2年': '2002年',
    '2。1年': '2001年',
    '2。 年': '2000年',
    '|国': '治国',
    '|家': '国家',
    '|': '治',

    # Misc
    '_设': '建设',  # _设项目 -> 建设项目
    '。12月': ',12月',

    # Chapter 9 specific - more case study cleanup
    '茉瘃痂n防H;-中*/9金阶。入詹民': '某研究机构',
    '塞等「品': '等',
    '赢法': '两家',
    '新楚实验室': '新建实验室',
    '协高': '协商',
    '共瘃': '某',
    'n防H': '的',
    '中*/9金阶': '的',
    '入詹民': '居民',
    '塞等': '等',
    '「品': '的',
    '赢法': '两家',
    'C': '',
    'I1': '',

    # 耿 -> wrong char in ch7
    '耿': '',

    # 必皴 -> 必须
    '必皴': '必须',
    '国崮': '国家',
    '圄家': '国家',

    # Chapter 12 cleanup
    '司焘': '同意',
    '《购进': '购进',
    '。.。62': '0.062',
    '元一龟': '元一包',
    '替怅柔': '替换',
    '。.1。元': '0.10元',
    '|抄': '找',
    '麻睿': '麻烦',
    '惫于': '由于',
    '忝幸': '众多',
    '消赘这': '消费者',
    '拒宪': '拒绝',
    '只圩': '只好',
    '恐气吞声': '忍气吞声',
    '《找补解巾': '找补纸巾',
    '即侠在': '即使在',
    '的足鲈': '的足够',
    '经补瞒客': '找补顾客',
    '怖沉': '情况',
    '载巾': '纸巾',
    '《钞': '钞票',
    '霾': '',
    '兴穷': '的',
    '因粼': '的',
    's。 压$h,$愆包劈详': '',
    '利 万余元': '获利万余元',
    '作为奖金分《': '作为奖金分配',
    '杓戚': '构成',
    '措皆': '欺诈',
    '屉否': '是否',
    '侵': '侵',
    '屎护': '保护',
    '消黛': '消费',
    '蚯A': '处理',
    '环坑法': '环境法',
    '栽罔': '我国',
    '蹄r': '加入',
    '圉际': '国际',
    '绕一': '统一',
    '佳体': '体系',
    '原副': '原则',
    '原刖': '原则',
    '原射': '原则',
    '1呐评剃': '影响评价制度',
    '度』': '度',
    '粪': '法',
    '贲源': '资源',
    '权尿': '权利',
    '档窠': '档案',
    '有幞': '有偿',
    '井责任': '和刑事责任',
    '民本4任': '民事责任',
    '行攻': '行政',
    '萼': '的',

    # Chapter 11 fixes
    '困此': '因此',
    '实法': '实践',
    '娈化': '变化',
    '的。': '的。',
    '上': '上。',
    '立法的娈化': '立法的变化',

    # Chapter 13 fixes
    '强制指施': '强制措施',
    '法带民事': '附带民事',
    '宪予执行': '先予执行',
    '算议': '复议',
    '本幸': '本章',
    '幸': '章',  # 本幸 -> 本章, 刑幸 -> 刑事

    # 诉法 -> 诉讼法  (partial)
    '刑幸': '刑事',
    '民幸': '民事',

    # Clean up remaining ascii garbage
    '!': '',
    ';': '',
    '}': '',
    '{': '',
    ']': '',
    '[': '',
    '|': '',

    # Fix remaining wrong chars
    '圉': '国',
    '圄': '国',
    '崮': '国',
    '我围': '我国',
    '我困': '我国',
    '困家': '国家',
    '围家': '国家',
    '画家': '国家',
    '国寡': '国家',
    '国窖': '国家',
    '我圉': '我国',
}

def apply_fixes(text):
    result = text
    for wrong, right in fixes.items():
        result = result.replace(wrong, right)
    return result

for item in data:
    item['text'] = apply_fixes(item['text'])

with open('/Users/wangzhida/Downloads/chapters_fixed.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Overcorrections fixed.")
