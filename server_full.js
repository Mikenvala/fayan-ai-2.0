// 法眼AI - 全功能 Node.js 服务器
// 前端页面 + 分析API（MiniMax LLM + 分类 + 检索）

const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');

const PORT = 5099;
const BASE = __dirname;

// MiniMax API 配置
const MINIMAX_API_KEY = process.env.MINIMAX_API_KEY || 
  "sk-api-isSvpHsS_0QtVXKFw1SnQPbylI5mr-8EebhW3qCl3291PK7Rn069JWlElOYSy6z8seZOhWfGtp_n1eIz1cckFz1HQX5a_lq_YBxjzq5yj1gnbRF2uEDb-eU";
const MINIMAX_BASE_URL = "https://api.minimax.chat/v1";
const MODEL = "MiniMax-M2.7";

// ========== 案例数据 ==========
let civilCases = [];
let criminalCases = [];
let casesLoaded = false;

function loadCases() {
  try {
    const civilPath = path.join(BASE, '..', 'extracted_cases', 'deduped', 'civil_cases.json');
    const criminalPath = path.join(BASE, '..', 'extracted_cases', 'deduped', 'criminal_cases_new.json');
    
    if (fs.existsSync(civilPath)) {
      civilCases = JSON.parse(fs.readFileSync(civilPath, 'utf-8'));
    }
    if (fs.existsSync(criminalPath)) {
      criminalCases = JSON.parse(fs.readFileSync(criminalPath, 'utf-8'));
    }
    casesLoaded = true;
    console.log(`案例加载: 民事 ${civilCases.length} 条, 刑事 ${criminalCases.length} 条`);
  } catch (e) {
    console.warn('案例加载失败:', e.message);
  }
}
loadCases();

// ========== CaseClassifier (Node.js 实现) ==========
const CRIMINAL_KEYWORDS = [
  "罪", "盗窃", "抢劫", "杀人", "故意伤害", "诈骗", "强奸", "猥亵",
  "走私", "贩毒", "吸毒", "赌博", "开设赌场", "组织卖淫",
  "贪污", "贿赂", "受贿", "行贿", "挪用", "滥用职权", "玩忽职守",
  "故意杀人", "过失致人死亡", "交通肇事", "非法拘禁", "绑架",
  "敲诈勒索", "抢夺", "侵占", "职务侵占", "寻衅滋事", "聚众斗殴",
  "黑社会", "贩卖毒品", "制造毒品", "非法吸收公众存款", "集资诈骗",
  "洗钱", "逃税", "虚开", "非法经营", "拐卖", "虐待",
  "危险驾驶", "醉驾", "传播淫秽物品", "伪造", "冒充",
];

const CROSS_KEYWORDS = [
  "刑民交叉", "刑事附带民事", "附带民事诉讼", "刑事追诉",
  "同时涉及刑事", "涉嫌犯罪", "移送公安", "先刑后民",
];

function classifyCase(text) {
  let criminalScore = 0;
  let crossScore = 0;

  for (const kw of CRIMINAL_KEYWORDS) {
    if (text.includes(kw)) criminalScore++;
  }
  for (const kw of CROSS_KEYWORDS) {
    if (text.includes(kw)) crossScore++;
  }

  let caseType, confidence;
  if (crossScore >= 1 || (criminalScore >= 1 && crossScore >= 1)) {
    caseType = "刑民交叉"; confidence = 0.85;
  } else if (criminalScore >= 2) {
    caseType = "刑事"; confidence = 0.9;
  } else if (criminalScore === 1) {
    caseType = "刑事"; confidence = 0.75;
  } else {
    caseType = "民事"; confidence = 0.8;
  }

  // 提取金额
  const amount = extractAmount(text);
  // 提取当事人数
  const partyCount = extractPartyCount(text);

  return { case_type: caseType, amount, party_count: partyCount, 
           amount_reason: amount ? `识别金额约 ${fmtAmount(amount)}` : "未识别到具体金额",
           party_count_reason: partyCount ? `识别到约 ${partyCount} 方当事人` : "无法识别当事人数",
           confidence: Math.round(confidence * 100) / 100 };
}

function extractAmount(text) {
  const patterns = [
    [/(\d[\d,，.]*)\s*万\s*(?:元|块)/g, 10000],
    [/(\d[\d,，.]*)\s*亿\s*(?:元|块)/g, 100000000],
    [/(\d[\d,，.]*)\s*(?:元|块)/g, 1],
    [/(\d[\d,，.]*)\s*(?:万元|万圆)/g, 10000],
  ];
  let candidates = [];
  for (const [pat, mult] of patterns) {
    let m;
    while ((m = pat.exec(text)) !== null) {
      const num = parseFloat(m[1].replace(/[,，]/g, ''));
      if (num > 0) candidates.push(num * mult);
    }
  }
  return candidates.length > 0 ? Math.max(...candidates) : null;
}

function extractPartyCount(text) {
  const parties = new Set();
  const letterPat = /[甲乙丙丁戊己庚辛壬癸](?:某|[A-Za-z0-9])?/g;
  let m;
  while ((m = letterPat.exec(text)) !== null) parties.add(m[0]);
  
  for (const role of ["原告", "被告", "第三人", "上诉人", "被上诉人", "申请人", "被申请人"]) {
    const rolePat = new RegExp(role + '[^，。,，、：:]{1,8}', 'g');
    while ((m = rolePat.exec(text)) !== null) parties.add(m[0]);
  }
  
  const defPat = /被告人[^，。,，、：:]{1,8}/g;
  while ((m = defPat.exec(text)) !== null) parties.add(m[0]);
  
  const multiPat = /[各多三]方/g;
  while ((m = multiPat.exec(text)) !== null) parties.add(m[0]);
  
  parties.add("当事人");
  
  const explicit = text.match(/(\d+)\s*(?:名|人|位|个)\s*(?:当事人|被告人|犯罪嫌疑人|原告|被告)/);
  let count = parties.size;
  if (explicit && parseInt(explicit[1]) > count) count = parseInt(explicit[1]);
  
  return count > 0 ? Math.min(count, 20) : null;
}

function fmtAmount(val) {
  if (!val) return '-';
  if (val >= 1e8) return (val / 1e8).toFixed(1) + ' 亿元';
  if (val >= 1e4) return (val / 1e4).toFixed(1) + ' 万元';
  return val.toLocaleString() + ' 元';
}

// ========== 复杂度判断 ==========
function judgeComplexity(amount, partyCount, hasEvidenceGap, hasCriminalCross) {
  if (hasCriminalCross) return "ultra";
  let score = 0;
  if (amount > 500000) score++;
  if (amount > 2000000) score++;
  if (partyCount > 5) score++;
  if (partyCount > 10) score++;
  if (hasEvidenceGap) score += 2;
  if (score >= 4) return "ultra";
  if (score >= 2) return "high";
  if (score >= 1) return "medium";
  return "low";
}

// ========== 简单关键词检索（不依赖jieba） ==========
function simpleRetrieve(query, cases, topK = 5) {
  if (!cases || cases.length === 0) return [];
  
  // 用查询词分词简单匹配
  const queryChars = new Set(query.replace(/\s+/g, '').split(''));
  
  const scored = cases.map(c => {
    const searchText = [
      c.title || '', c.cause_of_action || '', c.content || '',
      (c.metadata?.ruling_points || ''), (c.metadata?.keywords || []).join(' ')
    ].join(' ');
    
    let score = 0;
    // 字符重叠度
    for (const ch of queryChars) {
      if (searchText.includes(ch)) score++;
    }
    // 词组匹配
    const queryWords = query.split(/[,，。.、；;！!？?\s]+/).filter(w => w.length >= 2);
    for (const word of queryWords) {
      if (searchText.includes(word)) score += word.length * 2;
    }
    
    return { case: c, score: score / Math.max(searchText.length, 1) };
  });
  
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK);
}

// ========== MiniMax LLM 调用 ==========
function callMiniMaxLLM(messages) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify({
      model: MODEL,
      messages: messages,
      max_tokens: 2000,
      temperature: 0.3,
    });

    const url = new URL(MINIMAX_BASE_URL + '/chat/completions');
    const options = {
      hostname: url.hostname,
      port: 443,
      path: url.pathname,
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${MINIMAX_API_KEY}`,
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(data),
      },
      timeout: 60000,
    };

    const req = https.request(options, (res) => {
      let body = '';
      res.on('data', chunk => body += chunk);
      res.on('end', () => {
        try {
          const json = JSON.parse(body);
          if (json.choices && json.choices[0]) {
            resolve(json.choices[0].message.content);
          } else {
            reject(new Error(json.error?.message || 'LLM 返回格式异常'));
          }
        } catch (e) {
          reject(new Error('解析 LLM 响应失败: ' + body.slice(0, 200)));
        }
      });
    });

    req.on('error', reject);
    req.on('timeout', () => { req.destroy(); reject(new Error('LLM 请求超时')); });
    req.write(data);
    req.end();
  });
}

// ========== 完整分析流程 ==========
async function analyzeCase(caseText, amount, partyCount, hasEvidenceGap, hasCriminalCross) {
  // Step 1: 分类
  const classifyResult = classifyCase(caseText);
  const caseType = classifyResult.case_type;
  const finalAmount = amount || classifyResult.amount || 0;
  const finalPartyCount = partyCount || classifyResult.party_count || 2;

  // Step 2: 复杂度
  const complexity = judgeComplexity(finalAmount, finalPartyCount, hasEvidenceGap, hasCriminalCross);

  // Step 3: 律师介入检查
  if (complexity === 'high' || complexity === 'ultra' || hasEvidenceGap) {
    return {
      conclusions: [],
      complexity,
      lawyer_referral: true,
      lawyer_message: "基于目前风险评估，该案件已超出系统智能辅助范围，建议您考虑专业律师支持。如需了解公共法律服务资源，可通过12348法律服务热线获取帮助。",
      case_kind: caseType === '刑事' ? 'criminal' : 'civil',
    };
  }

  // Step 4: 检索相似案例
  const targetCases = caseType === '刑事' ? criminalCases : civilCases;
  const retrieved = simpleRetrieve(caseText, targetCases, 3);

  // Step 5: 构建 prompt
  let contextParts = [];
  for (let i = 0; i < retrieved.length; i++) {
    const c = retrieved[i].case;
    const part = `【案例${i+1}】${c.title || ''}
案由: ${c.cause_of_action || ''} | 法院: ${c.court || ''}
裁判要点: ${c.metadata?.ruling_points || c.content?.slice(0, 300) || ''}`;
    contextParts.push(part);
  }
  const context = contextParts.join('\n\n') || '暂无相关案例';

  const prompt = `你是一位专业的法律AI助手。请根据以下${caseType}相关判例，对用户的问题进行分析。

【相关判例】
${context}

【用户案情】
${caseText}

请结合判例给出专业分析。回答格式：
1. 先给出结论性意见
2. 结合判例说明法律依据
3. 如有不同观点或特殊情况请说明

注意：请勿做出"一定会赢/会输/胜诉率"等确定性承诺，仅提供法律分析参考。`;

  // Step 6: 调用 LLM
  const llmResponse = await callMiniMaxLLM([
    { role: "user", content: prompt }
  ]);

  // Step 7: 构建结论
  const conclusions = [{
    content: llmResponse,
    citations: retrieved.map(r => ({
      type: "case",
      id: r.case.case_number || r.case.id || '',
      text: (r.case.title || '').slice(0, 50),
    })),
  }];

  return {
    conclusions,
    complexity,
    lawyer_referral: false,
    lawyer_message: "",
    case_kind: caseType === '刑事' ? 'criminal' : 'civil',
  };
}

// ========== HTTP 服务器 ==========
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.svg': 'image/svg+xml',
};

function serveFile(res, filePath) {
  const ext = path.extname(filePath);
  try {
    const data = fs.readFileSync(filePath);
    res.writeHead(200, { 'Content-Type': MIME[ext] || 'application/octet-stream' });
    res.end(data);
  } catch {
    res.writeHead(404);
    res.end('Not Found');
  }
}

function json(res, data, status = 200) {
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(data, null, 2));
}

function readBody(req) {
  return new Promise((resolve) => {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      try { resolve(JSON.parse(body)); }
      catch { resolve({}); }
    });
  });
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  const pathname = url.pathname;

  // 静态文件
  if (pathname.startsWith('/static/')) {
    return serveFile(res, path.join(BASE, pathname));
  }

  // 首页
  if (pathname === '/' || pathname === '/index.html') {
    const html = fs.readFileSync(path.join(BASE, 'templates', 'index.html'), 'utf-8');
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    return res.end(html);
  }

  // API: 状态
  if (pathname === '/status') {
    return json(res, {
      ok: true,
      cases_count: civilCases.length + criminalCases.length,
      criminal_count: criminalCases.length,
      model: MODEL,
      api_key_configured: !!MINIMAX_API_KEY && MINIMAX_API_KEY.length > 10,
    });
  }

  // API: 健康检查
  if (pathname === '/health') {
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    return res.end('ok');
  }

  // API: 自动分类
  if (pathname === '/api/classify' && req.method === 'POST') {
    const body = await readBody(req);
    if (!body.case_text || body.case_text.length < 5) {
      return json(res, { error: '案情描述过短' }, 400);
    }
    return json(res, classifyCase(body.case_text));
  }

  // API: 检索
  if (pathname === '/api/retrieve' && req.method === 'POST') {
    const body = await readBody(req);
    const query = body.query || '';
    const topK = Math.min(body.top_k || 5, 10);
    if (!query) return json(res, { error: 'query 不能为空' }, 400);

    const results = simpleRetrieve(query, [...civilCases, ...criminalCases], topK);
    return json(res, {
      query, total: results.length,
      results: results.map(r => ({
        case_number: r.case.case_number || r.case.id || '',
        title: r.case.title || '',
        court: r.case.court || 'N/A',
        cause_of_action: r.case.cause_of_action || 'N/A',
        ruling_points: (r.case.metadata?.ruling_points || '').slice(0, 300),
        score: Math.round(r.score * 1000) / 1000,
      })),
    });
  }

  // API: 分析
  if (pathname === '/api/analyze' && req.method === 'POST') {
    const body = await readBody(req);
    const caseText = (body.case_text || '').trim();
    if (!caseText || caseText.length < 10) {
      return json(res, { error: '案情描述需至少10字' }, 400);
    }
    if (caseText.length > 5000) {
      return json(res, { error: '案情描述不超过5000字' }, 400);
    }

    try {
      const result = await analyzeCase(
        caseText,
        parseFloat(body.amount) || 0,
        parseInt(body.party_count) || 2,
        !!body.has_evidence_gap,
        !!body.has_criminal_cross,
      );
      return json(res, result);
    } catch (e) {
      console.error('分析失败:', e);
      return json(res, { error: `分析失败: ${e.message}` }, 500);
    }
  }

  // 404
  res.writeHead(404);
  res.end('Not Found');
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`✅ 法眼AI 全功能服务器已启动: http://localhost:${PORT}/`);
  console.log(`📋 API: /api/classify | /api/retrieve | /api/analyze`);
  console.log(`🤖 LLM: ${MODEL} | Key: ${MINIMAX_API_KEY ? '已配置 ✓' : '未配置 ✗'}`);
  console.log(`📚 案例库: ${civilCases.length + criminalCases.length} 条`);
});
