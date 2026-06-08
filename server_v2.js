// ╔══════════════════════════════════════════════════════════╗
// ║  法眼AI v2 · 全功能 Node.js 服务器                        ║
// ║  优化: Bigram分词 + BM25检索 + 结构化Prompt + 10K案例库   ║
// ╚══════════════════════════════════════════════════════════╝

const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');

const PORT = 5099;
const BASE = __dirname;
const DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY || "sk-eb33ed735f444009abc1575f4a010c81";
const DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1";
const MODEL = "deepseek-chat";

// ═══════════════════════════════════════════════════════
//  中文 Bigram 分词器（无需 jieba）
// ═══════════════════════════════════════════════════════
function bigramTokenize(text) {
  if (!text) return [];
  // 清理文本
  const cleaned = text.replace(/[^\u4e00-\u9fff\w]/g, ' ').replace(/\s+/g, ' ').trim();
  const tokens = [];
  // Bigram: 相邻字符对
  for (let i = 0; i < cleaned.length - 1; i++) {
    const ch1 = cleaned[i], ch2 = cleaned[i + 1];
    if (ch1 !== ' ' && ch2 !== ' ') {
      tokens.push(ch1 + ch2);
    }
  }
  // 也加入单字（处理奇数字符）
  for (const ch of cleaned) {
    if (ch !== ' ') tokens.push(ch);
  }
  // 提取连续汉字词（2-4字）
  const wordMatches = cleaned.match(/[\u4e00-\u9fff]{2,4}/g);
  if (wordMatches) tokens.push(...wordMatches);
  
  return tokens;
}

// ═══════════════════════════════════════════════════════
//  BM25 检索引擎
// ═══════════════════════════════════════════════════════
class BM25SearchEngine {
  constructor() {
    this.documents = [];       // 原始案例
    this.docTokens = [];       // 每个文档的 token 列表
    this.docLengths = [];      // 每个文档的 token 数
    this.avgDocLen = 0;
    this.idf = new Map();      // token -> IDF 值
    this.totalDocs = 0;
    this.k1 = 1.5;  // BM25 参数
    this.b = 0.75;
  }

  addDocuments(docs) {
    this.documents = docs;
    this.totalDocs = docs.length;
    
    // 1. Tokenize all documents
    const docFreq = new Map(); // token -> 出现文档数
    
    for (let i = 0; i < docs.length; i++) {
      const searchText = this._buildSearchText(docs[i]);
      const tokens = bigramTokenize(searchText);
      
      // 去重统计 DF
      const uniqueTokens = new Set(tokens);
      for (const t of uniqueTokens) {
        docFreq.set(t, (docFreq.get(t) || 0) + 1);
      }
      
      this.docTokens.push(tokens);
      this.docLengths.push(tokens.length);
    }
    
    // 2. 计算 IDF
    const N = this.totalDocs;
    for (const [token, df] of docFreq) {
      this.idf.set(token, Math.log((N - df + 0.5) / (df + 0.5) + 1));
    }
    
    // 3. 平均文档长度
    const totalLen = this.docLengths.reduce((a, b) => a + b, 0);
    this.avgDocLen = totalLen / N;
    
    console.log(`  索引构建完成: ${N} 文档, ${this.idf.size} 唯一 token, 平均长度 ${this.avgDocLen.toFixed(0)}`);
  }

  _buildSearchText(doc) {
    const parts = [
      (doc.title || '').repeat ? (doc.title || '').repeat(3) : (doc.title || ''),
      (doc.metadata?.keywords || []).join(' ').repeat ? (doc.metadata?.keywords || []).join(' ').repeat(2) : (doc.metadata?.keywords || []).join(' '),
      (doc.metadata?.ruling_points || '').repeat ? (doc.metadata?.ruling_points || '').repeat(2) : (doc.metadata?.ruling_points || ''),
      doc.cause_of_action || '',
      doc.content || '',
    ];
    return parts.filter(p => typeof p === 'string').join(' ');
  }

  search(query, topK = 5) {
    const queryTokens = bigramTokenize(query);
    if (queryTokens.length === 0) return [];
    
    const scores = [];
    
    for (let i = 0; i < this.totalDocs; i++) {
      let score = 0;
      const docLen = this.docLengths[i];
      
      for (const qt of queryTokens) {
        const idf = this.idf.get(qt) || 0;
        if (idf === 0) continue;
        
        // 统计 qt 在本文档中的词频
        const tf = this.docTokens[i].filter(t => t === qt).length;
        
        // BM25 公式
        const numerator = tf * (this.k1 + 1);
        const denominator = tf + this.k1 * (1 - this.b + this.b * docLen / this.avgDocLen);
        score += idf * numerator / denominator;
      }
      
      if (score > 0) {
        scores.push({ index: i, score });
      }
    }
    
    // 按分数排序
    scores.sort((a, b) => b.score - a.score);
    
    // 归一化分数
    const maxScore = scores.length > 0 ? scores[0].score : 1;
    
    return scores.slice(0, topK).map(s => ({
      case: this.documents[s.index],
      score: s.score / maxScore, // 归一化到 [0,1]
    }));
  }
}

// ═══════════════════════════════════════════════════════
//  案例库加载
// ═══════════════════════════════════════════════════════
let civilEngine = new BM25SearchEngine();
let criminalEngine = new BM25SearchEngine();
let allCases = [];
let casesReady = false;

function loadCases() {
  console.log('📚 加载案例库...');
  const civilPath = path.join(BASE, '..', 'extracted_cases', 'deduped', 'civil_cases.json');
  const criminalPath = path.join(BASE, '..', 'extracted_cases', 'deduped', 'criminal_cases_new.json');
  
  try {
    if (fs.existsSync(civilPath)) {
      const raw = fs.readFileSync(civilPath, 'utf-8');
      const civilCases = JSON.parse(raw);
      civilEngine.addDocuments(civilCases);
      allCases.push(...civilCases);
      console.log(`  民事案例: ${civilCases.length} 条`);
    }
    if (fs.existsSync(criminalPath)) {
      const raw = fs.readFileSync(criminalPath, 'utf-8');
      const criminalCases = JSON.parse(raw);
      criminalEngine.addDocuments(criminalCases);
      allCases.push(...criminalCases);
      console.log(`  刑事案例: ${criminalCases.length} 条`);
    }
    casesReady = true;
    console.log(`✅ 总计 ${allCases.length} 条案例就绪`);
  } catch (e) {
    console.error('❌ 案例加载失败:', e.message);
  }
}

// ═══════════════════════════════════════════════════════
//  案件分类器
// ═══════════════════════════════════════════════════════
const CRIMINAL_KW = [
  "罪", "盗窃", "抢劫", "故意杀人", "故意伤害", "诈骗", "强奸", "猥亵",
  "走私", "贩毒", "受贿", "行贿", "贪污", "挪用公款", "滥用职权",
  "非法拘禁", "绑架", "敲诈勒索", "抢夺", "侵占", "职务侵占",
  "寻衅滋事", "聚众斗殴", "黑社会", "贩卖毒品", "制造毒品",
  "非法吸收公众存款", "集资诈骗", "洗钱", "逃税", "虚开", "非法经营",
  "拐卖", "虐待", "遗弃", "危险驾驶", "醉驾", "交通肇事",
  "传播淫秽物品", "组织卖淫", "开设赌场", "赌博",
];
const CROSS_KW = ["刑民交叉", "刑事附带民事", "附带民事诉讼", "移送公安", "先刑后民", "涉嫌犯罪"];

function classifyCase(text) {
  let crimScore = 0, crossScore = 0;
  for (const kw of CRIMINAL_KW) if (text.includes(kw)) crimScore++;
  for (const kw of CROSS_KW) if (text.includes(kw)) crossScore++;

  let caseType, conf;
  if (crossScore >= 1) { caseType = "刑民交叉"; conf = 0.85; }
  else if (crimScore >= 2) { caseType = "刑事"; conf = 0.9; }
  else if (crimScore === 1) { caseType = "刑事"; conf = 0.75; }
  else { caseType = "民事"; conf = 0.8; }

  return {
    case_type: caseType,
    amount: extractAmount(text),
    party_count: extractPartyCount(text),
    amount_reason: "",
    party_count_reason: "",
    confidence: Math.round(conf * 100) / 100,
  };
}

function extractAmount(text) {
  let best = null;
  // 匹配"XX万元"等
  const m1 = text.match(/(\d[\d,，.]*)\s*万\s*(?:元|块|圆)/);
  if (m1) best = parseFloat(m1[1].replace(/[,，]/g, '')) * 10000;
  const m2 = text.match(/(\d[\d,，.]*)\s*亿\s*(?:元|块|圆)/);
  if (m2) best = Math.max(best || 0, parseFloat(m2[1].replace(/[,，]/g, '')) * 100000000);
  const m3 = text.match(/(\d[\d,，.]*)\s*(?:元|块)/g);
  if (m3 && !best) {
    const vals = m3.map(s => parseFloat(s.replace(/[,，元块]/g, ''))).filter(v => v > 100);
    if (vals.length) best = Math.max(...vals);
  }
  return best;
}

function extractPartyCount(text) {
  const parties = new Set();
  const letterPat = /[甲乙丙丁戊己庚辛壬癸](?:某|[A-Za-z0-9])?/g;
  let m; while ((m = letterPat.exec(text))) parties.add(m[0]);
  for (const role of ["原告", "被告", "第三人", "上诉人", "被上诉人", "申请人", "被申请人", "被告人"]) {
    const rp = new RegExp(role + '[^，。,，、：:]{1,8}', 'g');
    while ((m = rp.exec(text))) parties.add(m[0]);
  }
  const explicit = text.match(/(\d+)\s*(?:名|人|位|个)\s*(?:当事人|被告人|犯罪嫌疑人)/);
  let count = Math.max(parties.size, explicit ? parseInt(explicit[1]) : 0);
  return count > 0 ? Math.min(count, 20) : null;
}

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

// ═══════════════════════════════════════════════════════
//  MiniMax LLM 调用
// ═══════════════════════════════════════════════════════
function callMiniMax(messages, maxTokens = 2000) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify({ model: MODEL, messages, max_tokens: maxTokens, temperature: 0.3 });
    const url = new URL(DEEPSEEK_BASE_URL + '/chat/completions');
    const req = https.request({
      hostname: url.hostname, port: 443, path: url.pathname, method: 'POST',
      headers: { 'Authorization': `Bearer ${DEEPSEEK_API_KEY}`, 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) },
      timeout: 90000,
    }, (res) => {
      let body = '';
      res.on('data', c => body += c);
      res.on('end', () => {
        try {
          const j = JSON.parse(body);
          resolve(j.choices?.[0]?.message?.content || body);
        } catch { resolve(body); }
      });
    });
    req.on('error', reject);
    req.on('timeout', () => { req.destroy(); reject(new Error('LLM 超时')); });
    req.write(data);
    req.end();
  });
}

// ═══════════════════════════════════════════════════════
//  构建结构化 Prompt（v2 优化）
// ═══════════════════════════════════════════════════════
function buildPrompt(caseText, caseType, retrieved, complexity) {
  // 构建案例上下文
  let contextParts = [];
  let totalChars = 0;
  const MAX_CONTEXT = 500;

  for (let i = 0; i < retrieved.length; i++) {
    const c = retrieved[i].case;
    const score = (retrieved[i].score * 100).toFixed(0);
    const part = [
      `【参考案例${i+1} · 匹配度${score}%】`,
      `标题: ${c.title || '未知'}`,
      `案由: ${c.cause_of_action || 'N/A'}`,
      `要旨: ${(c.metadata?.ruling_points || '').slice(0, 60)}`,
      ,
      ,
    ].filter(Boolean).join('\n');
    
    if (totalChars + part.length > MAX_CONTEXT) break;
    contextParts.push(part);
    totalChars += part.length;
  }

  const context = contextParts.length > 0 
    ? contextParts.join('\n\n' + '─'.repeat(40) + '\n\n')
    : '暂无高度匹配的参考案例，请基于法律知识进行分析。';

  const complexityLabel = { low: '简单', medium: '中等', high: '复杂', ultra: '极复杂' }[complexity] || '中等';

  return `法律AI分析。参考${caseType}判例：

${context}

案情：${caseText}

请简洁分析（结论+依据+建议，不超过300字）。不做确定性承诺。`;
}

// ═══════════════════════════════════════════════════════
//  分析接口（v2 优化版）
// ═══════════════════════════════════════════════════════
async function analyzeCase(caseText, amount, partyCount, hasEvidenceGap, hasCriminalCross) {
  const classifyResult = classifyCase(caseText);
  const caseType = classifyResult.case_type;
  const finalAmount = amount || classifyResult.amount || 0;
  const finalPartyCount = partyCount || classifyResult.party_count || 2;
  const complexity = judgeComplexity(finalAmount, finalPartyCount, hasEvidenceGap, hasCriminalCross);

  // 律师介入
  if (complexity === 'ultra' || (complexity === 'high' && hasEvidenceGap)) {
    return {
      conclusions: [],
      complexity,
      lawyer_referral: true,
      lawyer_message: "基于风险评估，该案件已超出系统智能辅助范围，建议通过12348法律服务热线获取专业律师支持。",
      case_kind: caseType === '刑事' ? 'criminal' : 'civil',
      classify: classifyResult,
    };
  }

  // BM25 检索
  const engine = caseType === '刑事' ? criminalEngine : civilEngine;
  let retrieved = engine.search(caseText, 2);
  
  // 如果本库结果不够，跨库补充
  if (retrieved.length < 3) {
    const crossEngine = caseType === '刑事' ? civilEngine : criminalEngine;
    const crossResults = crossEngine.search(caseText, 2);
    retrieved = [...retrieved, ...crossResults].slice(0, 2);
  }

  // 构建 prompt
  const prompt = buildPrompt(caseText, caseType, retrieved, complexity);

  // 调用 LLM
  const llmResponse = await callMiniMax([{ role: "user", content: prompt }], 2500);

  // 清理 <think> 标签
  const cleanedResponse = llmResponse.replace(/<think>[\s\S]*?<\/think>/g, '').trim();

  const conclusions = [{
    content: cleanedResponse,
    citations: retrieved.slice(0, 2).map(r => ({
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
    classify: classifyResult,
  };
}


// ═══════════════════════════════════════════════════════
//  MiniMax LLM 流式调用 (SSE)
// ═══════════════════════════════════════════════════════
function callMiniMaxStream(messages, maxTokens, onChunk, onDone, onError) {
  const data = JSON.stringify({ model: MODEL, messages, max_tokens: maxTokens, temperature: 0.3, stream: true });
  const url = new URL(DEEPSEEK_BASE_URL + '/chat/completions');
  const req = https.request({
    hostname: url.hostname, port: 443, path: url.pathname, method: 'POST',
    headers: { 'Authorization': `Bearer ${DEEPSEEK_API_KEY}`, 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data), 'Accept': 'text/event-stream' },
    timeout: 120000,
  }, (res) => {
    let buffer = '';
    res.on('data', (chunk) => {
      buffer += chunk.toString();
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const jsonStr = line.slice(6).trim();
          if (jsonStr === '[DONE]') continue;
          try {
            const j = JSON.parse(jsonStr);
            const content = j.choices?.[0]?.delta?.content;
            if (content) onChunk(content);
          } catch {}
        }
      }
    });
    res.on('end', onDone);
    res.on('error', onError);
  });
  req.on('error', onError);
  req.on('timeout', () => { req.destroy(); onError(new Error('LLM 超时')); });
  req.write(data);
  req.end();
}

// ═══════════════════════════════════════════════════════
//  预设示例案例
// ═══════════════════════════════════════════════════════
const EXAMPLE_CASES = [
  {
    title: "借名账户执行异议",
    text: "甲借用乙的银行账户收取经营款项，后乙涉及债务纠纷被强制执行，甲主张账户内资金属于自己所有，请求排除强制执行，是否支持？",
    amount: 500000, party_count: 3,
  },
  {
    title: "网络司法拍卖房屋瑕疵",
    text: "买受人通过司法拍卖购得房屋一套，入住后发现房屋内曾发生过非正常死亡事件（凶宅），但拍卖公告中未予披露。买受人能否请求撤销拍卖或赔偿损失？",
    amount: 3000000, party_count: 4,
  },
  {
    title: "股东出资纠纷",
    text: "A公司注册资本1000万元，股东甲认缴600万元、乙认缴400万元，均未实缴。后A公司经营不善，债权人起诉要求甲、乙在未出资范围内对公司债务承担补充赔偿责任。甲抗辩称公司章程约定的出资期限尚未届满。法院应如何处理？",
    amount: 10000000, party_count: 5,
  },
];


// ═══════════════════════════════════════════════════════
//  进程守护 & 超时处理
// ═══════════════════════════════════════════════════════
process.on('uncaughtException', (err) => {
  console.error('❌ 未捕获异常:', err.message);
  // 不退出，继续服务
});

process.on('unhandledRejection', (reason) => {
  console.error('❌ 未处理 Promise 拒绝:', String(reason).slice(0, 200));
});

// API 超时包装
function withTimeout(promise, ms, fallback) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error('请求超时')), ms))
  ]).catch(err => fallback ? fallback(err) : Promise.reject(err));
}

// LLM 调用加超时兜底
function callMiniMaxSafe(messages, maxTokens = 600) {
  return withTimeout(
    callMiniMax(messages, maxTokens),
    45000,
    (err) => '分析请求超时，请稍后重试。错误信息：' + err.message
  );
}

function callMiniMaxStreamSafe(messages, maxTokens, onChunk, onDone, onError) {
  const timeout = setTimeout(() => {
    onError(new Error('LLM 请求超时(45s)'));
  }, 45000);
  
  callMiniMaxStream(messages, maxTokens,
    (chunk) => { clearTimeout(timeout); onChunk(chunk); },
    () => { clearTimeout(timeout); onDone(); },
    (err) => { clearTimeout(timeout); onError(err); }
  );
}

// ═══════════════════════════════════════════════════════
//  HTTP 服务器
// ═══════════════════════════════════════════════════════
const MIME = { '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8', '.js': 'application/javascript; charset=utf-8', '.svg': 'image/svg+xml', '.ico': 'image/x-icon' };

function serveFile(res, filePath) {
  const ext = path.extname(filePath);
  try {
    const data = fs.readFileSync(filePath);
    res.writeHead(200, { 'Content-Type': MIME[ext] || 'application/octet-stream', 'Cache-Control': 'public, max-age=3600' });
    res.end(data);
  } catch { res.writeHead(404); res.end('Not Found'); }
}

function json(res, data, status = 200) {
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(data));
}

function readBody(req) {
  return new Promise(resolve => {
    let body = '';
    req.on('data', c => body += c);
    req.on('end', () => { try { resolve(JSON.parse(body)); } catch { resolve({}); } });
  });
}

const server = http.createServer(async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') { res.writeHead(204); return res.end(); }

  const url = new URL(req.url, `http://localhost:${PORT}`);
  const pn = url.pathname;

  // 静态资源
  if (pn.startsWith('/static/')) return serveFile(res, path.join(BASE, pn));

  // 测试页
  if (pn === '/test.html') {
    const testHtml = fs.readFileSync(path.join(BASE, 'test.html'), 'utf-8');
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    return res.end(testHtml);
  }

  // 首页
  if (pn === '/' || pn === '/index.html') {
    const html = fs.readFileSync(path.join(BASE, 'templates', 'index.html'), 'utf-8');
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    return res.end(html);
  }

  // 状态
  if (pn === '/status') {
    return json(res, {
      ok: casesReady, cases_count: allCases.length,
      civil_count: allCases.filter(c => !CRIMINAL_KW.some(k => (c.content||'').includes(k))).length,
      criminal_count: allCases.filter(c => CRIMINAL_KW.some(k => (c.content||'').includes(k))).length,
      model: MODEL, api_key_configured: DEEPSEEK_API_KEY.length > 10,
      version: '2.0', features: ['BM25检索', 'Bigram分词', '结构化Prompt', '10K案例库'],
    });
  }

  if (pn === '/health') { res.writeHead(200, { 'Content-Type': 'text/plain' }); return res.end('ok'); }

  // 分类
  if (pn === '/api/classify' && req.method === 'POST') {
    const body = await readBody(req);
    if (!body.case_text || body.case_text.length < 5) return json(res, { error: '案情描述过短' }, 400);
    return json(res, classifyCase(body.case_text));
  }

  // 检索
  if (pn === '/api/retrieve' && req.method === 'POST') {
    const body = await readBody(req);
    const query = (body.query || '').trim();
    const topK = Math.min(body.top_k || 5, 10);
    if (!query) return json(res, { error: 'query 不能为空' }, 400);

    const results = civilEngine.search(query, topK);
    return json(res, {
      query, total: results.length,
      results: results.map(r => ({
        case_number: r.case.case_number || r.case.id || '',
        title: r.case.title || '',
        court: r.case.court || 'N/A',
        cause_of_action: r.case.cause_of_action || 'N/A',
        ruling_points: (r.case.metadata?.ruling_points || r.case.content || '').slice(0, 300),
        score: Math.round(r.score * 1000) / 1000,
      })),
    });
  }

  // 示例案例
  if (pn === '/api/examples') {
    return json(res, { examples: EXAMPLE_CASES });
  }

  // 流式分析 (SSE)
  if (pn === '/api/analyze/stream' && req.method === 'POST') {
    const body = await readBody(req);
    const caseText = (body.case_text || '').trim();
    if (!caseText || caseText.length < 10) {
      res.writeHead(400, { 'Content-Type': 'text/plain' });
      return res.end('data: {"error":"案情描述需至少10字"}\n\n');
    }
    if (caseText.length > 5000) {
      res.writeHead(400, { 'Content-Type': 'text/plain' });
      return res.end('data: {"error":"不超过5000字"}\n\n');
    }

    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'X-Accel-Buffering': 'no',
    });

    try {
      const classifyResult = classifyCase(caseText);
      const caseType = classifyResult.case_type;
      const finalAmount = parseFloat(body.amount) || classifyResult.amount || 0;
      const finalPartyCount = parseInt(body.party_count) || classifyResult.party_count || 2;
      const complexity = judgeComplexity(finalAmount, finalPartyCount, !!body.has_evidence_gap, !!body.has_criminal_cross);

      // 发送元数据
      res.write(`data: ${JSON.stringify({ type: 'meta', classify: classifyResult, complexity, case_kind: caseType === '刑事' ? 'criminal' : 'civil' })}\n\n`);

      if (complexity === 'ultra' || (complexity === 'high' && !!body.has_evidence_gap)) {
        res.write(`data: ${JSON.stringify({ type: 'warning', message: '基于风险评估，该案件已超出系统智能辅助范围，建议通过12348法律服务热线获取专业律师支持。' })}\n\n`);
        res.write('data: [DONE]\n\n');
        return res.end();
      }

      // 检索
      const engine = caseType === '刑事' ? criminalEngine : civilEngine;
      let retrieved = engine.search(caseText, 2);
      if (retrieved.length < 3) {
        const crossEngine = caseType === '刑事' ? civilEngine : criminalEngine;
        retrieved = [...retrieved, ...crossEngine.search(caseText, 2)].slice(0, 5);
      }

      // 发送引用
      const citations = retrieved.slice(0, 2).map(r => ({
        type: 'case', id: r.case.case_number || r.case.id || '',
        text: (r.case.title || '').slice(0, 50),
      }));
      res.write(`data: ${JSON.stringify({ type: 'citations', citations })}\n\n`);

      // 构建 prompt
      const prompt = buildPrompt(caseText, caseType, retrieved, complexity);

      // 流式调用 LLM
      let fullText = "";
      let cleanLen = 0;
      callMiniMaxStreamSafe(
        [{ role: 'user', content: prompt }], 600, (chunk) => {
          fullText += chunk;
          // 去掉 <think>...</think>
          var cleaned = fullText;
          var thinkIdx = cleaned.indexOf("<think>");
          while (thinkIdx >= 0) {
            var closeIdx = cleaned.indexOf("</think>", thinkIdx);
            if (closeIdx >= 0) {
              cleaned = cleaned.slice(0, thinkIdx) + cleaned.slice(closeIdx + 8);
            } else {
              cleaned = cleaned.slice(0, thinkIdx);
              break;
            }
            thinkIdx = cleaned.indexOf("<think>");
          }
          // 只推送新增的干净内容
          var newPart = cleaned.slice(cleanLen);
          cleanLen = cleaned.length;
          if (newPart) {
            res.write("data: " + JSON.stringify({ type: "text", content: newPart }) + "\n\n");
          }
        },
        () => {
          res.write(`data: ${JSON.stringify({ type: 'done', full_length: fullText.length })}\n\n`);
          res.end();
        },
        (err) => {
          if (fullText) {
            res.write(`data: ${JSON.stringify({ type: 'done', full_length: fullText.length })}\n\n`);
          } else {
            res.write(`data: ${JSON.stringify({ type: 'error', message: err.message })}\n\n`);
          }
          res.end();
        }
      );
    } catch (e) {
      console.error('流式分析失败:', e);
      res.write(`data: ${JSON.stringify({ type: 'error', message: e.message })}\n\n`);
      res.end();
    }
    return;
  }

  // 分析（非流式，保留兼容）
  if (pn === '/api/analyze' && req.method === 'POST') {
    const body = await readBody(req);
    const caseText = (body.case_text || '').trim();
    if (!caseText || caseText.length < 10) return json(res, { error: '案情描述需至少10字' }, 400);
    if (caseText.length > 5000) return json(res, { error: '不超过5000字' }, 400);

    try {
      const result = await analyzeCase(
        caseText, parseFloat(body.amount) || 0, parseInt(body.party_count) || 2,
        !!body.has_evidence_gap, !!body.has_criminal_cross
      );
      return json(res, result);
    } catch (e) {
      console.error('分析失败:', e);
      return json(res, { error: `分析失败: ${e.message}` }, 500);
    }
  }

  res.writeHead(404, { 'Content-Type': 'text/html; charset=utf-8' }); res.end('<html><head><meta http-equiv="refresh" content="0;url=/"></head><body>Redirecting...</body></html>');
});

// ═══════════════════════════════════════════════════════
//  启动
// ═══════════════════════════════════════════════════════
console.log('🚀 法眼AI v2 启动中...');
loadCases();

server.listen(PORT, '0.0.0.0', () => {
  console.log(`✅ 服务已启动: http://localhost:${PORT}/`);
  console.log(`📊 案例库: ${allCases.length} 条 | 🤖 ${MODEL}`);
  console.log(`🔍 检索: BM25 + Bigram分词 | 📝 Prompt: 五段式结构化`);
});
