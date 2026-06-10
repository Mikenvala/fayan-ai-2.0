// 法眼AI v3 - 精简稳定版
const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');

const PORT = 5099;
const BASE = __dirname;
const DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY || "sk-eb33ed735f444009abc1575f4a010c81";
const DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1";
const MODEL = "deepseek-chat";

// ========== Bigram 分词 ==========
function bigramTokenize(text) {
  if (!text) return [];
  const cleaned = text.replace(/[^\u4e00-\u9fff\w]/g, ' ').replace(/\s+/g, ' ').trim();
  const tokens = [];
  for (let i = 0; i < cleaned.length - 1; i++) {
    if (cleaned[i] !== ' ' && cleaned[i+1] !== ' ') tokens.push(cleaned[i] + cleaned[i+1]);
  }
  const words = cleaned.match(/[\u4e00-\u9fff]{2,4}/g);
  if (words) tokens.push(...words);
  return tokens;
}

// ========== 简易 BM25 ==========
class SimpleBM25 {
  constructor() { this.docs = []; this.docTokens = []; this.docLen = []; this.avgLen = 0; this.idf = new Map(); this.docTFs = []; this.k1 = 1.5; this.b = 0.75; }

  build(documents) {
    this.docs = documents;
    const N = documents.length;
    const df = new Map();

    for (let i = 0; i < N; i++) {
      const text = [documents[i].title||'', documents[i].cause_of_action||'', (documents[i].metadata?.ruling_points||''), documents[i].content||''].join(' ');
      const tokens = bigramTokenize(text);
      this.docTokens.push(tokens);
      this.docLen.push(tokens.length);
      const tf = new Map();
      for (const t of tokens) tf.set(t, (tf.get(t)||0)+1);
      this.docTFs.push(tf);
      for (const t of new Set(tokens)) df.set(t, (df.get(t)||0)+1);
    }
    for (const [t, d] of df) this.idf.set(t, Math.log((N-d+0.5)/(d+0.5)+1));
    this.avgLen = this.docLen.reduce((a,b)=>a+b,0)/N;
    console.log('  索引: '+N+' 文档, '+this.idf.size+' token, 均长 '+this.avgLen.toFixed(0));
  }

  search(query, K=5) {
    const qt = bigramTokenize(query).filter(t=>t.length>=2);
    if (!qt.length) return [];
    const N = this.docs.length;
    const scores = new Float64Array(N);
    for (const q of qt) {
      const idf = this.idf.get(q)||0;
      if (!idf) continue;
      for (let i=0;i<N;i++) {
        const tf = this.docTFs[i]?.get(q)||0;
        if (!tf) continue;
        scores[i] += idf * (tf*(this.k1+1)) / (tf+this.k1*(1-this.b+this.b*this.docLen[i]/this.avgLen));
      }
    }
    const idx = Array.from({length:N},(_,i)=>i).filter(i=>scores[i]>0).sort((a,b)=>scores[b]-scores[a]).slice(0,K);
    const mx = idx.length?scores[idx[0]]:1;
    return idx.map(i=>({case:this.docs[i], score:scores[i]/mx}));
  }

  save(file) {
    const d = { docLen: this.docLen, avgLen: this.avgLen, idf: [...this.idf], docTFs: this.docTFs.map(m=>[...m]), n: this.docs.length };
    fs.writeFileSync(file, JSON.stringify(d));
    console.log('  缓存: '+path.basename(file)+' ('+(JSON.stringify(d).length/1024/1024).toFixed(1)+' MB)');
  }

  load(file, docs) {
    if (!fs.existsSync(file)) return false;
    try {
      const d = JSON.parse(fs.readFileSync(file,'utf-8'));
      this.docs = docs; this.docLen = d.docLen; this.avgLen = d.avgLen;
      this.idf = new Map(d.idf); this.docTFs = d.docTFs.map(a=>new Map(a));
      this.docTokens = null; return true;
    } catch(e) { return false; }
  }
}

// ========== 加载 ==========
const civilBM = new SimpleBM25(), criminalBM = new SimpleBM25();
let allCases = [];

function loadCases() {
  console.log('📚 加载案例库...');
  const dir = path.join(BASE, '..', 'extracted_cases', 'deduped');
  const cache = path.join(dir, 'cache'); if (!fs.existsSync(cache)) fs.mkdirSync(cache, {recursive:true});
  const src = [
    { bm: civilBM, data: path.join(dir,'civil_cases.json'), cache: path.join(cache,'civil_v3.json'), label: '民事' },
    { bm: criminalBM, data: path.join(dir,'criminal_cases_new.json'), cache: path.join(cache,'criminal_v3.json'), label: '刑事' },
  ];
  for (const s of src) {
    if (!fs.existsSync(s.data)) continue;
    const cases = JSON.parse(fs.readFileSync(s.data,'utf-8'));
    const cacheOK = fs.existsSync(s.cache) && fs.statSync(s.cache).mtimeMs >= fs.statSync(s.data).mtimeMs;
    if (cacheOK && s.bm.load(s.cache, cases)) {
      console.log('  '+s.label+': '+cases.length+' 条 (缓存)');
    } else {
      console.log('  '+s.label+': 构建索引...');
      s.bm.build(cases);
      s.bm.save(s.cache);
      console.log('  '+s.label+': '+cases.length+' 条');
    }
    allCases.push(...cases);
  }
  console.log('总计 '+allCases.length+' 条案例就绪');
}

// ========== 分类 ==========
const CRIMKW = ["罪","盗窃","抢劫","杀人","故意伤害","诈骗","强奸","贩毒","贪污","受贿","挪用","寻衅滋事","危险驾驶","非法拘禁","绑架","敲诈勒索","职务侵占","走私"];
const CROSSKW = ["刑民交叉","刑事附带民事","附带民事诉讼","移送公安","先刑后民","涉嫌犯罪"];
function classify(text) {
  let cs=0, xs=0;
  for (const k of CRIMKW) if (text.includes(k)) cs++;
  for (const k of CROSSKW) if (text.includes(k)) xs++;
  let t='民事', c=0.8;
  if (xs>=1) {t='刑民交叉';c=0.85}
  else if (cs>=2) {t='刑事';c=0.9}
  else if (cs===1) {t='刑事';c=0.75}
  return {case_type:t, amount:extAmt(text), party_count:extPty(text), confidence:c};
}
function extAmt(text) {
  let m=text.match(/(\d[\d,，.]*)\s*万\s*(?:元|块)/);
  if (m) return parseFloat(m[1].replace(/[,，]/g,''))*10000;
  m=text.match(/(\d[\d,，.]*)\s*亿/);
  if (m) return parseFloat(m[1].replace(/[,，]/g,''))*1e8;
  return null;
}
function extPty(text) {
  const s=new Set();
  let m; const r=/[甲乙丙丁戊己庚辛壬癸](?:某|[A-Z])?/g;
  while((m=r.exec(text))) s.add(m[0]);
  for(const r of ["原告","被告","第三人","上诉人","被上诉人","被告人"]) {
    const p=new RegExp(r+'[^，。,，]{1,6}','g');
    while((m=p.exec(text))) s.add(m[0]);
  }
  const ex=text.match(/(\d+)\s*(?:名|人|位)\s*(?:当事人|被告人)/);
  let c=s.size; if(ex&&parseInt(ex[1])>c) c=parseInt(ex[1]);
  return c>0?Math.min(c,20):null;
}
function complexity(amt,pc,eg,cc) {
  if (cc) return 'ultra';
  let s=0; if(amt>5e5)s++; if(amt>2e6)s++; if(pc>5)s++; if(pc>10)s++; if(eg)s+=2;
  if(s>=4)return'ultra'; if(s>=2)return'high'; if(s>=1)return'medium';
  return'low';
}

// ========== LLM ==========
function callLLM(msgs) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify({model:MODEL,messages:msgs,max_tokens:600,temperature:0.3});
    const u = new URL(DEEPSEEK_BASE_URL+'/chat/completions');
    const req = https.request({hostname:u.hostname,port:443,path:u.pathname,method:'POST',
      headers:{'Authorization':'Bearer '+DEEPSEEK_API_KEY,'Content-Type':'application/json','Content-Length':Buffer.byteLength(data)},timeout:45000},
      res=>{let b='';res.on('data',c=>b+=c);res.on('end',()=>{
        try{resolve(JSON.parse(b).choices?.[0]?.message?.content||b)}catch{resolve(b)}
      })});
    req.on('error',reject);req.on('timeout',()=>{req.destroy();reject(new Error('超时'))});
    req.write(data);req.end();
  });
}

// ========== HTTP ==========
const MIME={'.html':'text/html;charset=utf-8','.css':'text/css','.js':'application/javascript','.svg':'image/svg+xml'};
function serve(res, p) {
  try{const e=path.extname(p);res.writeHead(200,{'Content-Type':MIME[e]||'text/plain'});res.end(fs.readFileSync(p))}
  catch{res.writeHead(404);res.end('404')}
}
function json(res,d,s=200){res.writeHead(s,{'Content-Type':'application/json;charset=utf-8'});res.end(JSON.stringify(d))}
function body(req){return new Promise(r=>{let b='';req.on('data',c=>b+=c);req.on('end',()=>{try{r(JSON.parse(b))}catch{r({})}})});}

const server = http.createServer(async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin','*');
  if (req.method==='OPTIONS'){res.writeHead(204);return res.end()}
  const pn = new URL(req.url,'http://x').pathname;
  if (pn.startsWith('/static/')) return serve(res, path.join(BASE, pn));
  if (pn==='/'||pn==='/index.html') {res.writeHead(200,{'Content-Type':'text/html;charset=utf-8'});return res.end(fs.readFileSync(path.join(BASE,'templates','index.html'),'utf-8'))}
  if (pn==='/health') {res.writeHead(200,{'Content-Type':'text/plain'});return res.end('ok')}
  if (pn==='/status') return json(res, {ok:true,cases_count:allCases.length,model:MODEL,api_key_configured:true});

  if (pn==='/api/classify'&&req.method==='POST') {
    const b=await body(req); if(!b.case_text||b.case_text.length<5) return json(res,{error:'过短'},400);
    return json(res, classify(b.case_text));
  }
  if (pn==='/api/retrieve'&&req.method==='POST') {
    const b=await body(req); const q=(b.query||'').trim(); if(!q) return json(res,{error:'空查询'},400);
    const r=civilBM.search(q, Math.min(b.top_k||5,10));
    return json(res,{query:q,total:r.length,results:r.map(x=>({case_number:x.case.case_number||'',title:x.case.title||'',court:'N/A',cause_of_action:x.case.cause_of_action||'',ruling_points:(x.case.metadata?.ruling_points||'').slice(0,300),score:Math.round(x.score*1e3)/1e3}))});
  }
  if (pn==='/api/examples') return json(res, {examples:[
    {title:'借名账户执行异议',text:'甲借用乙的银行账户收取经营款项，后乙涉及债务纠纷被强制执行，甲主张账户内资金属于自己所有，请求排除强制执行，是否支持？',amount:500000,party_count:3},
    {title:'网络司法拍卖瑕疵',text:'买受人通过司法拍卖购得房屋，入住后发现房屋内曾发生过非正常死亡事件（凶宅），但拍卖公告中未予披露。买受人能否请求撤销拍卖或赔偿损失？',amount:3000000,party_count:4},
    {title:'股东出资纠纷',text:'A公司注册资本1000万元，股东甲认缴600万元、乙认缴400万元，均未实缴。后A公司经营不善，债权人起诉要求甲、乙在未出资范围内对公司债务承担补充赔偿责任。',amount:10000000,party_count:5},
  ]});
  if (pn==='/api/analyze'&&req.method==='POST') {
    const b=await body(req); const t=(b.case_text||'').trim();
    if (!t||t.length<10) return json(res,{error:'需至少10字'},400);
    try {
      const cl=classify(t); const k=cl.case_type; const a=b.amount||cl.amount||0; const p=b.party_count||cl.party_count||2;
      const cp=complexity(a,p,!!b.has_evidence_gap,!!b.has_criminal_cross);
      if (cp==='ultra') return json(res,{conclusions:[],complexity:cp,lawyer_referral:true,lawyer_message:'建议通过12348法律服务热线获取专业律师支持。',case_kind:k==='刑事'?'criminal':'civil',classify:cl});
      const bm = k==='刑事'?criminalBM:civilBM;
      let ret=bm.search(t,3);
      const ctx=ret.map((x,i)=>'【案例'+(i+1)+'】'+x.case.title+'\n案由:'+(x.case.cause_of_action||'')+'\n'+(x.case.metadata?.ruling_points||'').slice(0,100)).join('\n\n');
      const prompt='法律AI。参考'+k+'判例：\n'+ctx+'\n\n案情：'+t+'\n\n简洁分析（结论+依据+建议，300字内）。不做确定性承诺。';
      const ans = await callLLM([{role:'user',content:prompt}]);
      return json(res,{conclusions:[{content:ans,citations:ret.slice(0,2).map(x=>({type:'case',id:x.case.case_number||'',text:(x.case.title||'').slice(0,50)}))}],complexity:cp,lawyer_referral:false,case_kind:k==='刑事'?'criminal':'civil',classify:cl});
    } catch(e) { return json(res,{error:'分析失败:'+e.message},500); }
  }
  res.writeHead(404); res.end('404');
});

loadCases();
server.listen(PORT, '0.0.0.0', () => {
  console.log('✅ 法眼AI v3 就绪: http://localhost:'+PORT+'/');
  console.log('📊 '+allCases.length+' 条案例 | 🤖 '+MODEL);
});
