/* 法眼AI v2 - 健壮版 */
(function() {
  const $ = id => document.getElementById(id);
  if (!$('btnClassify') || !$('btnAnalyze')) {
    console.warn('DOM 未就绪，等待...');
    setTimeout(arguments.callee, 100);
    return;
  }

  const API_BASE = '';
  let classifyData = null;

  async function fetchJSON(url, opts) {
    try {
      const r = await fetch(API_BASE + url, opts);
      return r.ok ? await r.json() : { error: 'HTTP ' + r.status };
    } catch(e) { return { error: e.message }; }
  }

  async function loadExamples() {
    try {
      const d = await fetchJSON('/api/examples');
      if (!d.examples) return;
      const bar = document.createElement('div');
      bar.className = 'examples-bar';
      bar.innerHTML = '<span style="font-size:.78rem;color:#9CA3AF;margin-right:6px;">💡 快速示例：</span>';
      d.examples.forEach(ex => {
        const b = document.createElement('button');
        b.className = 'btn-example'; b.textContent = ex.title;
        b.onclick = () => { $('caseText').value = ex.text; $('charCount').textContent = ex.text.length; if(ex.amount)$('amount').value=ex.amount; if(ex.party_count)$('partyCount').value=ex.party_count; };
        bar.appendChild(b);
      });
      $('caseText').parentElement.insertBefore(bar, $('caseText'));
    } catch(e) {}
  }

  window._fayan = {
    classify: async function() {
      const text = $('caseText').value.trim();
      if (text.length < 5) return alert('请输入至少5个字');
      const btn = $('btnClassify'); btn.disabled = true; btn.innerHTML = '识别中...';
      try {
        const d = await fetchJSON('/api/classify', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({case_text:text}) });
        classifyData = d;
        const cls = $('classifyResult');
        const tc = {'民事':'cls-type-civil','刑事':'cls-type-crime','刑民交叉':'cls-type-cross'};
        cls.style.display = 'flex';
        cls.innerHTML = '<strong style="font-size:.82rem;color:#6B7280;">识别结果：</strong>' +
          `<span class="cls-item ${tc[d.case_type]||''}">${d.case_type||'?'}</span>` +
          `<span class="cls-item cls-confidence">置信度 ${Math.round((d.confidence||0)*100)}%</span>` +
          (d.amount ? `<span class="cls-item cls-confidence">💰 ${fmtAmt(d.amount)}</span>` : '') +
          (d.party_count ? `<span class="cls-item cls-confidence">👥 ${d.party_count} 人</span>` : '');
        if (d.amount && !$('amount').value) $('amount').value = Math.round(d.amount);
        if (d.party_count) $('partyCount').value = d.party_count;
        if (d.case_type === '刑民交叉') $('criminalCross').checked = true;
      } catch(e) { alert('分类失败: '+e.message); }
      finally { btn.disabled = false; btn.innerHTML = '🔍 自动识别'; }
    },

    analyze: async function() {
      const text = $('caseText').value.trim();
      if (text.length < 10) return alert('请输入至少10个字');
      if (text.length > 5000) return alert('请控制在5000字以内');
      
      const btnA = $('btnAnalyze'), btnC = $('btnClassify');
      btnA.disabled = true; btnC.disabled = true;
      btnA.innerHTML = '⏳ 分析中...';
      
      $('emptyState').style.display = 'none';
      $('loadingState').style.display = 'block';
      $('loadingText').textContent = '正在检索相似案例...';
      $('resultState').style.display = 'block';
      $('errorAlert').style.display = 'none';
      $('lawyerAlert').style.display = 'none';
      $('conclusionsList').innerHTML = '';
      $('caseTypeBadge').textContent = '分析中...';
      
      const streamDiv = document.createElement('div');
      streamDiv.style.cssText = 'padding:16px;background:#F9FAFB;border-radius:8px;min-height:100px;font-size:.92rem;line-height:1.8;white-space:pre-wrap;word-break:break-word;';
      $('conclusionsList').appendChild(streamDiv);
      
      try {
        const resp = await fetch(API_BASE + '/api/analyze/stream', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({
            case_text:text,
            amount: parseFloat($('amount').value)||0,
            party_count: parseInt($('partyCount').value)||2,
            has_evidence_gap: $('evidenceGap').checked,
            has_criminal_cross: $('criminalCross').checked,
          })
        });
        
        if (!resp.ok) { $('errorAlert').style.display='block'; $('errorAlert').textContent='❌ HTTP '+resp.status; btnA.disabled=false; btnC.disabled=false; btnA.innerHTML='⚖️ 开始分析'; return; }
        
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        
        while (true) {
          const {done, value} = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, {stream:true});
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';
          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const d = line.slice(6).trim();
            if (d === '[DONE]') continue;
            try {
              const msg = JSON.parse(d);
              if (msg.type === 'meta') {
                $('loadingState').style.display = 'none';
                $('caseTypeBadge').textContent = msg.case_kind==='criminal'?'刑事':'民事';
                $('caseTypeBadge').className = 'tag '+(msg.case_kind==='criminal'?'tag-crime':'tag-civil');
                const cm = {low:'简单',medium:'中等',high:'复杂',ultra:'极复杂'};
                $('complexityBadge').textContent = cm[msg.complexity]||msg.complexity;
                $('complexityBadge').className = 'badge badge-'+(msg.complexity||'medium');
              } else if (msg.type === 'text') {
                streamDiv.textContent += msg.content;
              } else if (msg.type === 'warning') {
                $('lawyerAlert').style.display='block';
                $('lawyerAlert').innerHTML = '⚠️ <strong>建议寻求专业律师</strong><br>'+msg.message;
              } else if (msg.type === 'error') {
                $('errorAlert').style.display='block';
                $('errorAlert').textContent = '❌ '+msg.message;
              }
            } catch(e) {}
          }
        }
        
        // 添加导出按钮
        const bar = document.createElement('div');
        bar.style.cssText = 'display:flex;gap:8px;margin-top:12px;';
        bar.innerHTML = '<button class="btn btn-outline" style="font-size:.82rem;padding:6px 14px;" onclick="navigator.clipboard.writeText(this.parentElement.previousSibling.textContent).then(()=>alert('已复制! '))">📋 复制</button>';
        $('conclusionsList').appendChild(bar);
        
        // 检索相似案例
        try {
          const rd = await fetchJSON('/api/retrieve', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({query:text, top_k:5}) });
          if (rd.results) {
            $('retrieveCard').style.display = 'block';
            $('retrieveSummary').textContent = '找到 '+rd.total+' 条相似案例';
            const list = $('retrieveResults'); list.innerHTML = '';
            rd.results.forEach(r => {
              const div = document.createElement('div');
              div.className = 'retrieve-item';
              div.innerHTML = `<div class="retrieve-header"><span class="retrieve-badge">案例</span><span class="retrieve-score">匹配度 ${(r.score*100).toFixed(1)}%</span></div><h4 class="retrieve-title">${r.title||r.case_number}</h4><p class="retrieve-meta"><span>${r.court}</span><span>${r.cause_of_action}</span></p><p class="retrieve-points">${r.ruling_points||''}</p>`;
              list.appendChild(div);
            });
          }
        } catch(e) {}
        
      } catch(e) {
        $('errorAlert').style.display = 'block';
        $('errorAlert').textContent = '❌ '+e.message;
      }
      btnA.disabled = false; btnC.disabled = false;
      btnA.innerHTML = '⚖️ 开始分析';
    }
  };

  // 初始化
  $('caseText').addEventListener('input', function() { $('charCount').textContent = this.value.length; });
  $('caseText').addEventListener('keydown', function(e) { if(e.ctrlKey && e.key==='Enter') window._fayan.analyze(); });
  loadExamples();
  
  // 状态检查（静默更新）
  fetchJSON('/status').then(d => {
    if (d.ok) {
      $('statusDot').className = 'status-dot online';
      $('statusText').textContent = '在线 · '+d.model+' · '+(d.cases_count||0)+' 条案例';
    }
  }).catch(()=>{});
})();

function fmtAmt(v) {
  if (!v) return '-';
  if (v>=1e8) return (v/1e8).toFixed(1)+'亿元';
  if (v>=1e4) return (v/1e4).toFixed(1)+'万元';
  return v.toLocaleString()+'元';
}
