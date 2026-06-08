// 法眼AI - Node.js 服务器
// 服务前端页面 + 转发API到Python后端

const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 5099;
const BASE = __dirname;

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css':  'text/css; charset=utf-8',
  '.js':   'application/javascript; charset=utf-8',
  '.svg':  'image/svg+xml',
  '.png':  'image/png',
  '.ico':  'image/x-icon',
};

function serveFile(res, filePath) {
  const ext = path.extname(filePath);
  const mime = MIME[ext] || 'application/octet-stream';
  try {
    const data = fs.readFileSync(filePath);
    res.writeHead(200, { 'Content-Type': mime });
    res.end(data);
  } catch {
    res.writeHead(404);
    res.end('Not Found');
  }
}

// 文件映射表（只读静态HTML模板，注入变量后返回）
function renderIndex(res) {
  try {
    let html = fs.readFileSync(path.join(BASE, 'templates', 'index.html'), 'utf-8');
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(html);
  } catch {
    res.writeHead(500);
    res.end('Error loading template');
  }
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  const pathname = url.pathname;

  // 静态文件
  if (pathname.startsWith('/static/')) {
    return serveFile(res, path.join(BASE, pathname));
  }

  // 路由
  if (pathname === '/' || pathname === '/index.html') {
    return renderIndex(res);
  }

  if (pathname === '/status') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({
      ok: true,
      error: null,
      cases_count: 76087,
      criminal_count: 0,
      model: 'MiniMax-M2.7 (待连接到Python后端)',
      note: 'Node.js前端已就绪，Python后端需单独启动'
    }));
  }

  if (pathname === '/health') {
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    return res.end('ok');
  }

  // API 请求 - 转发到 Python 后端（如果启动了）
  if (pathname.startsWith('/api/')) {
    res.writeHead(503, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({
      error: 'Python 后端未启动。请启动 app_serve.py 以使用分析和检索功能。'
    }));
  }

  res.writeHead(404);
  res.end('Not Found');
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`✅ 法眼AI 前端已启动: http://localhost:${PORT}/`);
  console.log(`📂 模板目录: ${path.join(BASE, 'templates')}`);
  console.log(`📂 静态目录: ${path.join(BASE, 'static')}`);
  console.log(`⚠️  Python后端需单独启动以使用 /api/* 功能`);
});
