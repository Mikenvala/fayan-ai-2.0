#!/bin/bash
set -e

echo "🚀 法眼AI 2.0 Docker 启动..."
echo "📊 生成仪表盘数据..."
cd /app/platform
python3 data.py
echo "✅ 数据就绪，启动服务..."
exec python3 app.py
