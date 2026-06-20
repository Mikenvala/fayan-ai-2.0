#!/bin/bash
# 法眼AI 2.0 全页截图脚本
# 前提：python3 app.py 正在运行 (localhost:8800)
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT=screenshots

mkdir -p $OUT

echo "📸 截取仪表盘全页..."
"$CHROME" --headless --disable-gpu --screenshot="$OUT/dashboard_full.png" --window-size=1440,3000 http://localhost:8800/

echo "📸 截取案例分析界面..."
"$CHROME" --headless --disable-gpu --screenshot="$OUT/case_analysis.png" --window-size=1440,2000 "http://localhost:8800/#qa"

echo "📸 截取辩论界面..."
"$CHROME" --headless --disable-gpu --screenshot="$OUT/debate.png" --window-size=1440,2000 "http://localhost:8800/#debate"

echo "📸 截取报告导出界面..."
"$CHROME" --headless --disable-gpu --screenshot="$OUT/export.png" --window-size=1440,2000 "http://localhost:8800/#report"

echo "✅ 完成！截图已保存到 $OUT/"
ls -lh $OUT/
