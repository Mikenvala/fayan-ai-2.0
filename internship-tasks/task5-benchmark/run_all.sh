#!/bin/bash
# 多模型法律能力评测一键运行
# 用法: bash run_all.sh
set -e
cd "$(dirname "$0")"
echo "🚀 开始多模型法律能力评测..."
echo "模型: MiniMax-M3 | DeepSeek-V4-Pro | Kimi-K2.7 | Qwen3.7-Max | GLM-5.2 | Mimo"
echo "题目: 30题 (法条/罪名/量刑/案例/程序)"
echo ""
python3 multi_benchmark.py
echo ""
echo "📊 生成对比雷达图..."
python3 compare_chart.py
echo ""
echo "✅ 完成！结果文件:"
echo "   multi_benchmark_results.json  - 详细评测结果"
echo "   figures/fig06_multimodel_radar.png  - 对比雷达图"
