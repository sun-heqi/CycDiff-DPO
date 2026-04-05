#!/bin/bash
#
# 环肽生成结果过滤脚本 - 根据条件2筛选满足 head-to-tail 几何约束的环肽
#
# 条件: N端C原子与C端C原子距离在 3~8 Å 之间
#
# 用法:
#   bash scripts/filter_success.sh <input_jsonl> [output_jsonl]
#
# 示例:
#   bash scripts/filter_success.sh ./results/LNR_CPSea/condition2_w5_40samples/results.jsonl
#   bash scripts/filter_success.sh ./results/results.jsonl ./good_results.jsonl

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$CODE_DIR"


INPUT_JSONL="$1"

if [ -z "$INPUT_JSONL" ]; then
    echo "用法: bash scripts/filter_success.sh <input_jsonl> [output_jsonl]"
    echo ""
    echo "条件: N端C原子与C端C原子距离 3~8 A (head-to-tail)"
    echo ""
    echo "示例:"
    echo "  bash scripts/filter_success.sh ./results/results.jsonl"
    exit 1
fi

if [ ! -f "$INPUT_JSONL" ]; then
    echo "错误: 输入文件不存在: $INPUT_JSONL"
    exit 1
fi

if [ -n "$2" ]; then
    OUTPUT_JSONL="$2"
else
    INPUT_DIR=$(dirname "$INPUT_JSONL")
    OUTPUT_JSONL="${INPUT_DIR}/good_results.jsonl"
fi

echo "========================================="
echo "  Cyclic Peptide Filtering (Condition 2)"
echo "========================================="
echo "Input:  $INPUT_JSONL"
echo "Output: $OUTPUT_JSONL"
echo "Rule:   N-term C to C-term C distance 3~8 A"
echo "========================================="


python evaluate_utils/filter_success.py \
    --input "$INPUT_JSONL" \
    --output "$OUTPUT_JSONL"

echo ""
echo "Done."
