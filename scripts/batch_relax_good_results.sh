#!/bin/bash
#
# 并行批量运行 head_tail.py 对 good_results.jsonl 中的所有 PDB 文件进行环化和能量优化
# 直接从 good_results.jsonl 读取文件路径和链信息
# 使用 GNU parallel 或 xargs 并行处理
#

export CUDA_VISIBLE_DEVICES=2

# 设置路径
INPUT_DIR=./results/PPI_targets/condition2_w5_5samples/candidates
OUTPUT_DIR=./results/PPI_targets/condition2_w5_5samples/relaxed
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RELAXER_SCRIPT="${SCRIPT_DIR}/../relaxer/head_tail.py"
GOOD_RESULTS_FILE="${INPUT_DIR}/../good_results.jsonl"

# 设置并行数（根据CPU核心数调整）
NUM_CORES=10

# 检查必要文件
if [ ! -f "$RELAXER_SCRIPT" ]; then
    echo "错误: 找不到 relaxer 脚本: $RELAXER_SCRIPT"
    exit 1
fi

if [ ! -f "$GOOD_RESULTS_FILE" ]; then
    echo "错误: 找不到 good_results.jsonl: $GOOD_RESULTS_FILE"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "并行批量环化处理 - CP-Composer (基于 good_results.jsonl)"
echo "=========================================="
echo "输入文件: $GOOD_RESULTS_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "并行数:   $NUM_CORES"
echo ""

# 创建临时任务列表
TASK_LIST=$(mktemp)
trap "rm -f $TASK_LIST" EXIT

# 解析 JSONL 并生成任务列表
echo "解析 good_results.jsonl 并生成任务列表..."
python3 -c "
import json
import sys
import os

good_results_file = '$GOOD_RESULTS_FILE'
output_dir = '$OUTPUT_DIR'

with open(good_results_file, 'r') as f:
    for line in f:
        data = json.loads(line.strip())
        
        pdb_id = data['id']
        input_pdb = data['gen_pdb']
        peptide_chain = data['lig_chain']
        
        # 检查输入文件是否存在
        if not os.path.exists(input_pdb):
            continue
        
        # 生成输出路径
        target_name = os.path.basename(os.path.dirname(input_pdb))
        filename = os.path.basename(input_pdb).replace('.pdb', '')
        target_output_dir = os.path.join(output_dir, target_name)
        output_pdb = os.path.join(target_output_dir, f'{filename}.pdb')
        
        # 创建输出目录
        os.makedirs(target_output_dir, exist_ok=True)
        
        # 跳过已存在的文件
        if os.path.exists(output_pdb):
            continue
        
        # 输出任务: input_pdb|output_pdb|peptide_chain|pdb_id
        print(f'{input_pdb}|{output_pdb}|{peptide_chain}|{pdb_id}')
" > "$TASK_LIST"

total_tasks=$(wc -l < "$TASK_LIST")
total_in_jsonl=$(wc -l < "$GOOD_RESULTS_FILE")
echo "JSONL 中共有 $total_in_jsonl 个结果"
echo "待处理任务数: $total_tasks"
echo ""

if [ $total_tasks -eq 0 ]; then
    echo "所有文件已处理完成！"
    
    # 统计已完成的数量
    relaxed_count=$(find "$OUTPUT_DIR" -name "*.pdb" | wc -l)
    echo "已有 $relaxed_count 个环化文件"
    exit 0
fi

# 定义处理函数
process_one_pdb() {
    local task_line="$1"
    
    # 解析任务行
    IFS='|' read -r input_pdb output_pdb peptide_chain pdb_id <<< "$task_line"
    
    local filename=$(basename "$input_pdb" .pdb)
    local target_name=$(basename "$(dirname "$input_pdb")")
    
    # 运行 head_tail.py
    python3 "$RELAXER_SCRIPT" "$input_pdb" "$output_pdb" "$peptide_chain" > /dev/null 2>&1
    
    if [ -f "$output_pdb" ]; then
        echo "✓ $target_name/$filename (chain $peptide_chain)"
        return 0
    else
        echo "✗ $target_name/$filename (chain $peptide_chain)"
        return 1
    fi
}

export -f process_one_pdb
export RELAXER_SCRIPT

# 检查是否安装了 GNU parallel
if command -v parallel &> /dev/null; then
    echo "使用 GNU parallel 进行并行处理..."
    echo ""
    
    # 使用 parallel 直接读取每行作为一个参数
    parallel -j "$NUM_CORES" --will-cite process_one_pdb :::: "$TASK_LIST"
    
else
    echo "GNU parallel 未安装，使用后台任务并行处理..."
    echo "提示: 安装 GNU parallel 可获得更好的性能和进度显示"
    echo ""
    
    # 使用简单的后台任务并行
    count=0
    while IFS= read -r task_line; do
        process_one_pdb "$task_line" &
        
        ((count++))
        
        # 控制并发数
        if [ $((count % NUM_CORES)) -eq 0 ]; then
            wait
        fi
    done < "$TASK_LIST"
    
    # 等待所有后台任务完成
    wait
fi

echo ""
echo "=========================================="
echo "统计结果..."
echo "=========================================="

# 统计成功和失败的数量
total_in_jsonl=$(wc -l < "$GOOD_RESULTS_FILE")
relaxed_files=$(find "$OUTPUT_DIR" -name "*.pdb" | wc -l)
failed=$((total_in_jsonl - relaxed_files))

echo "JSONL 总数:  $total_in_jsonl"
echo "已处理:      $relaxed_files"
echo "失败:        $failed"
if [ $total_in_jsonl -gt 0 ]; then
    success_rate=$(awk "BEGIN {printf \"%.1f\", $relaxed_files/$total_in_jsonl*100}")
    echo "成功率:      ${success_rate}%"
fi
echo ""
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="

