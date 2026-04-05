#!/bin/bash
#
# Head-to-tail cyclization script - converts head-to-tail cyclic peptides via relaxation.
# Reads from good_results.jsonl (output of filter_success.sh), then applies
# head-to-tail cyclization in parallel.
#
# Usage:
#   INPUT_DIR=<path_to_candidates_dir> bash scripts/batch_relax_good_results.sh
#
# Environment variables:
#   INPUT_DIR   : Path to the candidates/ directory (default: ./results/LNR_CPSea/condition2_w5_5samples/candidates)
#   OUTPUT_DIR  : Output directory for relaxed PDBs (default: same parent as INPUT_DIR/relaxed)
#   NUM_CORES   : Number of parallel CPU cores (default: 10)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$CODE_DIR"

RELAXER_SCRIPT="${SCRIPT_DIR}/../relaxer/head_tail.py"
INPUT_DIR="${INPUT_DIR:-./results/LNR_CPSea/condition2_w5_5samples/candidates}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$INPUT_DIR")/relaxed}"
NUM_CORES="${NUM_CORES:-10}"
RESULTS_FILE="${INPUT_DIR}/../good_results.jsonl"

CONDA_ENV="${CONDA_ENV:-CycDiff_DPO}"
CONDA_PATH="${CONDA_PATH:-$(dirname "$(dirname "$(which conda)")")}"

echo "=========================================="
echo "  Head-to-Tail Cyclization Pipeline"
echo "=========================================="
echo "Input dir:   $INPUT_DIR"
echo "Results:     $RESULTS_FILE"
echo "Output dir:  $OUTPUT_DIR"
echo "Cores:       $NUM_CORES"
echo "=========================================="
echo ""

# Activate conda environment
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Check good_results.jsonl exists
if [ ! -f "$RESULTS_FILE" ]; then
    echo "Error: good_results.jsonl not found at $RESULTS_FILE"
    echo "Hint: Run filter_success.sh first to generate it."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Prepare tasks from good_results.jsonl
echo "[Step 1] Preparing cyclization tasks from good_results.jsonl..."
python3 -c "
import json
import os

results_file = '$RESULTS_FILE'
output_dir = '$OUTPUT_DIR'
input_dir = '$INPUT_DIR'

valid_count = 0
with open(results_file, 'r') as f:
    for line in f:
        data = json.loads(line.strip())
        pdb_id = data['id']
        gen_pdb = data['gen_pdb']

        if not os.path.exists(gen_pdb):
            continue

        target_output_dir = os.path.join(output_dir, pdb_id)
        os.makedirs(target_output_dir, exist_ok=True)

        output_pdb = os.path.join(target_output_dir, os.path.basename(gen_pdb))
        print(f'{gen_pdb}|{output_pdb}|{pdb_id}')
        valid_count += 1

print(f'Filtered {valid_count} geometrically valid peptides.', file=open(os.devnull, 'w'))
" > /tmp/relax_tasks.txt 2>/dev/null

total_tasks=$(wc -l < /tmp/relax_tasks.txt)
echo "Found $total_tasks valid peptides to cyclize."
echo ""

if [ $total_tasks -eq 0 ]; then
    echo "No valid peptides found."
    exit 0
fi

# Step 2: Parallel cyclization
echo "[Step 2] Running head-to-tail cyclization in parallel..."

process_one_pdb() {
    local input_pdb="\$1"
    local output_pdb="\$2"
    local pdb_id="\$3"
    python3 "$RELAXER_SCRIPT" "$input_pdb" "$output_pdb" "L" > /dev/null 2>&1
    if [ -f "$output_pdb" ]; then
        echo "  [OK] $pdb_id"
    else
        echo "  [FAIL] $pdb_id"
    fi
}
export -f process_one_pdb
export RELAXER_SCRIPT

if command -v parallel &> /dev/null; then
    parallel -j "$NUM_CORES" --will-cite process_one_pdb :::: /tmp/relax_tasks.txt
else
    count=0
    while IFS='|' read -r input_pdb output_pdb pdb_id; do
        process_one_pdb "$input_pdb" "$output_pdb" "$pdb_id" &
        ((count++))
        if [ $((count % NUM_CORES)) -eq 0 ]; then wait; fi
    done < /tmp/relax_tasks.txt
    wait
fi

# Summary
relaxed_count=$(find "$OUTPUT_DIR" -name "*.pdb" | wc -l)
echo ""
echo "=========================================="
echo "  Cyclization complete!"
echo "  Total valid: $total_tasks"
echo "  Cyclized:    $relaxed_count"
echo "  Output:      $OUTPUT_DIR"
echo "=========================================="
