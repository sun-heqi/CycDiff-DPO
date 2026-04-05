#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$CODE_DIR"


export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1

echo "========================================="
echo "  Build Preference Pairs with XGBoost"
echo "========================================="
echo "Working directory: $(pwd)"
echo "Conda environment: $CONDA_ENV"
echo "Input:   ./datasets/train_valid/"
echo "Output:  ./datasets/train_valid/generated_pairs.pkl"
echo "========================================="
echo ""


python scripts/build_pairs_xgboost.py \
    --generated_dir ./datasets/train_valid/pdbs \
    --cpsea_file ./datasets/train_valid/all.txt \
    --index_file ./datasets/train_valid/processed/train_index.txt \
    --output_scores ./datasets/train_valid/generated_scores.pkl \
    --output_pairs ./datasets/train_valid/generated_pairs.pkl \
    --min_score_diff 0.1 \
    --seed 42 \
    --n_jobs 40

echo ""
echo "========================================="
echo "  Done"
echo "========================================="
