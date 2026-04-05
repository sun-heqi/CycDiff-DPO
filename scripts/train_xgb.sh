#!/bin/bash
#
# XGBoost Ensemble Training for CycDiff-DPO
# Usage: bash scripts/train_xgb.sh
#
# Trains 10-model XGBoost ensemble for Caco2 permeability prediction.
# Output: ckpts/xgboost_ensemble/
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$CODE_DIR"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=40
export MKL_NUM_THREADS=40

echo "========================================="
echo "  XGBoost Ensemble Training"
echo "========================================="
echo "Working directory: $(pwd)"
echo "Conda environment: $CONDA_ENV"
echo "Output: ./ckpts/xgboost_ensemble/"
echo "========================================="
echo ""


python scripts/train_xgb.py

echo ""
echo "========================================="
echo "  Training complete!"
echo "========================================="
