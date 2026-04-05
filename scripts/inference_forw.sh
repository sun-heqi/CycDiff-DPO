#!/bin/bash

CONFIG_PATH="./configs/pepbench/test_LNR.yaml"
CKPT_PATH="./ckpts/dpo/epoch44_step513090.ckpt"
GPU=1
BASE_SAVE_DIR="./results/LNR_CPSea"

export CONDITION=2
export N_SAMPLES=5

for guidance_strength in {5..5}; do
    echo "Running with guidance_strength=${guidance_strength}, n_samples=${N_SAMPLES}"

    sed -i "s/guidance_strength: [0-9]\+/guidance_strength: ${guidance_strength}/" "$CONFIG_PATH"
    sed -i "s/n_samples: [0-9]\+/n_samples: ${N_SAMPLES}/" "$CONFIG_PATH"

    SAVE_DIR="${BASE_SAVE_DIR}/condition${CONDITION}_w${guidance_strength}_${N_SAMPLES}samples"
    python generate.py --config "$CONFIG_PATH" --ckpt "$CKPT_PATH" --gpu $GPU --save_dir "$SAVE_DIR"

    echo "Finished run with guidance_strength=${guidance_strength}, results saved in ${SAVE_DIR}"
done

echo "All experiments completed!"
