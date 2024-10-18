#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --job-name=run_eval_shard
#SBATCH --output /data/elias_stengel_eskin/trained_calibration/logs/debug.out


safe_filename() {
    local input_path="$1"
    local base=$(basename "${input_path}")
    local parent=$(basename $(dirname ${input_path}))
    local path="${parent}/${base}"
    echo "${path//\//-}"
}

SAFE_MODEL_A=$(safe_filename "${MODEL_A}")
SAFE_MODEL_B=$(safe_filename "${MODEL_B}")
echo ${SAFE_MODEL_A}
echo ${SAFE_MODEL_B} 


python trained_calibration/rl/dataset/dialogue_dataset.py \
    --shard_idx ${SHARD_IDX} \
    --limit 1000 \
    --model_a ${MODEL_A} \
    --model_b ${MODEL_B} \
    --both_do_first \
    --out_dir eval/${SAFE_MODEL_A}_AND_${SAFE_MODEL_B} \
    --max_turns 4 \
    --extractor_model "mistralai/Mistral-7B-Instruct-v0.2" \
    --shard_size 500 \
    --deterministic 

