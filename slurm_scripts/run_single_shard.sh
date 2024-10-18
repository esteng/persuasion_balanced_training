#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name=run_calibration_shard
#SBATCH --output logs/run_shard.out

conda activate cal

python trained_calibration/rl/dataset/dialogue_dataset.py \
    --shard_idx ${SHARD_IDX} \
    --limit 76495 \
    --both_do_first \
    --out_dir data/dialogue_data_persuasion_more_more \
    --max_turns 4 \
    --shard_size 8000   

