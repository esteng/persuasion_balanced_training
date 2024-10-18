#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --job-name=eval
#SBATCH --output /data/elias_stengel_eskin/trained_calibration/logs/eval.out

seed=${SEED}
python trained_calibration/eval/get_features.py \
	--jsonl_file dialogue_models/trivia_qa/llama_3.1_8B_combo_dpo_limited_filtered_big_${seed}_seed/flipflop_output.jsonl \
	--prefs_dir data/preprocessed_filtered/big_prefs_llama_3.1 \
	--out_path dialogue_models/trivia_qa/llama_3.1_8B_combo_dpo_limited_filtered_big_${seed}_seed/uncert.json \
	--orig_model meta-llama/Meta-Llama-3.1-8B-Instruct  
