#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --job-name=eval
#SBATCH --output /data/elias_stengel_eskin/trained_calibration/logs/eval.out




MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct" 
python trained_calibration/eval/flipflop_eval.py \
	--model ${MODEL} \
	--prefs_dir data/preprocessed_filtered/big_prefs_llama_3.1 \
	--output_dir dialogue_models/trivia_qa/llama_base_results_big_${SEED}_seed/ \
	--split test \
	--model_filtering 
