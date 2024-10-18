#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --job-name=eval
#SBATCH --output /data/elias_stengel_eskin/trained_calibration/logs/eval.out


MODEL="mistralai/Mistral-7B-Instruct-v0.2" 
python trained_calibration/eval/flipflop_eval.py \
	--model ${MODEL} \
	--prefs_dir data/preprocessed_filtered/big_prefs_mistral_v0.2 \
	--output_dir dialogue_models/trivia_qa/mistral_v0.2_7B_base_results_big_${SEED}_seed/ \
	--split test \
	--model_filtering 
