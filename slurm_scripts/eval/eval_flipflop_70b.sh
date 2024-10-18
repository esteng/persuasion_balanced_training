#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=3
#SBATCH --time=02:00:00
#SBATCH --job-name=eval
#SBATCH --output /data/elias_stengel_eskin/trained_calibration/logs/eval.out



python trained_calibration/eval/flipflop_eval.py \
	--model ${MODEL} \
	--prefs_dir data/preprocessed_filtered/big_prefs_llama_3.1 \
	--model_filtering \
	--split test 


