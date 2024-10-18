#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=3:00:00
#SBATCH --job-name=preproc
#SBATCH --output /data/elias_stengel_eskin/trained_calibration/logs/preprocess.out


SEED=12

llama_sft_ckpt="meta-llama/Meta-Llama-3-8B-Instruct" 
python trained_calibration/rl/train/train_dialogue.py \
	--model ${llama_sft_ckpt} \
	--output_dir "data/preprocessed_filtered/big_prefs_llama_3.1" \
	--write_prefs_to_dir \
	--trajectory_dir data/dialogue_data_persuasion \
	--model_filtering \
	--combo_only \
    --split devtest \
	--seed ${SEED} 
	
