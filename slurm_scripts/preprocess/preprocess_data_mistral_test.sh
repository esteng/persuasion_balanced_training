#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=16:00:00
#SBATCH --job-name=preproc
#SBATCH --output /data/elias_stengel_eskin/trained_calibration/logs/preprocess_sft.out


SEED=12

mistral_ckpt="mistralai/Mistral-7B-Instruct-v0.2" 
python trained_calibration/rl/train/train_dialogue.py \
	--model ${mistral_ckpt} \
	--output_dir "data/preprocessed_filtered/big_prefs_mistral_v0.2" \
	--write_prefs_to_dir \
	--trajectory_dir data/dialogue_data_persuasion \
	--model_filtering \
	--combo_only \
	--split devtest \
	--seed ${SEED} 
	
