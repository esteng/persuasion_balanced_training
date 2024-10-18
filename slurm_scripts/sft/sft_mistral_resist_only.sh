#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --job-name=sft
#SBATCH --output /data/elias_stengel_eskin/trained_calibration/logs/sft_resist.out


mistral_ckpt="mistralai/Mistral-7B-Instruct-v0.2" 
python trained_calibration/rl/train/train_dialogue.py \
	--output_dir dialogue_models/trivia_qa/mistral_v0.2_7B_sft_resist_filtered_big \
	--prefs_dir  data/preprocessed_filtered/sft_big_prefs_mistral_v0.2/ \
	--model ${mistral_ckpt} \
	--eval_steps 5 \
	--warmup_steps 100 \
	--save_steps 5 \
	--model_filtering \
	--resist_only \
	--balance \
	--valid_limit 1000 \
	--per_device_train_batch_size 8 \
	--gradient_accumulation_steps 10 \
	--per_device_eval_batch_size 6 \
	--n_eval_batches 83 \
	--max_length 1000  \
	--max_steps 240 \
	--seed ${SEED} \
	--sft
	

