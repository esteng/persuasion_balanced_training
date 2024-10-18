#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=14:00:00
#SBATCH --job-name=train
#SBATCH --output /data/elias_stengel_eskin/trained_calibration/logs/train_resist.out



llama_sft_ckpt="dialogue_models/trivia_qa/llama_3.1_8B_sft_resist_filtered_big/checkpoint-240/"
python trained_calibration/rl/train/train_dialogue.py \
	--output_dir dialogue_models/trivia_qa/llama_3.1_8B_dpo_resist_limited_filtered_big_${SEED}_seed \
	--prefs_dir data/preprocessed_filtered/big_prefs_llama_3.1/ \
	--model ${llama_sft_ckpt} \
	--eval_steps 50 \
	--warmup_steps 100 \
	--save_steps 50 \
	--model_filtering \
	--valid_limit 1000 \
	--per_device_train_batch_size 7 \
	--gradient_accumulation_steps 5 \
	--per_device_eval_batch_size 6 \
	--n_eval_batches 128 \
	--max_length 1000  \
	--seed ${SEED} \
	--num_epochs 5 \
	--resist_only \
	--balance 
	
