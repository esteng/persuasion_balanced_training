#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=5
#SBATCH --time=24:00:00
#SBATCH --job-name=train
#SBATCH --output /data/elias_stengel_eskin/trained_calibration/logs/train_70_main.out




llama_sft_ckpt="dialogue_models/trivia_qa/llama_3.1_70B_sft_limited_filtered_big/checkpoint-240/" 
python trained_calibration/rl/train/train_dialogue.py \
	--output_dir dialogue_models/trivia_qa/llama_3.1_70B_combo_dpo_limited_filtered_big_${SEED}_seed \
	--prefs_dir data/preprocessed_filtered/big_prefs_llama_3.1/ \
	--model ${llama_sft_ckpt} \
	--eval_steps 50 \
	--warmup_steps 100 \
	--save_steps 50 \
	--valid_limit 1000 \
	--model_filtering \
	--combo_only \
	--balance \
	--per_device_train_batch_size 5 \
	--gradient_accumulation_steps 7 \
	--per_device_eval_batch_size 3 \
	--n_eval_batches 128 \
	--max_length 1000  \
	--num_epochs 2 \
	--seed ${SEED}
	
