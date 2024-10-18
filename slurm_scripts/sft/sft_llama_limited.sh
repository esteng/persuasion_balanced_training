#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --job-name=sft
#SBATCH --output /data/elias_stengel_eskin/trained_calibration/logs/sft_ltd.out


llama_ckpt="meta-llama/Meta-Llama-3.1-8B-Instruct" 
python trained_calibration/rl/train/train_dialogue.py \
	--output_dir dialogue_models/trivia_qa/llama_3.1_8B_sft_limited_filtered_big \
	--prefs_dir  data/preprocessed_filtered/sft_big_prefs_llama_3.1/ \
	--model ${llama_ckpt} \
	--eval_steps 5 \
	--warmup_steps 100 \
	--save_steps 5 \
	--model_filtering \
	--combo_only \
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
	

	# --train_limit 2354 \