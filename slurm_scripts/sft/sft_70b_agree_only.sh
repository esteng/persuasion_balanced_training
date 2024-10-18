#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=3
#SBATCH --time=24:00:00
#SBATCH --job-name=sft
#SBATCH --output /data/elias_stengel_eskin/trained_calibration/logs/sft_agree_70b.out


llama_ckpt="meta-llama/Meta-Llama-3.1-70B-Instruct" 
python trained_calibration/rl/train/train_dialogue.py \
	--output_dir dialogue_models/trivia_qa/llama_3.1_70B_sft_agree_filtered_big \
	--prefs_dir  data/preprocessed_filtered/sft_big_prefs_llama_3.1/ \
	--model ${llama_ckpt} \
	--eval_steps 5 \
	--warmup_steps 100 \
	--save_steps 5 \
	--model_filtering \
	--balance \
	--agree_only \
	--valid_limit 1000 \
	--per_device_train_batch_size 6 \
	--gradient_accumulation_steps 13 \
	--per_device_eval_batch_size 6 \
	--n_eval_batches 83 \
	--max_length 1000  \
	--max_steps 240 \
	--seed ${SEED} \
	--sft
	
