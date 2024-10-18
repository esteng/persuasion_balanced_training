#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=24:00:00
#SBATCH --job-name=train
#SBATCH --output /data/elias_stengel_eskin/trained_calibration/logs/train.out




mistral_sft_ckpt="dialogue_models/trivia_qa/mistral_v0.2_7B_sft_limited_filtered_big/checkpoint-240" 
python trained_calibration/rl/train/train_dialogue.py \
	--output_dir dialogue_models/trivia_qa/mistral_v0.2_7B_combo_dpo_limited_filtered_big_${SEED}_seed \
	--prefs_dir data/preprocessed_filtered/big_prefs_mistral_v0.2/ \
	--model ${mistral_sft_ckpt} \
	--eval_steps 50 \
	--warmup_steps 100 \
	--save_steps 50 \
	--valid_limit 1000 \
	--model_filtering \
	--combo_only \
	--balance \
	--per_device_train_batch_size 7 \
	--gradient_accumulation_steps 5 \
	--per_device_eval_batch_size 6 \
	--n_eval_batches 128 \
	--max_length 1000  \
	--num_epochs 5 \
	--seed ${SEED}
	
