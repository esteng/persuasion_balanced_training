#!/bin/bash

for file in preprocess_data_sft_llama_3.1 preprocess_data_sft_mistral preprocess_data_llama_3.1 preprocess_data_mistral 
do 
	sbatch slurm_scripts/preprocess/${file}.sh
done

