#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=05:00:00
#SBATCH --job-name=ays
#SBATCH --output /data/elias_stengel_eskin/trained_calibration/logs/eval.out



python trained_calibration/eval/areyousure.py \
   --model ${MODEL} \
   --seed ${SEED} \
   --limit 1000 
