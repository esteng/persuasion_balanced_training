#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --job-name=traj_eval
#SBATCH --output /data/elias_stengel_eskin/trained_calibration/logs/debug.out


python trained_calibration/eval/trajectory_eval.py \
    --dir ${MODEL_DIR} \
    --baseline \
    --do_postprocess 

python trained_calibration/eval/trajectory_eval.py \
    --dir ${MODEL_DIR} \
    --do_postprocess 