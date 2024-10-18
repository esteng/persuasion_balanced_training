#!/bin/bash

for shard in $(seq 0 10)
do  
    export SHARD_IDX=${shard}
    sbatch  slurm_scripts/run_single_shard.sh 
done
