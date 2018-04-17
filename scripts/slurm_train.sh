#!/bin/bash

#SBATCH --gres=gpu:1
# In the above line, replace 1 with the number of GPUs you wish to reserve (1
# or 2) per job.
#SBATCH --job-name=lunateGridTraining
#SBATCH --output=lunateGridTraining_%A_%a.out
#SBATCH --array=0-9%2

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

time inkid-train-and-predict -d ~/data/lunate-sigma/grid-2x5.json -o ~/data/out -k $SLURM_ARRAY_TASK_ID
