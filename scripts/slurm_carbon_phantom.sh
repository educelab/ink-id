#!/bin/bash

# Number of GPUs per job. For now is set to one so that we can run
# multiple jobs at once, rather than one job with two GPUs.
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=64
#SBATCH --job-name=k_fold_training_and_prediction
#SBATCH --output=k_fold_training_and_prediction_%A_%a.out

# Array to iterate over, as well as (%) number of jobs to run at once.
#SBATCH --array=0-4%2

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

time inkid-train-and-predict -d ~/data/CarbonPhantomV3.volpkg/working/1/Col1_k-fold-characters-region-set.json -o $1 -k $SLURM_ARRAY_TASK_ID
