#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=lunateGridTraining
#SBATCH --output=lunateGridTraining_%A_%a.out
#SBATCH --array=0-9
# In the above line, replace 1 with the number of GPUs you wish to reserve (1
# or 2).

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

# Updated with refactored script format. Will need paths passed to it.
python ~/devel/ink-id/scripts/train-and-predict.py --data DATAPATH --groundtruth GROUNDTRUTH --surfacemask SURFACEMASK --surfacedata SURFACEDATA --gridtestsquare $SLURM_ARRAY_TASK_ID
