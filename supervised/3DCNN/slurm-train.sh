#!/bin/bash
#SBATCH --gres=gpu:1
# In the above line, replace 1 with the number of GPUs you wish to reserve (1
# or 2).
python experiment.py
