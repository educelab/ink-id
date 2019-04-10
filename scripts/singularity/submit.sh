#!/bin/bash

#SBATCH -A gol_seales_uksr
#SBATCH -p P4V12_SKY32M192_D
#SBATCH --gres=gpu:1

module load ccs/singularity
singularity run --nv inkid-gpu.sif inkid-train-and-predict

