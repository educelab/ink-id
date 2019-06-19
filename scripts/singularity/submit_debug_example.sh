#!/bin/bash

#SBATCH -A gol_seales_uksr
#SBATCH -p P4V12_SKY32M192_D 
#SBATCH --gres=gpu:1

#SBATCH --time=00:01:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mhaya2@uky.edu


module load ccs/singularity

singularity run --nv inkid-gpu-copy-code.sif inkid-train-and-predict -h
