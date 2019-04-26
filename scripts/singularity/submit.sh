#!/bin/bash

#SBATCH -A gol_seales_uksr
#SBATCH -p <partition name>
#SBATCH --gres=gpu:1
#SBATCH --mem=<memory required, typically 2*(input data size)

#SBATCH --time=<time required in the form of (days):hh:mm:ss>
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your email>

module load ccs/singularity
singularity run --nv inkid-gpu.sif inkid-train-and-predict  \
	<input location> <output location> <other args> 

