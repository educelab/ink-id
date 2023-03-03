#!/bin/bash

#SBATCH -A col_seales_uksr
#SBATCH --mail-type=END
#SBATCH --job-name=vc_general
#SBATCH --output=out/vc_general_%A.out

module load ccs/singularity

time singularity run --overlay vc.overlay vc.sif "$@" < input.txt
