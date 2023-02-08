#!/bin/bash

#SBATCH -A col_seales_uksr
#SBATCH --mail-type=END
#SBATCH --job-name=inkid_general
#SBATCH --output=out/inkid_general_%A.out

module load ccs/singularity

time singularity run --overlay inkid.overlay inkid.sif "$@"
