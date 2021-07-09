#!/bin/bash

#SBATCH -A col_seales_uksr
#SBATCH --mail-type=END
#SBATCH --job-name=inkid_rclone_upload
#SBATCH --output=out/inkid_rclone_upload_%A.out
#SBATCH --partition=SKY32M192_L
#SBATCH --time=1-00:00:00

# Make rclone available on the container and tell system where to look
#SBATCH --export=SINGULARITY_BIND='/share/singularity/bin',SINGULARITYENV_PREPEND_PATH='/share/singularity/bin'

module load ccs/singularity

time singularity run --overlay inkid.overlay inkid.sif inkid-rclone-upload "$1"