#!/bin/bash

#SBATCH -A col_seales_uksr
#SBATCH --mail-type=END
#SBATCH --job-name=inkid_summary
#SBATCH --output=out/inkid_summary_%A_%a.out
#SBATCH --partition=HAS24M128_M
#SBATCH --time=1-00:00:00

# Make rclone available on the container and tell system where to look
#SBATCH --export=SINGULARITY_BIND='/share/singularity/bin',SINGULARITYENV_PREPEND_PATH='/share/singularity/bin'

module load ccs/singularity

time singularity run --overlay inkid.overlay inkid.sif inkid-summary "$1" \
     --all --rclone-transfer-remote dri-experiments-drive