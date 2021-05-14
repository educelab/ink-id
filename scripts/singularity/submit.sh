#!/bin/bash

#SBATCH -A gol_seales_uksr
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --job-name=inkid
#SBATCH --output=out/inkid_%A_%a.out

# Make rclone available on the container and tell system where to look
#SBATCH --export=SINGULARITY_BIND='/share/singularity/bin',SINGULARITYENV_PREPEND_PATH='/share/singularity/bin'

module load ccs/singularity

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    time singularity run --nv --overlay inkid.overlay inkid.sif inkid-train-and-predict "$@"
else
    time singularity run --nv --overlay inkid.overlay inkid.sif inkid-train-and-predict "$@" --cross-validate-on $SLURM_ARRAY_TASK_ID
fi
