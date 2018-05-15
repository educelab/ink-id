#!/bin/bash

#SBATCH --job-name=slurm_descriptive_statistics
#SBATCH --output=slurm_descriptive_statistics_%A_%a.out

time python3 /home/$USER/ink-id/scripts/misc/descriptive_statistics.py "$@"

# Backup ~/data/out to Team Google Drive.
time rclone sync -v /home/$USER/data/out/ dri-datasets-drive:ml-results/$USER
