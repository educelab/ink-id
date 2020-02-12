#!/bin/bash

#SBATCH -A gol_seales_uksr
#SBATCH -p P4V12_SKY32M192_S 
#SBATCH --gres=gpu:1
#SBATCH --mem=75G

#SBATCH --time=3:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mhaya2@uky.edu


module load ccs/singularity

singularity run --nv inkid-gpu-new.sif inkid-train-and-predict  \
/scratch/mhaya2/working/2/Col2_k-fold-characters-region-set.json \
/pscratch/seales_uksr/result0/ \
--final-prediction-on-all  \
--training-max-batches 100000  \
--subvolume-shape 34 34 34  

