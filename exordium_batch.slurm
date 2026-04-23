#!/bin/bash
#SBATCH --nodes=1
#SBATCH -J exordium_extract
#SBATCH --chdir=/dl_scratch1/berksezer/gaze4hri_new_model
#SBATCH --output=/dl_scratch1/berksezer/gaze4hri_new_model/slurm-%j.out
#SBATCH --error=/dl_scratch1/berksezer/gaze4hri_new_model/slurm-%j.err
#SBATCH --time=24:00:00

echo "Job started on $(hostname) at $(date)"

source /dl_scratch1/berksezer/miniconda3/etc/profile.d/conda.sh
conda activate blink

python exordium_landmarks_batch.py

echo "Job finished at $(date)"
