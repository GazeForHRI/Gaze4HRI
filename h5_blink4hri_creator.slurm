#!/bin/bash
#SBATCH --nodes=1
#SBATCH -J blink4hri_create
#SBATCH --chdir=/dl_scratch1/berksezer/gaze4hri_new_model
#SBATCH --output=/dl_scratch1/berksezer/gaze4hri_new_model/slurm-%j.out
#SBATCH --error=/dl_scratch1/berksezer/gaze4hri_new_model/slurm-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4

echo "Job started on $(hostname) at $(date)"

# Load Conda environment
source /dl_scratch1/berksezer/miniconda3/etc/profile.d/conda.sh
conda activate blink

# Run the creator script
python h5_blink4hri_creator.py

echo "Job finished at $(date)"
