#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=debug
#SBATCH -J blmt_train
#SBATCH --chdir=/dl_scratch1/berksezer/gaze4hri_new_model
#SBATCH --output=/dl_scratch1/berksezer/gaze4hri_new_model/slurm-%j.out
#SBATCH --error=/dl_scratch1/berksezer/gaze4hri_new_model/slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=4

echo "Job started on $(hostname) at $(date)"

source /dl_scratch1/berksezer/miniconda3/etc/profile.d/conda.sh
conda activate blink

python blmt_train.py --weights "none" --run_name "blmt_random_init" --epochs "50"
echo "Job finished at $(date)"
