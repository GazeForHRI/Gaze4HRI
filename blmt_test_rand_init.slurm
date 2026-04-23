#!/bin/bash
#SBATCH --nodes=1
#SBATCH -J blmt_test
#SBATCH --chdir=/dl_scratch1/berksezer/gaze4hri_new_model
#SBATCH --output=/dl_scratch1/berksezer/gaze4hri_new_model/slurm-%j.out
#SBATCH --error=/dl_scratch1/berksezer/gaze4hri_new_model/slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a6000:1

echo "Job started on $(hostname) at $(date)"

source /dl_scratch1/berksezer/miniconda3/etc/profile.d/conda.sh
conda activate blink

python blmt_test.py --split "val" --weights "/dl_scratch1/berksezer/gaze4hri_dataset/July-Experiments/blink4hri_torch_dataset/checkpoints/o5ufrl90/best.pt" --model_name "blmt_random_init_o5ufrl90_val_split"
echo "Job finished at $(date)"
