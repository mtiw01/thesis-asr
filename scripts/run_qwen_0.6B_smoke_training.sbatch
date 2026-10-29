#!/bin/bash
#SBATCH --job-name=qwen-training-0_6-smoke
#SBATCH --partition=standard
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/qwen-06-training-smoke-%j.out

module load miniforge3/26.3.2-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/matiwa/envs/thesis-asr

cd /scratch/matiwa/thesis-asr

nvidia-smi
./.venv/bin/python -u scripts/qwen_training_0_6.py

