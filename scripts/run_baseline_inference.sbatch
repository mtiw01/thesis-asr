#!/bin/bash
#SBATCH --job-name=baseline-inference
#SBATCH --partition=standard
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/baseline-inference-%j.out

module load miniforge3/26.3.2-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/matiwa/envs/thesis-asr

cd /scratch/matiwa/thesis-asr

./.venv/bin/python -u scripts/baseline_inference.py
