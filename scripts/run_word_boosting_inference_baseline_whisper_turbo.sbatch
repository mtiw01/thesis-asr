#!/bin/bash
#SBATCH --job-name=baseline-word-boosting-inference-whisper-turbo
#SBATCH --partition=standard
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/baseline-word-boosting-inference-whisper-turbo-%j.out

module load miniforge3/26.3.2-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/matiwa/envs/thesis-asr

cd /scratch/matiwa/thesis-asr

./.venv/bin/python -u scripts/baseline_word_boosting_inference_whisper_turbo.py
