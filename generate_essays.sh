#!/bin/bash
#SBATCH --job-name=generate_essays
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/jadeoakes/capstone/logs/essay_gen_%j.out
#SBATCH --error=/home/jadeoakes/capstone/logs/essay_gen_%j.err

cd /home/jadeoakes/capstone

mkdir -p logs
mkdir -p data/new_dataset/ai_essays

source .venv312/bin/activate
source .env_llm

python generate_essays.py \
  --prompts-file data/new_dataset/prompts.json \
  --output-dir data/new_dataset/ai_essays \
  --candidates-per-prompt 2 \
  --sleep-seconds 1.0