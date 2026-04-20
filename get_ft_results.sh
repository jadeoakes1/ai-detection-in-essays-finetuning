#!/bin/bash
#SBATCH --job-name=get_ft_results
#SBATCH --output=logs/openai_training/get_ft_results_%j.out
#SBATCH --error=logs/openai_training/get_ft_results_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=2G

# Go to project directory
cd /home/jadeoakes/capstone

# Make sure directories exist
mkdir -p logs/openai_training

# Activate environment
source .venv312/bin/activate
source .env_llm

# Run your script
python scripts/get_ft_results.py \
  --job_id ftjob-rrQOedo4jPctvrxFqMabhi6D \
  --output_csv logs/openai_training/openai_small1_results.csv \
  --events_output logs/openai_training/openai_small1_events.txt