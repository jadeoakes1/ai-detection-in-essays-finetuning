#!/bin/bash
#SBATCH --job-name=llama31_sft
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/jadeoakes/capstone/logs/llama31_sft_%j.out
#SBATCH --error=/home/jadeoakes/capstone/logs/llama31_sft_%j.err

cd /home/jadeoakes/capstone
mkdir -p logs

source .venv312/bin/activate

TRAIN_FILE="$1"
DEV_FILE="$2"
OUTPUT_DIR="$3"

if [ -z "$TRAIN_FILE" ] || [ -z "$DEV_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: sbatch train_llama_sft.sh <train_file> <dev_file> <output_dir>"
  exit 1
fi

echo "===== RUN CONFIG ====="
echo "Train file: $TRAIN_FILE"
echo "Dev file: $DEV_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "======================"

python scripts/train_llama_sft.py \
  --train_file "$TRAIN_FILE" \
  --dev_file "$DEV_FILE" \
  --output_dir "$OUTPUT_DIR"