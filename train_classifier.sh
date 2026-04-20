#!/bin/bash
#SBATCH --job-name=classifier_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/jadeoakes/capstone/logs/classifier_train_%j.out
#SBATCH --error=/home/jadeoakes/capstone/logs/classifier_train_%j.err

cd /home/jadeoakes/capstone
mkdir -p logs models

source .venv/bin/activate

MODEL_NAME="$1"
TRAIN_FILE="$2"
DEV_FILE="$3"
TEST_FILE="$4"
OUT_DIR="$5"
LR="${6:-2e-5}"

if [ -z "$MODEL_NAME" ] || [ -z "$TRAIN_FILE" ] || [ -z "$DEV_FILE" ] || [ -z "$TEST_FILE" ] || [ -z "$OUT_DIR" ]; then
  echo "Usage: sbatch train_classifier.slurm <model_name> <train_csv> <dev_csv> <test_csv> <out_dir> [lr]"
  echo 'Example: sbatch train_classifier.slurm microsoft/deberta-v3-base data/splits/source_holdout/train.csv data/splits/source_holdout/dev.csv data/splits/source_holdout/test.csv models/deberta_v3_base_source_holdout 1e-5'
  exit 1
fi

python scripts/train_classifier.py \
  --model_name "$MODEL_NAME" \
  --train_file "$TRAIN_FILE" \
  --dev_file "$DEV_FILE" \
  --test_file "$TEST_FILE" \
  --out_dir "$OUT_DIR" \
  --lr "$LR"
