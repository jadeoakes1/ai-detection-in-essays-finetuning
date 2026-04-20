#!/bin/bash
#SBATCH --job-name=classifier_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/jadeoakes/capstone/logs/classifier_eval_%j.out
#SBATCH --error=/home/jadeoakes/capstone/logs/classifier_eval_%j.err

cd /home/jadeoakes/capstone
mkdir -p logs
mkdir -p results/classifiers

source .venv/bin/activate

MODEL_DIR="$1"
DATA_FILE="$2"

if [ -z "$MODEL_DIR" ] || [ -z "$DATA_FILE" ]; then
  echo "Usage: sbatch eval_classifier.sh <model_dir> <data_file>"
  echo "Example: sbatch eval_classifier.sh models/deberta_v3_base_source_holdout data/splits/source_holdout/test.csv"
  exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="$(basename "$MODEL_DIR")_$(basename "$DATA_FILE" .csv)_${TIMESTAMP}"

python scripts/eval_classifier.py \
  --model_dir "$MODEL_DIR" \
  --data_file "$DATA_FILE" \
  --save_predictions "results/classifiers/${RUN_NAME}_predictions.csv" \
  --save_metrics "results/classifiers/${RUN_NAME}_metrics.json"
