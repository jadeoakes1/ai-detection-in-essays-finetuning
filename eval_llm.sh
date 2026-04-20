#!/bin/bash
#SBATCH --job-name=llm_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/jadeoakes/capstone/logs/llm_eval_%j.out
#SBATCH --error=/home/jadeoakes/capstone/logs/llm_eval_%j.err

cd /home/jadeoakes/capstone
mkdir -p logs
mkdir -p results/llm

source .venv312/bin/activate
source .env_llm

# Gemini / Vertex setup
export GOOGLE_APPLICATION_CREDENTIALS="/home/jadeoakes/capstone/gemini_key.json"
export GOOGLE_CLOUD_PROJECT=437056147552
export GOOGLE_CLOUD_LOCATION=us-central1
export GOOGLE_GENAI_USE_VERTEXAI=True

PROVIDER="$1"
MODEL_NAME="$2"
DATA_FILE="$3"
DEBUG_N=""
LLAMA_MODE="finetuned"
LLAMA_BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"

# If $4 is a number → it's DEBUG_N
if [[ "$4" =~ ^[0-9]+$ ]]; then
  DEBUG_N="$4"
  LLAMA_MODE="${5:-finetuned}"
  LLAMA_BASE_MODEL="${6:-meta-llama/Llama-3.1-8B-Instruct}"
else
  # Otherwise, shift arguments (no debug_n provided)
  LLAMA_MODE="${4:-finetuned}"
  LLAMA_BASE_MODEL="${5:-meta-llama/Llama-3.1-8B-Instruct}"
fi

if [ -z "$PROVIDER" ] || [ -z "$MODEL_NAME" ] || [ -z "$DATA_FILE" ]; then
  echo "Usage: sbatch eval_llm.sh <provider> <model_name> <data_file> [debug_n] [llama_mode] [llama_base_model]"
  echo "Example: sbatch eval_llm.sh openai gpt-5.4-mini data/splits/source_holdout/test.csv 10"
  echo "Example: sbatch eval_llm.sh llama models/llama31_8b_clean_2000_lora data/splits/balanced/test.csv 50 finetuned meta-llama/Llama-3.1-8B-Instruct"
  echo "Example: sbatch eval_llm.sh llama meta-llama/Llama-3.1-8B-Instruct data/splits/balanced/test.csv 50 zeroshot meta-llama/Llama-3.1-8B-Instruct"
  exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${PROVIDER}_$(basename "$MODEL_NAME" | tr '/' '_')_$(basename "$DATA_FILE" .csv)_${TIMESTAMP}"

if [ -n "$DEBUG_N" ]; then
  RUN_NAME="${RUN_NAME}_debug${DEBUG_N}"
fi

if [ "$PROVIDER" = "llama" ]; then
  RUN_NAME="${RUN_NAME}_${LLAMA_MODE}"
fi

CMD=(
  python scripts/eval_llm.py
  --provider "$PROVIDER"
  --model_name "$MODEL_NAME"
  --data_file "$DATA_FILE"
  --save_predictions "results/llm/${RUN_NAME}_predictions.csv"
  --save_metrics "results/llm/${RUN_NAME}_metrics.json"
)

if [ "$PROVIDER" = "llama" ]; then
  CMD+=(--llama_mode "$LLAMA_MODE" --llama_base_model "$LLAMA_BASE_MODEL")
fi

# Claude-specific retry settings
if [ "$PROVIDER" = "anthropic" ]; then
  CMD+=(--max_retries 6 --sleep_seconds 2)
fi

if [ -n "$DEBUG_N" ]; then
  CMD+=(--debug_n "$DEBUG_N")
fi

"${CMD[@]}"