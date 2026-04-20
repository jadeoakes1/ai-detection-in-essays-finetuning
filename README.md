# AI-Generated Essay Detection Using Fine-Tuned Language Models

## Overview

This project explores the detection of AI-generated content in essay writing using a combination of fine-tuned transformer-based language models and API-based large language models. As generative AI tools become more prevalent in academic settings, distinguishing between human-written and machine-generated text has become increasingly important.

The goal of this project is to evaluate and compare multiple models, including fine-tuned classifiers and prompt-based LLMs, to determine their effectiveness in identifying AI-generated essays.

---

## Objectives

* Research and develop a reliable system for detecting AI-generated text in essays
* Fine-tune transformer models for binary classification (human vs. AI-written)
* Compare performance across different model architectures
* Evaluate both open-source and API-based models
* Analyze strengths, weaknesses, and failure cases

---

## Setup

Create a `.env` file with the following:

OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...

---

## Models Used

This project experiments with a mix of fine-tuned models and prompt-based LLMs:

### Zeroshot Models

* LLaMA 3.1 8B Instruct
* OpenAI GPT 5.4 mini & GPT 4.1 mini
* Anthropic Claude Sonnet 4-6
* Google Gemini 2.5 Flash

### Fine-Tuned Models

* RoBERTa
* DeBERTa
* Llama 3.1 8B Instruct
* OpenAI GPT 4.1 mini
* Google Gemini 2.5 Flash

---

## Methodology

1. **Dataset Collection**

   * [DAIGT v2 Dataset](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset)
   * Essays sourced from human-written datasets and AI-generated outputs
   * Balanced dataset for fair classification
   * Mini dataset of new human-written and AI generated essays specifically for this project

2. **Preprocessing**

   * Text cleaning and normalization
   * Tokenization using model-specific tokenizers

3. **Model Training**

   * Fine-tuning language models on labeled data
   * Binary classification: Human vs AI-generated

4. **Evaluation**

   * Metrics: Accuracy, Precision, Recall, F1-score
   * Cross-model comparison
   * Testing generalization on unseen prompts

5. **LLM Evaluation**

   * Prompting large models (GPT, Claude, Gemini, LLaMA)
   * Comparing zero-shot with fine-tuned results

---

## Tech Stack

* Python
* PyTorch / Hugging Face Transformers
* Scikit-learn
* APIs from OpenAI, Anthropic, and Google
* VS Code

---

## Project Structure

```
/data
    ├── clean
    ├── example_pool
    ├── finetune
    ├── splits
    ├── daigtv2_stats_2.txt
    ├── daigtve2_stats.txt
/results
/scripts
    ├── preprocessing.py
    ├── train.py
    ├── evaluate.py
    |--- analyze_inter_annotator_agreement.py
    |--- analyze_prompts.py
    |--- batch_convert_ft_datasets.py
    |--- check_gemini_ft_job.py
    |--- check_openai_ft_job.py
    |--- compare_texts.py
    |--- convert_csv_to_ft_jsonl.py
    |--- convert_openai_jsonl_to_gemini.py
    |--- create_gemini_ft_job.py
    |--- create_openai_ft_job.py
    |--- download_aide.py
    |--- download_daigt_v2.py
    |--- eval_annotations_against_gold.py
    |--- eval_classifier.py
    |--- eval_llm.py
    |--- ft_formatters.py
    |--- get_ft_results.py
    |--- inspect_aide.py
    |--- inspect_daigtv2.py
    |--- make_annotation_subset.py
    |--- make_clean_train_subsets.py
    |--- make_splits_balanced.py
    |--- make_splits_baseline.py
    |--- make_splits_source_holdout.py
    |--- prepare_clean_data.py
    |--- sample_training_examples.py
    |--- train_classifier.py
    |--- train_llama_sft.py
    |--- upload_gemini_ft_files.py
    |--- upload_openai_ft_files.py
.gitignore
eval_classifier.sh
eval_llm.sh
get_ft_results.sh
README.md
train_classifier.sh
train_llama_sft.sh
```

---

## Current Status

This project is currently in the **model development and experimentation phase**, with:

* Initial datasets prepared
* Fine-tuning pipelines implemented
* Early testing across multiple models underway

---

## Future Work

* Improve dataset diversity and size
* Explore ensemble methods for better accuracy
* Investigate adversarial examples (hard-to-detect AI text)
* Build a simple web interface for real-time detection
* Optimize model performance and latency

---

## Challenges

* Distinguishing high-quality AI text from human writing
* Avoiding bias toward specific models or prompts
* Generalizing across different writing styles and topics

---

## Author

Jade Oakes

---

## Acknowledgments

* Hugging Face for model architectures and tooling
* OpenAI, Anthropic, and Google for API access to LLMs
* Academic datasets used for training and evaluation

---
