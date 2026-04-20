# AI-Generated Essay Detection Using Fine-Tuned Language Models

## Overview

This project explores the detection of AI-generated content in essay writing using a combination of fine-tuned transformer-based language models and API-based large language models. As generative AI tools become more prevalent in academic settings, distinguishing between human-written and machine-generated text has become increasingly important.

The goal of this project is to evaluate and compare multiple models, including fine-tuned classifiers and prompt-based LLMs, to determine their effectiveness in identifying AI-generated essays.

---

## Objectives

* Develop a reliable system for detecting AI-generated text in essays
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

### Fine-Tuned Models

* RoBERTa
* DeBERTa

### API-Based / Prompted Models

* LLaMA 3.1 8B Instruct
* OpenAI GPT 5.4 mini and 4.1 mini
* Anthropic Claude Sonnet 4-6
* Google Gemini 2.5 Flash

---

## Methodology

1. **Dataset Collection**

   * Essays sourced from human-written datasets and AI-generated outputs
   * Balanced dataset for fair classification

2. **Preprocessing**

   * Text cleaning and normalization
   * Tokenization using model-specific tokenizers

3. **Model Training**

   * Fine-tuning transformer models (RoBERTa, DeBERTa) on labeled data
   * Binary classification: Human vs AI-generated

4. **Evaluation**

   * Metrics: Accuracy, Precision, Recall, F1-score
   * Cross-model comparison
   * Testing generalization on unseen prompts

5. **LLM Evaluation**

   * Prompting large models (GPT, Claude, Gemini, LLaMA)
   * Comparing zero-shot / few-shot detection performance

---

## Tech Stack

* Python
* PyTorch / Hugging Face Transformers
* Scikit-learn
* APIs from OpenAI, Anthropic, and Google
* Jupyter Notebooks / VS Code

---

## Project Structure

```
/data
/models
/notebooks
/src
    ├── preprocessing.py
    ├── train.py
    ├── evaluate.py
/results
README.md
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
