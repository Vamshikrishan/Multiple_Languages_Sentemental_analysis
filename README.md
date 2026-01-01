# Multiple_Languages_Sentemental_analysis

A research / engineering repository for multilingual sentiment analysis across multiple languages and datasets. This project provides tools and example pipelines for preprocessing multilingual text, training and evaluating transformer-based and classical baseline models, and running inference in a repeatable way.

> Note: repository name intentionally preserves the original spelling "Sentemental". Replace with "Sentimental" in your own forks if preferred.

---

## Table of Contents

- [Project overview](#project-overview)
- [Key features](#key-features)
- [Supported languages](#supported-languages)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Data format and datasets](#data-format-and-datasets)
- [Models & Methods](#models--methods)
- [Training & evaluation](#training--evaluation)
- [Inference example](#inference-example)
- [Notebooks & experiments](#notebooks--experiments)
- [Reproducibility](#reproducibility)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements & citation](#acknowledgements--citation)
- [Contact](#contact)

---

## Project overview

Multilingual sentiment analysis presents challenges such as differing tokenization, idioms, code-switching, and resource imbalance between languages. This repository aims to:

- Provide a modular pipeline for preprocessing multilingual corpora.
- Offer training scripts for transformer-based multilingual encoders (e.g., XLM-R, mBERT) and classical baselines.
- Provide evaluation tools for per-language and cross-lingual metrics.
- Make it straightforward to add new languages and datasets.

The code emphasizes clarity and reproducibility so it can be used for experimentation and small-scale production.

---

## Key features

- Preprocessing utilities for multilingual text (normalization, tokenization, language-aware cleaning).
- Training/evaluation scripts compatible with Hugging Face Transformers.
- Baseline classical models (TF-IDF + Logistic Regression / SVM) for quick comparisons.
- Scripts for cross-lingual transfer experiments (train on language A, test on language B).
- Notebook examples for exploratory data analysis and model inspection.
- Experiment logging (e.g., CSV/JSON, support for simple MLFlow or Weights & Biases hooks).

---

## Supported languages

Out of the box the project includes examples and evaluation scripts for (configurable and extensible):

- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Hindi (hi)
- Arabic (ar)
- Portuguese (pt)
- Italian (it)
- Russian (ru)
- Chinese (zh)

You can add more languages by supplying data in the expected format (see "Data format and datasets").

---

## Repository structure

(Example — adjust to actual layout in the repository.)

- data/  
  - raw/ — raw downloaded datasets
  - processed/ — preprocessed CSV/JSON used for training
- src/  
  - preprocess.py — text cleaning & dataset merging utilities  
  - features.py — TF-IDF / embedding feature extraction  
  - train.py — training loop for transformers & classical models  
  - evaluate.py — evaluation metrics and per-language reports  
  - inference.py — single/batch inference utilities  
  - utils/ — helpers, tokenizers, data loaders
- notebooks/ — exploratory analysis and demo notebooks
- configs/ — YAML/JSON configs for experiments
- models/ — saved model checkpoints
- experiments/ — logs, metric outputs, plots
- requirements.txt
- README.md

If your repository structure differs, adapt commands below to match.

---

## Installation

Recommended: Python 3.8+ and a virtual environment.

1. Clone the repository:
   git clone https://github.com/Vamshikrishan/Multiple_Languages_Sentemental_analysis.git
   cd Multiple_Languages_Sentemental_analysis

2. Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\Scripts\activate       # Windows (PowerShell: venv\Scripts\Activate.ps1)

3. Install dependencies:
   pip install -r requirements.txt

Typical dependencies (examples):
- torch (>=1.8)
- transformers (Hugging Face)
- datasets (Hugging Face Datasets)
- scikit-learn
- pandas
- numpy
- sentencepiece (for some tokenizers)
- tqdm

If you plan to use GPU, install the appropriate PyTorch build for your CUDA version.

---

## Quick start

1. Preprocess data
   - Example:
     python src/preprocess.py --input data/raw/twitter_multilingual.csv --output data/processed/twitter_cleaned.csv --lang-column lang --text-column text --label-column label

   - Preprocessing tasks typically include: lowercasing (language sensitive), removing URLs/mentions, Unicode normalization, optional language-specific tokenization.

2. Train model
   - Transformer (example with XLM-R):
     python src/train.py --config configs/train_xlm-r.yaml --device cuda

   - Classical baseline:
     python src/train.py --config configs/train_baseline_tfidf_lr.yaml

3. Evaluate
   - Example:
     python src/evaluate.py --model-path models/xlm-r-run1 --test-file data/processed/test.csv --report experiments/xlm-r-run1/report.json

All scripts accept a `--config` option to centralize hyperparameters (learning rate, batch size, max epochs, seed, tokenizer/model name, etc).

---

## Data format and datasets

Expected processed CSV format (one row per sample):

- id: unique identifier
- text: raw or preprocessed text
- label: sentiment label (e.g., positive / neutral / negative or numeric 0/1/2)
- lang: language code (ISO 639-1), e.g., en, es, fr

Example:
id,text,label,lang
1,"I love this product!",positive,en

Datasets commonly used (links & licensing vary — read dataset licenses before use):
- Sentiment140 (Twitter) — English
- Multilingual Amazon Reviews (Hugging Face Datasets)
- SemEval (various tasks, some multilingual)
- Twitter corpora from shared tasks
- User-contributed translated datasets

To add a new dataset: place raw files in `data/raw/` and add an entry or script to `src/preprocess.py` to convert it to the expected CSV.

---

## Models & Methods

Transformer-based approaches:
- XLM-R (XLM-Roberta) — strong multilingual encoder
- mBERT — multilingual BERT
- Fine-tuning heads: classification head with one or two linear layers and dropout
- Options for pooling: [CLS] token, mean pooling, max pooling

Baseline/classical approaches:
- TF-IDF vectorizer + Logistic Regression
- TF-IDF + Linear SVM
- FastText embeddings + simple feed-forward network

Cross-lingual strategies:
- Train on multilingual combined dataset
- Translate-train or translate-test (use translation tools / APIs)
- Zero-shot transfer: train on high-resource language and evaluate on low-resource languages using a multilingual encoder

---

## Training & evaluation

Key considerations:
- Use stratified splits per-language if class imbalance exists.
- Use macro F1 for balanced evaluation across classes and languages; also report per-language F1.
- Save model checkpoints and best validation checkpoint by chosen metric (e.g., validation F1).
- Use learning rate schedulers and weight decay when fine-tuning transformers.

Suggested example hyperparameters for XLM-R:
- batch_size: 16 (per GPU)
- learning_rate: 2e-5
- epochs: 3-5
- max_seq_length: 128
- optimizer: AdamW

Evaluation metrics:
- Accuracy
- Precision / Recall / F1 (macro & per-class)
- Confusion matrices (per-language)
- ROC/AUC (where applicable)

Example evaluation command:
python src/evaluate.py --predictions experiments/xlm-r-run1/preds.csv --gold data/processed/test.csv --output experiments/xlm-r-run1/metrics.json

---

## Inference example

Python API usage (example):

from src.inference import SentimentPredictor
predictor = SentimentPredictor(model_path="models/xlm-r-run1", device="cuda")
preds = predictor.predict(["I love this!", "No me gustó nada"], langs=["en","es"])
print(preds)  # -> list of labels/scores

CLI (if implemented):
python src/inference.py --model models/xlm-r-run1 --input-text "That was great!" --lang en

---

## Notebooks & experiments

- notebooks/ contains step-by-step demos for:
  - EDA: language distribution, class imbalance, token length distributions
  - Model introspection: tokenization examples, attention visualization (if added)
  - Quick baseline comparisons

Use notebooks for prototyping before moving to full training runs.

---

## Reproducibility

- Set seeds: PYTHONHASHSEED, numpy, torch, random
- Record the exact model checkpoint, tokenizer version, and config used for every experiment (store configs in experiments/ or use MLFlow/W&B).
- Consider Docker or a conda environment file for exact dependency pinning.
- For deterministic torch behavior:
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

---

## Contributing

Contributions are welcome. Suggested workflow:

1. Open an issue describing the feature/bug/experiment you want to add.
2. Create a branch: git checkout -b feat/<short-description>
3. Implement changes and add tests or a notebook demonstrating the change.
4. Submit a pull request with a clear description of changes and motivation.

Coding style:
- Python: follow PEP8
- Add docstrings for public functions and modules
- Include minimal reproducible examples for major changes

---

## Troubleshooting & FAQ

- Out of memory on GPU: reduce batch size, use gradient accumulation, or enable mixed precision (AMP).
- Slow training: ensure tokenization and data loading are optimized (use datasets with caching, set num_workers > 0).
- Model isn't learning: check label distribution, learning rate, tokenizer mismatch (e.g., using an English tokenizer on multilingual text), and overfitting signs.

---

## License

This project is provided under the MIT License. See the LICENSE file for full details. If a LICENSE file is not present, add one to make intentions explicit.

---

## Acknowledgements & citation

- Hugging Face Transformers and Datasets libraries for models and dataset utilities.
- Public datasets and shared task organizers for multilingual sentiment resources.
- If you use this work in research, please cite the repository and any base models you fine-tune (e.g., the XLM-R paper).

Suggested citation (adapt to your citation style):
- Vamshikrishan. Multiple_Languages_Sentemental_analysis. GitHub repository. https://github.com/Vamshikrishan/Multiple_Languages_Sentemental_analysis

---

## Contact

Maintainer: Vamshikrishan  
GitHub: https://github.com/Vamshikrishan

For questions, feature requests, or dataset contributions, open an issue on the repository.

---

Thank you for checking out this repository — whether you're reproducing experiments, benchmarking models, or extending support to new languages, this project is structured to make multilingual sentiment analysis more accessible and repeatable.
