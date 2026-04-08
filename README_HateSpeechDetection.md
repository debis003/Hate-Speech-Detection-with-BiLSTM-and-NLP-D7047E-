# Hate Speech Detection with BiLSTM and NLP

A deep learning pipeline for detecting hate speech and offensive language in text, built as a group project for the **Advanced Deep Learning (D7047E)** course at Luleå University of Technology.

## Overview

This project implements a **Bidirectional LSTM (BiLSTM)** classifier for hate speech detection, trained and evaluated on multiple benchmark datasets. The pipeline includes text preprocessing, word embeddings, model training with K-Fold cross-validation, and hyperparameter optimization.

## Architecture

```
Input Text → Preprocessing → Word Embeddings → BiLSTM Layers → Dense → Sigmoid → Prediction
                                                    ↕
                                          K-Fold Cross-Validation
                                          Hyperparameter Tuning
```

## Datasets

- **HASOC** (Hate Speech and Offensive Content Identification) — Multi-class hate speech dataset
- **OffenseEval** — Offensive language identification from SemEval shared task
- **OLID** (Offensive Language Identification Dataset) — Preprocessed variant with hierarchical annotation

## Key Features

- **BiLSTM Architecture**: Bidirectional LSTM captures context from both directions in text sequences
- **K-Fold Cross-Validation**: Robust evaluation using K-Fold splits to prevent overfitting and ensure generalizability
- **Hyperparameter Optimization**: Systematic search over learning rates, hidden dimensions, dropout rates, and batch sizes
- **NLP Preprocessing Pipeline**: Tokenization, lowercasing, stopword removal, and text normalization
- **Multi-Dataset Evaluation**: Model tested across HASOC, OffenseEval, and OLID for cross-domain robustness

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python |
| Deep Learning | PyTorch |
| Model | BiLSTM |
| NLP | Custom preprocessing pipeline |
| Evaluation | K-Fold CV, F1-Score, Precision, Recall |
| Data | HASOC, OffenseEval, OLID |

## Project Structure

```
├── Pedro_BiLSTM.py                  # Core BiLSTM model implementation
├── Pedro_BiLSTM_KFold_HParam.py     # K-Fold CV + hyperparameter tuning
├── Project_NLP_ADL.pdf              # Full project report
├── Project pipeline.pdf             # Methodology pipeline
├── HASOCData/                       # HASOC dataset
├── OffenseEval/                     # OffenseEval dataset
├── OlidPreprcessed/                 # Preprocessed OLID dataset
└── Presentation/                    # Group presentation slides
```

## How to Run

```bash
# Install dependencies
pip install torch numpy pandas scikit-learn

# Train the BiLSTM model
python Pedro_BiLSTM.py

# Run with K-Fold cross-validation and hyperparameter tuning
python Pedro_BiLSTM_KFold_HParam.py
```

## Results

The BiLSTM model achieves competitive performance on hate speech detection benchmarks, evaluated using F1-score as the primary metric to handle class imbalance inherent in hate speech datasets.

## Course Context

**Course**: D7047E — Advanced Deep Learning, Luleå University of Technology  
**Year**: 2025  
**Type**: Group project (Group 5)

## License

This project was developed for academic purposes at LTU.
