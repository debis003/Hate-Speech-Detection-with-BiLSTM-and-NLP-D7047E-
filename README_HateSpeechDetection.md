# Hate Speech Detection with RoBERTa and Adversarial Training

A transformer-based pipeline for detecting hate speech and offensive language in social media posts, built as a group project for the **Advanced Deep Learning (D7047E)** course at Luleå University of Technology.

## Overview

This project fine-tunes a **pretrained RoBERTa model** (`cardiffnlp/twitter-roberta-base`) for hate speech classification on the HASOC 2019 benchmark dataset. The system incorporates **adversarial training** via Fast Gradient Method (FGM) to improve robustness, and provides **explainability** through attention visualization — making predictions both accurate and human-interpretable.

## Architecture

```
Input Tweet → Text Cleaning → RoBERTa Tokenizer (max_len=128)
                                        ↓
              Pretrained RoBERTa (twitter-roberta-base, 58M tweets)
                                        ↓
                        [CLS] Token Representation
                                        ↓
                    Linear Classification Head → Hate / Non-Hate
                                        ↕
                    FGM Adversarial Perturbation (ε = 0.1)
                    Attention Heat-Maps for Explainability
```

## Dataset

- **HASOC 2019** — ~9,000 manually-labelled social media posts annotated for hate speech
- Real-world data with slang, sarcasm, emojis, and adversarial patterns
- Split: 70% train / 15% validation / 15% test with balanced label distribution

## Key Features

- **RoBERTa Fine-Tuning**: Leverages `cardiffnlp/twitter-roberta-base`, a transformer pretrained on 58M tweets — captures Twitter-specific language (slang, hashtags, emojis) better than generic BERT
- **Adversarial Training (FGM)**: Fast Gradient Method (ε = 0.1) perturbs word embeddings during training, improving robustness against obfuscated hate speech (e.g., "I h@te you")
- **Attention-Based Explainability**: Extracts self-attention weights to generate CLS token attention bar charts and token-to-token heatmaps that justify every prediction
- **NLP Preprocessing Pipeline**: Lowercasing, URL/mention stripping, emoji-to-text conversion, HTML entity replacement, contraction cleaning, and deduplication
- **Class Imbalance Handling**: Inverse class weights in the loss function ensure the minority (hate) class is not ignored

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 0.87 |
| F1 Score | 0.86 |
| Precision | 0.88 |
| Recall | 0.84 |

FGM adversarial training improved F1 by **3 percentage points** compared to the baseline model.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python |
| Deep Learning | PyTorch |
| Model | RoBERTa (cardiffnlp/twitter-roberta-base) |
| NLP | Hugging Face Transformers, AutoTokenizer |
| Adversarial Training | Fast Gradient Method (FGM) |
| Explainability | Attention visualization (bar charts + heatmaps) |
| Evaluation | F1-Score, Precision, Recall, Accuracy |
| Data Processing | Pandas, NumPy, scikit-learn, emoji |
| Visualization | Matplotlib |

## Training Details

- **Optimizer**: AdamW with linear warmup and decay schedule
- **Learning Rate**: 2e-5
- **Batch Size**: 16 with gradient accumulation
- **Epochs**: 5 with early stopping (patience = 2)
- **Adversarial ε**: 0.1

## Project Structure

```
├── src/                                 # Source code
│   ├── train.py                         # Training loop with FGM adversarial training
│   ├── model.py                         # RoBERTa model configuration
│   ├── preprocessing.py                 # Text cleaning and tokenization pipeline
│   └── evaluate.py                      # Evaluation and attention visualization
├── data/
│   ├── HASOC/                           # HASOC 2019 dataset
│   ├── OffenseEval/                     # OffenseEval dataset
│   └── OLID/                            # Preprocessed OLID dataset
├── results/
│   ├── training_history_*.png           # Training curves per subtask
│   ├── attn_heatmap_*.png              # Attention heatmap visualizations
│   └── attn_bar_*.png                  # CLS attention bar charts
├── presentation/                        # Group presentation slides
└── README.md
```

## How to Run

```bash
# Install dependencies
pip install torch transformers pandas numpy scikit-learn matplotlib emoji

# Train the model
python src/train.py --dataset hasoc --epochs 5 --fgm_epsilon 0.1

# Evaluate with attention visualization
python src/evaluate.py --model_path best_model.pt --visualize
```

## Results Visualizations

The project includes training history plots for HASOC subtasks 1–3, attention bar charts showing which tokens influenced predictions, and token-to-token heatmaps revealing contextual relationships learned by the model.

## Course Context

**Course**: D7047E — Advanced Deep Learning, Luleå University of Technology  
**Year**: 2025  
**Type**: Group project (Group 5)

## License

This project was developed for academic purposes at LTU.
