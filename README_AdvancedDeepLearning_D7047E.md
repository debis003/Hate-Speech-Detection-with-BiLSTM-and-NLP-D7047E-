# Advanced Deep Learning — D7047E

Course projects and labs from **Advanced Deep Learning (D7047E)** at Luleå University of Technology, covering transformer fine-tuning, adversarial training, image captioning, and generative adversarial networks.

---

## Project: Hate Speech Detection with RoBERTa and Adversarial Training
**Group Project (Group 5)**

Fine-tuned a pretrained **RoBERTa transformer** (`cardiffnlp/twitter-roberta-base`) for hate speech classification on the HASOC 2019 benchmark dataset. The system incorporates adversarial training via Fast Gradient Method (FGM) and provides explainability through attention visualization.

### Architecture

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

### Key Details

- **Model**: `cardiffnlp/twitter-roberta-base` — RoBERTa pretrained on 58M tweets, loaded via `AutoModelForSequenceClassification`
- **Dataset**: HASOC 2019 (~9,000 manually-labelled social media posts) — split 70/15/15
- **Adversarial Training**: Fast Gradient Method (ε = 0.1) perturbs word embeddings during training, improving robustness against obfuscated hate speech
- **Explainability**: Self-attention weights extracted for CLS token bar charts and token-to-token heatmaps
- **Preprocessing**: Lowercasing, URL/mention stripping, emoji-to-text conversion, HTML entity replacement, contraction cleaning, deduplication
- **Training**: AdamW optimizer, lr=2e-5 with linear warmup, batch size 16 with gradient accumulation, early stopping (patience=2)
- **Class Imbalance**: Inverse class weights in CrossEntropyLoss

### Results

| Metric | Score |
|--------|-------|
| Accuracy | 0.87 |
| F1 Score | 0.86 |
| Precision | 0.88 |
| Recall | 0.84 |

FGM adversarial training improved F1 by **3 percentage points** vs. baseline.

### Tech Stack

Python, PyTorch, Hugging Face Transformers, Pandas, NumPy, Matplotlib, scikit-learn, emoji

---

## Lab 3: Image Captioning with Encoder-Decoder Networks

Built an image captioning system using a **CNN encoder (ResNet)** paired with an **LSTM decoder with attention** to generate natural language descriptions of images.

### Architecture

```
Image → CNN Encoder (ResNet) → Feature Vector (2048-dim) → LSTM Decoder + Attention → Caption
```

### Key Details

- **Encoder**: Pre-trained ResNet extracts spatial feature maps from input images
- **Decoder**: LSTM-based sequence generator with attention mechanism — focuses on relevant image regions at each decoding step
- **Training**: Teacher forcing for faster convergence
- **Inference**: Beam search for higher-quality caption generation
- **Transfer Learning**: Leverages ImageNet-trained CNN features

### Tech Stack

Python, PyTorch, ResNet, LSTM, Attention Mechanism

---

## Lab 2: GAN Implementation — Image Generation & Adversarial Attacks

Explored two applications of adversarial machine learning: training a GAN to generate realistic handwritten digits, and crafting gradient-based adversarial attacks that fool classifiers.

### Architecture

**GAN for Image Generation:**
```
Random Noise (z) → Generator → Fake Image
                                    ↓
Real Image ──────→ Discriminator → Real/Fake?
                                    ↓
                        Adversarial Training Loop
```

**Adversarial Image Attacks:**
```
Original Image → Add Perturbation → Adversarial Image → Classifier → Wrong Prediction
                      ↑                                      ↓
                      └──────── Gradient-based Attack ────────┘
```

### Key Details

- **GAN Training**: Alternating optimization of generator and discriminator networks on MNIST
- **Adversarial Examples**: Demonstrates neural network vulnerability to carefully crafted, human-imperceptible input perturbations
- **Results**: GAN generates convincing handwritten digits; small perturbations cause high-confidence misclassifications

### Tech Stack

Python, PyTorch, GANs, MNIST, Matplotlib

---

## Repository Structure

```
├── project/                                    # Hate Speech Detection (Group Project)
│   ├── src/
│   │   ├── train.py                            # Training loop with FGM adversarial training
│   │   ├── model.py                            # RoBERTa model configuration
│   │   ├── preprocessing.py                    # Text cleaning and tokenization
│   │   └── evaluate.py                         # Evaluation and attention visualization
│   ├── data/
│   │   └── HASOC/                              # HASOC 2019 dataset
│   └── results/
│       ├── training_history_*.png              # Training curves
│       ├── attn_heatmap_*.png                  # Attention heatmaps
│       └── attn_bar_*.png                      # CLS attention bar charts
│
├── lab3_image_captioning/                      # Image Captioning
│   ├── task1_task2_image_captioning.ipynb
│   └── Lab_3_Task_3.1_Report.pdf
│
├── lab2_gan/                                   # GAN & Adversarial Attacks
│   ├── GAN.py
│   ├── MNIST-adversarial-images.ipynb
│   └── Lab_2_Task_4.ipynb
│
└── README.md
```

## How to Run

```bash
# Install dependencies
pip install torch torchvision transformers pandas numpy scikit-learn matplotlib emoji

# --- Hate Speech Detection ---
python project/src/train.py --dataset hasoc --epochs 5 --fgm_epsilon 0.1
python project/src/evaluate.py --model_path best_model.pt --visualize

# --- Image Captioning ---
jupyter notebook lab3_image_captioning/task1_task2_image_captioning.ipynb

# --- GAN ---
python lab2_gan/GAN.py
jupyter notebook lab2_gan/MNIST-adversarial-images.ipynb
```

## Course Context

**Course**: D7047E — Advanced Deep Learning, Luleå University of Technology  
**Year**: 2025  
**Components**: 1 group project + 2 labs

## License

This project was developed for academic purposes at LTU.
