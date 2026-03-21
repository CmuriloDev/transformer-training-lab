# Transformer Training Lab (End-to-End)

## Overview

This project implements a complete Transformer pipeline, including training and inference, based on previous laboratory activities.

The goal is to connect all previously developed components (attention, encoder, decoder) and demonstrate that the model is capable of learning patterns from real data by reducing the loss during training.

This is an academic project focused on understanding the architecture rather than achieving high-quality translations.

---

## Academic Information

Academic project for the course Artificial Intelligence Topics
Professor: Dimmy Magalhães
Institution: Faculdade iCEV

---

## Objectives

* Integrate a real dataset from Hugging Face
* Apply tokenization to convert text into numerical representations
* Implement a full training loop (forward, loss, backward, optimizer step)
* Monitor loss reduction across epochs
* Implement auto-regressive generation
* Perform an overfitting test

---

## Technologies Used

* Python 3.x
* PyTorch
* Hugging Face Datasets
* Hugging Face Transformers

---

## Project Structure

```
transformer-training-lab
│
├── dataset.py
├── tokenizer_utils.py
├── model.py
├── train.py
├── inference.py
└── main.py
```

---

## How to Run

1. Install dependencies:

```
pip install torch datasets transformers
```

2. Run the project:

```
python main.py
```

---

## Example Output

```
TRANSFORMER TRAINING LAB

Epoch 1, Loss: 8.70
Epoch 10, Loss: 6.11
Epoch 20, Loss: 3.54

Generated: [101, 2210, 898, ...]

Overfitting test:
Expected: [...]
Generated: [...]
```

The important aspect is that the **loss decreases consistently**, demonstrating that the model is learning.

---

## Concepts Demonstrated

* Transformer Encoder-Decoder architecture
* Tokenization and padding
* Teacher Forcing
* Cross-Entropy Loss
* Adam optimizer
* Backpropagation
* Auto-regressive sequence generation

---

## Integration with Previous Labs

This project integrates concepts and structures developed in previous labs:

* Scaled Dot-Product Attention
* Encoder structure with Add & Norm
* Decoder with Masked Self-Attention and Cross-Attention
* Auto-regressive generation loop

These components were adapted to work within a training pipeline using PyTorch.

---

## AI-Assisted Complementary Support

AI-based tools were used as a complementary resource in specific parts of the project, mainly for:

* Assisting with dataset integration using Hugging Face
* Supporting tokenization adjustments and padding strategies
* Debugging tensor shape inconsistencies during training
* Clarifying the implementation of teacher forcing in the training loop

All suggested adjustments were carefully reviewed, tested, and adapted manually.

Final review and validation:

Carlos Murilo Nogueira Portela

---

## Version

v1.0
