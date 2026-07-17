# LoRA and AdaLoRA Fine-Tuning of DistilBERT

This project evaluates parameter-efficient fine-tuning techniques for adapting DistilBERT to text classification on the AG News dataset.

The main goal was to compare full fine-tuning with LoRA and AdaLoRA and determine whether similar classification performance could be achieved while training far fewer model parameters.

## Project Overview

Fine-tuning an entire transformer model can require significant memory, training time, and computational resources.

LoRA addresses this by freezing the original model weights and introducing small trainable low-rank matrices into selected attention layers. AdaLoRA extends this idea by dynamically allocating the available rank budget based on the importance of different model parameters.

This project compares:

- Full fine-tuning
- LoRA with ranks 4, 8, 16, and 32
- LoRA applied to different attention projections
- AdaLoRA with adaptive rank allocation

## Dataset

The experiments use the AG News dataset, which contains news articles classified into four categories:

1. World
2. Sports
3. Business
4. Science and Technology

The training split contains approximately **120,000 samples**.

## Model and Tools

- DistilBERT
- PyTorch
- Hugging Face Transformers
- Hugging Face PEFT
- LoRA
- AdaLoRA
- AG News dataset
- Google Colab / Jupyter Notebook

## Experiment Results

| Method | Accuracy | Training Time | Trainable Parameters | GPU Memory |
|---|---:|---:|---:|---:|
| Full Fine-Tuning | 94.70% | 55.7 min | 66.9M (100%) | ~14 GB |
| LoRA r=4 (Q+V) | 91.07% | 14.7 min | 0.67M (0.99%) | 1.28 GB |
| LoRA r=8 (Q+V) | 91.61% | 14.3 min | 0.74M (1.09%) | 1.53 GB |
| LoRA r=16 (Q+V) | 91.97% | 14.4 min | 0.89M (1.31%) | 1.80 GB |
| LoRA r=32 (Q+V) | 92.41% | 14.3 min | 1.18M (1.74%) | 2.05 GB |
| LoRA r=32 (Q+K+V+O) | 92.74% | 21.2 min | 1.77M (2.58%) | — |
| LoRA r=32 (separate projections) | **93.49%** | 31.5 min | 1.77M (2.58%) | — |
| AdaLoRA | 91.20% | 51.9 min | 1.77M (2.58%) | — |

## Key Findings

The best LoRA configuration achieved **93.49% accuracy**, only **1.21 percentage points below full fine-tuning**.

It trained only **1.77 million parameters**, representing **2.58% of the complete model** and reducing the number of trainable parameters by approximately **97.4%**.

The best LoRA configuration also reduced training time from **55.7 minutes to 31.5 minutes**, an improvement of approximately **43%**.

Increasing the LoRA rank generally improved classification accuracy, while applying adapters to additional attention projections produced the strongest LoRA result.

AdaLoRA achieved 91.20% accuracy on this task. Its adaptive rank allocation did not outperform the best fixed-rank LoRA configuration for DistilBERT and AG News classification.

## Main Takeaway

LoRA achieved near-baseline classification accuracy while training only a small fraction of DistilBERT's parameters.

The results demonstrate the trade-off between model accuracy and computational efficiency:

- Full fine-tuning produced the highest accuracy.
- LoRA significantly reduced trainable parameters, GPU memory usage, and training time.
- The best LoRA configuration retained most of the full fine-tuning performance with substantially lower computational cost.

## Repository Contents

```text
.
├── LoRA_implementation.ipynb
└── README.md
