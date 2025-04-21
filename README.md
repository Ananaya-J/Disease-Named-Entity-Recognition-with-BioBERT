# 🧬 Biomedical Named Entity Recognition (BioNER) for Diseases using BioBERT

A streamlined BioNER pipeline powered by **BioBERT (`dmis-lab/biobert-base-cased-v1.1`)**, designed specifically to detect **disease entities** in biomedical literature. This model leverages multiple curated datasets to improve robustness and generalization.

---

## 🔍 Overview

This project implements a **Disease-focused Named Entity Recognition (NER)** system, trained on high-quality biomedical corpora. The system identifies and classifies **Disease entities (`DIS`)** using the BioBERT architecture fine-tuned on:

- 🧫 **NCBI Disease**
- 💊 **BC5CDR**
- 📚 **BioNLP 2013 CG**

---

## ✨ Features

- ✅ **Unified Label Schema**: Simplified to focus solely on disease mentions (`B-DIS`, `I-DIS`)
- 🔁 **Multi-Dataset Integration**: Harmonizes multiple datasets with varying structures
- 🧠 **Entity-Aware Tokenization**: Ensures accurate alignment of tokens and entity spans
- ⚖️ **Support for Class Balancing**: Optional inverse-frequency loss weighting
- 📊 **Entity-Level Evaluation**: Precision, recall, and F1 for `DIS` entity class
- 💾 **Persistence**: Saves tokenized datasets for faster reloading

---

## 📦 Requirements

- Python 3.8+
- PyTorch
- Huggingface Transformers
- Datasets
- Evaluate
- Numpy

### Install dependencies:

```bash
pip install torch transformers datasets evaluate numpy
