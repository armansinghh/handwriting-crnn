# Handwriting CRNN

A deep learning project for recognizing handwritten text using a **CRNN (Convolutional Recurrent Neural Network)**.

The goal is to build a system that can take an image of handwritten text and convert it into machine-readable text.

Eventually, the model should be able to process **entire handwritten pages**, detect individual text lines, and recognize them.

---

## Project Status

🚧 Early development stage

Current progress:

* Environment setup complete
* CUDA-enabled PyTorch configured
* Dataset analysis phase starting

---

## Dataset

This project uses the **IAM Handwriting Database**.

U. Marti and H. Bunke
*“The IAM-database: an English sentence database for offline handwriting recognition”*
International Journal on Document Analysis and Recognition, 2002.

Dataset:
http://www.fki.inf.unibe.ch/databases/iam-handwriting-database

> The dataset is **not included in this repository** due to licensing restrictions.

Expected structure:

```
data/
 └── raw/
      └── IAM/
           ├── words/
           └── ascii/
                ├── words.txt
                └── lines.txt
```

---

## Setup

Create a virtual environment:

```
py -3.12 -m venv venv
```

Activate it:

```
venv\Scripts\Activate.ps1
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Planned Architecture

The final OCR pipeline will follow this structure:

```
Page Image
   ↓
Line Segmentation
   ↓
CRNN Recognition
   ↓
Text Output
```

The recognition model will be based on:

```
CNN → BiLSTM → CTC Decoder
```

---

## Project Structure

```
handwriting-crnn
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│
├── notebooks/
│
├── requirements.txt
├── requirements-min.txt
└── README.md
```

---

## Roadmap

Phase 0 — Environment Setup ✔
Phase 1 — Dataset Analysis
Phase 2 — Line-level CRNN Training
Phase 3 — Line Segmentation
Phase 4 — Full Page OCR Pipeline
