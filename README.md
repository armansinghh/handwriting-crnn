# Handwriting CRNN

A deep learning project for recognizing handwritten text using a **CRNN (Convolutional Recurrent Neural Network)**.

The goal is to build a system that converts handwritten text images into machine-readable text. Eventually, the model should be capable of processing **entire handwritten pages**, detecting individual text lines, and recognizing them.

---

## Project Status

🚧 Early development stage

Current progress:

* Environment setup complete
* CUDA-enabled PyTorch configured
* Dataset analysis phase starting

---

## Requirements

* Python 3.12
* PyTorch
* CUDA (optional but recommended for training)
* NVIDIA GPU (tested on RTX 4050)

---

## Dataset

This project uses the **IAM Handwriting Database**.

U. Marti and H. Bunke
*“The IAM-database: an English sentence database for offline handwriting recognition”*
International Journal on Document Analysis and Recognition, 2002.

Dataset:
http://www.fki.inf.unibe.ch/databases/iam-handwriting-database

> The dataset is **not included in this repository** due to licensing restrictions.

> Note: The IAM Handwriting Database is restricted to **non-commercial research use**.
> This repository only contains code and does not redistribute the dataset.

Expected dataset structure:

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

Upgrade pip:

```
python -m pip install --upgrade pip
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Quick Start

After installing dependencies, verify that PyTorch and CUDA are correctly configured:

```
python -c "import torch; print(torch.cuda.is_available())"
```

If CUDA is configured correctly, this should return:

```
True
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

The recognition model architecture:

```
CNN → BiLSTM → CTC Decoder
```

This structure allows the model to convert an image of handwritten text into a sequence of characters.

---

## Project Structure

```
handwriting-crnn/
│
├── data/                # Dataset directory (not included in repo)
│   ├── raw/
│   └── processed/
│
├── src/                 # Core model and training code
│
├── notebooks/           # Experiments and dataset analysis
│
├── requirements.txt
├── requirements-min.txt
├── LICENSE
└── README.md
```

---

## Roadmap

Phase 0 — Environment Setup ✔

Phase 1 — Dataset Analysis

Phase 2 — Line-level CRNN Training

Phase 3 — Line Segmentation

Phase 4 — Full Page OCR Pipeline

---

## Future Work

* Implement line-level CRNN training
* Build text line segmentation pipeline
* Combine segmentation + recognition for full-page OCR
* Add demo interface for handwritten text recognition

---

## License

This project is licensed under the MIT License. See the **LICENSE** file for details.
