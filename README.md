# Handwriting CRNN

A deep learning project for recognizing handwritten text using a **CRNN (Convolutional Recurrent Neural Network)**.

The goal is to build an end-to-end system that converts handwritten text images into machine-readable text. The final system will be capable of processing **entire handwritten pages**, detecting individual text lines, and recognizing them.

---

## Project Status

🚧 Early development stage

Current progress:

* Environment setup complete
* CUDA-enabled PyTorch configured
* IAM dataset parsing and preprocessing complete
* `dataset.csv` generated with image paths and labels

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
*"The IAM-database: an English sentence database for offline handwriting recognition"*
International Journal on Document Analysis and Recognition, 2002.

Dataset:
http://www.fki.inf.unibe.ch/databases/iam-handwriting-database

> The dataset is **not included in this repository** due to licensing restrictions.

> Note: The IAM Handwriting Database is restricted to **non-commercial research use**.
> This repository only contains code and does not redistribute the dataset.

### Expected Dataset Structure

```
data/
 └── raw/
      ├── words/
      └── words.txt
```

---

## Data Pipeline

The IAM words dataset is processed into a structured CSV format for training.

Steps:

1. Parse `words.txt`
2. Filter valid samples (`ok`)
3. Reconstruct image file paths
4. Validate image existence
5. Generate `dataset.csv`

### Output Format

```
image_path,label
data/raw/words/a01/a01-000u/a01-000u-00-00.png,A
```

This CSV serves as the input for the PyTorch Dataset class.

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

Verify that PyTorch and CUDA are correctly configured:

```
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output:

```
True
```

---

## Outputs

Processed dataset:

```
data/processed/dataset.csv
```

This file contains image paths and corresponding labels used for training.

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

### Recognition Model

```
CNN → BiLSTM → CTC Loss
```

* **CNN** extracts visual features from the image
* **BiLSTM** models sequential dependencies
* **CTC Loss** enables alignment-free sequence prediction

---

## Project Structure

```
handwriting-crnn/
│
├── data/                # Dataset directory (not included in repo)
│   ├── raw/             # Original IAM dataset (words/, words.txt)
│   └── processed/       # Generated dataset (dataset.csv)
│
├── src/                 
│   └── dataset/         # Dataset handling
│       └── iam_dataset.py
│
├── prepare_dataset.py   # Script to generate dataset.csv from words.txt
├── test_dataset.py      # Temporary script to verify dataset loading
│
├── requirements.txt
├── requirements-min.txt
├── LICENSE
└── README.md
```

```

---

## Roadmap

* Phase 0 — Environment Setup ✔
* Phase 1 — Dataset Preparation ✔
* Phase 2 — CRNN Model Implementation
* Phase 3 — Training Pipeline
* Phase 4 — Line Segmentation
* Phase 5 — Full Page OCR Pipeline

---

## Future Work

* Implement CRNN architecture
* Build PyTorch Dataset and DataLoader
* Train model using CTC loss
* Add decoding (greedy / beam search)
* Develop line segmentation pipeline
* Create demo interface for handwritten text recognition

---

## License

This project is licensed under the MIT License. See the **LICENSE** file for details.
