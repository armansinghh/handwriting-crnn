import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset.iam_dataset import IAMDataset
from src.dataset.collate import collate_fn
from src.models.crnn import CRNN
from src.utils.decoder import Decoder


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset
dataset = IAMDataset("data/processed/dataset.csv")

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn
)


# Model
model = CRNN().to(device)


# Loss (CTC)
criterion = nn.CTCLoss(blank=0)


# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)


# Decoder
chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
decoder = Decoder(chars)


# 🚀 Training loop
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_idx, (images, labels, label_lengths) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        # Forward
        outputs = model(images)   # [T, B, C]
        log_probs = torch.log_softmax(outputs, dim=2)

        T, B, _ = log_probs.size()

        # Input lengths (all same)
        input_lengths = torch.full((B,), T, dtype=torch.long).to(device)

        # Flatten labels (remove padding)
        labels_list = []
        for i in range(labels.size(0)):
            length = label_lengths[i]
            labels_list.append(labels[i, :length])

        labels_flat = torch.cat(labels_list)

        # Loss
        loss = criterion(
            log_probs,
            labels_flat,
            input_lengths,
            label_lengths
        )

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Print progress
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

    # Epoch summary
    avg_loss = total_loss / len(dataloader)
    print(f"\nEpoch {epoch+1} completed | Avg Loss: {avg_loss:.4f}")

    # 🔽 Decode sample predictions (from last batch)
    pred_texts = decoder.decode(outputs)
    print("Sample Predictions:", pred_texts[:5])
    print("-" * 50)