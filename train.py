import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset.iam_dataset import IAMDataset
from src.dataset.collate import collate_fn
from src.models.crnn import CRNN


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset
dataset = IAMDataset("data/processed/dataset.csv")

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn
)


# Model
model = CRNN().to(device)


# Loss (CTC)
criterion = nn.CTCLoss(blank=0)


# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 🔁 One training step (just test)
model.train()

images, labels, label_lengths = next(iter(dataloader))

images = images.to(device)
labels = labels.to(device)
label_lengths = label_lengths.to(device)

# Forward
outputs = model(images)   # [T, B, C]

# Convert to log probs (VERY IMPORTANT)
log_probs = torch.log_softmax(outputs, dim=2)

from src.utils.decoder import Decoder

chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
decoder = Decoder(chars)

pred_texts = decoder.decode(outputs)
print("Predictions:", pred_texts)

T, B, _ = log_probs.size()

# Input lengths (all same)
input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long).to(device)

# Flatten labels (CTC requirement)
labels_list = []

for i in range(labels.size(0)):  # batch size
    length = label_lengths[i]
    labels_list.append(labels[i, :length])

labels_flat = torch.cat(labels_list)

# Compute loss
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

print("Loss:", loss.item())