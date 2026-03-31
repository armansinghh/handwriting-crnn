import torch
from src.models.crnn import CRNN

model = CRNN()

x = torch.randn(4, 1, 32, 128)

out = model(x)

print("Output shape:", out.shape)