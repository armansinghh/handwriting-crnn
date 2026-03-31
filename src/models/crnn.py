import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),   # [B, 64, 32, 128]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # [B, 64, 16, 64]

            nn.Conv2d(64, 128, 3, padding=1), # [B, 128, 16, 64]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # [B, 128, 8, 32]

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),             # [B, 256, 4, 32]

            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),             # [B, 512, 2, 32]

            nn.Conv2d(512, 512, 2),           # [B, 512, 1, 31]
            nn.ReLU()
        )

    def forward(self, x):
        features = self.cnn(x)
        # Remove height dimension
        features = features.squeeze(2)   # [B, 512, 31]

        # Permute to [W, B, C]
        features = features.permute(2, 0, 1)  # [31, B, 512]

        return features