import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),   # [B, 64, 32, 128]
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # [B, 64, 16, 64]

            nn.Conv2d(64, 128, 3, padding=1), # [B, 128, 16, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # [B, 128, 8, 32]

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),             # [B, 256, 4, 32]

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),             # [B, 512, 2, 32]

            nn.Conv2d(512, 512, 2),           # [B, 512, 1, 31]
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # RNN (BiLSTM)
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            dropout=0.0
        )

        # Final classification layer
        self.fc = nn.Linear(256, 63)  # 62 chars + 1 blank

    def forward(self, x):
        # CNN feature extraction
        features = self.cnn(x)        # [B, 512, 1, 31]

        # Remove height dimension
        features = features.squeeze(2)   # [B, 512, 31]

        # Convert to sequence
        features = features.permute(2, 0, 1)  # [31, B, 512]

        # RNN
        sequence, _ = self.rnn(features)      # [31, B, 512]

        # Classification
        output = self.fc(sequence)            # [31, B, 63]

        return output