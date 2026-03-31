import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class IAMDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        # Character set (simple for now)
        self.chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        self.char_to_idx = {char: i + 1 for i, char in enumerate(self.chars)}  # 0 reserved for CTC blank

        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((32, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def encode_label(self, label):
        return [self.char_to_idx[c] for c in label if c in self.char_to_idx]

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]["image_path"]
        label = self.data.iloc[idx]["label"]

        # Load image
        while True:
             try:
                image = Image.open(image_path).convert("L")
                break
             except Exception:
                idx = (idx + 1) % len(self)
                image_path = self.data.iloc[idx]["image_path"]
                label = self.data.iloc[idx]["label"]
        image = self.transform(image)

        # Encode label
        encoded_label = self.encode_label(label)
        # handle empty encoded labels by skipping invalid samples
        if len(encoded_label) == 0:
            return self.__getitem__((idx + 1) % len(self))

        return image, encoded_label