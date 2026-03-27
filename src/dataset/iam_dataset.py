import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class IAMDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        # Transform pipeline
        self.transform = transforms.Compose([
            transforms.Resize((32, 128)),   # (height, width)
            transforms.ToTensor()           # converts to tensor [0,1]
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]["image_path"]
        label = self.data.iloc[idx]["label"]

        image = Image.open(image_path).convert("L")

        image = self.transform(image)

        return image, label