# from src.dataset.iam_dataset import IAMDataset

# dataset = IAMDataset("data/processed/dataset.csv")

# print("Dataset size:", len(dataset))

# img, label = dataset[0]

# print("Label:", label)
# print("Shape:", img.shape)

from src.dataset.iam_dataset import IAMDataset
from torch.utils.data import DataLoader

dataset = IAMDataset("data/processed/dataset.csv")

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

# Get one batch
images, labels = next(iter(dataloader))

print("Batch image shape:", images.shape)
print("Labels:", labels)