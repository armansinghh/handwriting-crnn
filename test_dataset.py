from src.dataset.iam_dataset import IAMDataset
from torch.utils.data import DataLoader
from src.dataset.collate import collate_fn

dataset = IAMDataset("data/processed/dataset.csv")

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)

images, labels, lengths = next(iter(dataloader))

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Lengths:", lengths)
print("Labels:", labels)