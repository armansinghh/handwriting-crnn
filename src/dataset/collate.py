import torch


def collate_fn(batch):
    images, labels = zip(*batch)

    # Stack images (already same size)
    images = torch.stack(images, 0)

    # Get label lengths
    label_lengths = [len(label) for label in labels]

    # Find max length in batch
    max_length = max(label_lengths)

    # Pad labels
    padded_labels = []
    for label in labels:
        padded = label + [0] * (max_length - len(label))  # 0 = CTC blank
        padded_labels.append(padded)

    padded_labels = torch.tensor(padded_labels, dtype=torch.long)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)

    return images, padded_labels, label_lengths