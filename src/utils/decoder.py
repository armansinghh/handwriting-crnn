import torch


class Decoder:
    def __init__(self, chars):
        self.chars = chars
        self.idx_to_char = {i + 1: c for i, c in enumerate(chars)}
        self.blank = 0

    def decode(self, outputs):
        # outputs: [T, B, C]
        preds = torch.argmax(outputs, dim=2)  # [T, B]

        preds = preds.permute(1, 0)  # [B, T]

        results = []

        for pred in preds:
            decoded = []
            prev = None

            for p in pred:
                p = p.item()

                # skip blanks and repeated chars
                if p != self.blank and p != prev:
                    decoded.append(self.idx_to_char.get(p, ""))

                prev = p

            results.append("".join(decoded))

        return results