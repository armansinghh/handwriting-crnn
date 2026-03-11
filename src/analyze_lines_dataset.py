from collections import Counter

LINES_FILE = "data/raw/IAM/ascii/lines.txt"

vocab = set()
line_lengths = []
char_counter = Counter()

with open(LINES_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("#"):
            continue

        parts = line.strip().split()
        if len(parts) < 9:
            continue

        transcription = " ".join(parts[8:])

        line_lengths.append(len(transcription))
        vocab.update(transcription)
        char_counter.update(transcription)

print("Total lines:", len(line_lengths))
print("Vocabulary size:", len(vocab))
print("Characters:", sorted(vocab))
print("Longest line length:", max(line_lengths))
print("Average line length:", sum(line_lengths)/len(line_lengths))