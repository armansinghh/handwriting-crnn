import os
import csv

WORDS_FILE = "data/raw/words.txt"
WORDS_DIR = "data/raw/words"
OUTPUT_FILE = "data/processed/dataset.csv"


def main():
    data = []

    # Check if file exists
    if not os.path.exists(WORDS_FILE):
        print(f"ERROR: words.txt not found at {WORDS_FILE}")
        return

    with open(WORDS_FILE, "r") as f:
        lines = f.readlines()

    print(f"Total lines in words.txt: {len(lines)}")

    for line in lines:
        if line.startswith("#"):
            continue

        parts = line.strip().split()

        if len(parts) < 2:
            continue

        if parts[1] != "ok":
            continue

        image_id = parts[0]
        label = parts[-1]

        parts_id = image_id.split("-")

        if len(parts_id) < 2:
            continue

        folder1 = parts_id[0]
        folder2 = parts_id[0] + "-" + parts_id[1]

        image_path = os.path.join(
            WORDS_DIR,
            folder1,
            folder2,
            image_id + ".png"
        )

        if not os.path.exists(image_path):
            continue

        data.append((image_path, label))

    print(f"Collected samples: {len(data)}")

    # Create processed folder if needed
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Write CSV
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])

        for item in data:
            writer.writerow(item)

    print(f"Dataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()