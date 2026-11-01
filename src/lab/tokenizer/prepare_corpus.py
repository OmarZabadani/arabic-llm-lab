from pathlib import Path

from datasets import load_dataset

SAMPLE_SIZE = 50_000
OUTPUT_PATH = Path("data/arabic_train.txt")


def main() -> None:
    dataset = load_dataset(
        "Jr23xd23/ArabicText-Large",
        split="train",
        streaming=True,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    count = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for row in dataset:
            text = row.get("text", "").strip()

            if not text:
                continue

            file.write(text.replace("\n", " ") + "\n")
            count += 1

            if count >= SAMPLE_SIZE:
                break

    print(f"Saved {count:,} articles to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
