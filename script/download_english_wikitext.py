from pathlib import Path

from datasets import load_dataset

DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"

OUTPUT_DIR = Path("data/english/wikitext-103")

SPLITS = ("train", "validation", "test")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        print(f"Downloading {split} split...")

        dataset = load_dataset(
            DATASET_NAME,
            DATASET_CONFIG,
            split=split,
        )

        output_path = OUTPUT_DIR / f"{split}.txt"

        record_count = 0
        word_count = 0
        byte_count = 0

        with output_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as file:
            for row in dataset:
                text = row["text"].strip()

                if not text:
                    continue

                line = text + "\n"
                file.write(line)

                record_count += 1
                word_count += len(text.split())
                byte_count += len(line.encode("utf-8"))

        print(f"Saved: {output_path}")
        print(f"Records: {record_count:,}")
        print(f"Words: {word_count:,}")
        print(f"UTF-8 bytes: {byte_count:,}")
        print("-" * 40)

    print("English dataset download completed.")


if __name__ == "__main__":
    main()
