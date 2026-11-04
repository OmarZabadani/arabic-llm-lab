from pathlib import Path

SOURCE = Path("data") / "arabic_train.txt"
DESTINATION = Path("data") / "arabic_train_clean.txt"


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Source corpus not found: {SOURCE.resolve()}")

    seen: set[str] = set()
    original_count = 0
    unique_count = 0

    with SOURCE.open("r", encoding="utf-8") as source:
        with DESTINATION.open("w", encoding="utf-8") as destination:
            for line in source:
                original_count += 1
                text = line.strip()

                if not text or text in seen:
                    continue

                seen.add(text)
                destination.write(text + "\n")
                unique_count += 1

    print("\nCorpus cleaning completed successfully.")
    print("-" * 40)
    print(f"Original articles : {original_count:,}")
    print(f"Unique articles   : {unique_count:,}")
    print(f"Duplicates removed: {original_count - unique_count:,}")
    print(f"Output file       : {DESTINATION.resolve()}")
    print(f"Output size       : {DESTINATION.stat().st_size / (1024**2):.2f} MB")


if __name__ == "__main__":
    main()
