from pathlib import Path

SOURCE = Path("arabic-llm-lab/data/arabic_train.txt")
DESTINATION = Path("D:\download\job\ACES\arabic-llm-lab\data\arabic_train_clean.txt")


def main() -> None:
    seen: set[str] = set()
    original_count = 0
    unique_count = 0

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)

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

    print(f"Original articles: {original_count:,}")
    print(f"Unique articles: {unique_count:,}")
    print(f"Duplicates removed: {original_count - unique_count:,}")
    print(f"Saved to: {DESTINATION}")


if __name__ == "__main__":
    main()
