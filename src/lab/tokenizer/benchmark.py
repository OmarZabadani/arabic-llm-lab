import random
from pathlib import Path

from transformers import AutoTokenizer

from lab.tokenizer.bpe import ByteLevelBPE

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "benchmark.txt"
REPORT_PATH = PROJECT_ROOT / "reports" / "tokenizer_benchmark.md"

VOCAB_SIZE = 512
TEST_RATIO = 0.2
RANDOM_SEED = 42

TOKENIZERS = {
    "GPT-2": "gpt2",
    "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
}


def count_words(text: str) -> int:
    """Count whitespace-separated words."""
    return len(text.split())


def tokens_per_word(token_count: int, word_count: int) -> float:
    """Calculate the average number of tokens per word."""
    if word_count == 0:
        return 0.0

    return token_count / word_count


def split_language_data(
    lines: list[str],
) -> tuple[list[str], list[str]]:
    """Split sentences into training and testing sets."""
    if len(lines) < 2:
        raise ValueError("Each language needs at least two sentences.")

    shuffled = lines.copy()
    random.Random(RANDOM_SEED).shuffle(shuffled)

    test_size = max(1, round(len(shuffled) * TEST_RATIO))

    return shuffled[test_size:], shuffled[:test_size]


def evaluate_custom(
    tokenizer: ByteLevelBPE,
    lines: list[str],
) -> tuple[int, int]:
    """Evaluate the custom BPE tokenizer."""
    token_count = sum(len(tokenizer.encode(line)) for line in lines)
    word_count = sum(count_words(line) for line in lines)

    return token_count, word_count


def evaluate_pretrained(
    tokenizer: object,
    lines: list[str],
) -> tuple[int, int]:
    """Evaluate a pretrained tokenizer."""
    token_count = sum(
        len(tokenizer.encode(line, add_special_tokens=False))  # type: ignore[attr-defined]
        for line in lines
    )
    word_count = sum(count_words(line) for line in lines)

    return token_count, word_count


def main() -> None:
    """Train on one split and evaluate on unseen sentences."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Benchmark dataset not found: {DATA_PATH}")

    lines = [
        line.strip()
        for line in DATA_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    arabic_lines = [
        line for line in lines if any("\u0600" <= char <= "\u06ff" for char in line)
    ]
    english_lines = [line for line in lines if line not in arabic_lines]

    if not arabic_lines or not english_lines:
        raise ValueError(
            "benchmark.txt must contain both Arabic and English sentences."
        )

    arabic_train, arabic_test = split_language_data(arabic_lines)
    english_train, english_test = split_language_data(english_lines)

    train_lines = arabic_train + english_train

    test_sets = {
        "Arabic": arabic_test,
        "English": english_test,
    }

    print(f"Training sentences: {len(train_lines)}")
    print(f"Testing sentences: {len(arabic_test) + len(english_test)}")

    print("\nTraining custom BPE tokenizer...")
    custom_tokenizer = ByteLevelBPE.train(
        train_lines,
        vocab_size=VOCAB_SIZE,
    )

    results: dict[str, dict[str, tuple[int, int]]] = {}

    for name, model_name in TOKENIZERS.items():
        print(f"Loading {name} tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        results[name] = {
            language: evaluate_pretrained(tokenizer, language_lines)
            for language, language_lines in test_sets.items()
        }

    results["Custom BPE"] = {
        language: evaluate_custom(custom_tokenizer, language_lines)
        for language, language_lines in test_sets.items()
    }

    report = [
        "# Held-Out Tokenizer Benchmark",
        "",
        f"- Total sentences: {len(lines)}",
        f"- Training sentences: {len(train_lines)}",
        f"- Testing sentences: {len(arabic_test) + len(english_test)}",
        f"- Training split: {(1 - TEST_RATIO) * 100:.0f}%",
        f"- Testing split: {TEST_RATIO * 100:.0f}%",
        f"- Random seed: {RANDOM_SEED}",
        f"- Custom BPE vocabulary size: {len(custom_tokenizer.vocab)}",
        "",
        "## Evaluation Results",
        "",
        "| Tokenizer | Language | Words | Tokens | Tokens per word |",
        "|---|---|---:|---:|---:|",
    ]

    for tokenizer_name, language_results in results.items():
        for language, (token_count, word_count) in language_results.items():
            ratio = tokens_per_word(token_count, word_count)

            report.append(
                f"| {tokenizer_name} | {language} | {word_count} "
                f"| {token_count} | {ratio:.3f} |"
            )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n".join(report))
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
