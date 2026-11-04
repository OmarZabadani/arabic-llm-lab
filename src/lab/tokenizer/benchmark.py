import random
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from lab.tokenizer.fast_bpe import FastByteLevelBPE

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ARABIC_DATA_PATH = PROJECT_ROOT / "data" / "arabic_train_clean.txt"
ENGLISH_DATA_PATH = PROJECT_ROOT / "data" / "benchmark.txt"
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


def load_lines(path: Path) -> list[str]:
    """Load nonempty lines from a text file."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def contains_arabic(text: str) -> bool:
    """Check whether text contains Arabic characters."""
    return any("\u0600" <= char <= "\u06ff" for char in text)


def contains_english(text: str) -> bool:
    """Check whether text contains Latin letters."""
    return any(("a" <= char.lower() <= "z") for char in text)


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
    tokenizer: FastByteLevelBPE,
    lines: list[str],
) -> tuple[int, int]:
    """Evaluate the custom BPE tokenizer."""
    token_count = sum(len(tokenizer.encode(line)) for line in lines)
    word_count = sum(count_words(line) for line in lines)

    return token_count, word_count


def evaluate_pretrained(
    tokenizer: PreTrainedTokenizerBase,
    lines: list[str],
) -> tuple[int, int]:
    """Evaluate a pretrained tokenizer."""
    token_count = sum(
        len(tokenizer.encode(line, add_special_tokens=False)) for line in lines
    )
    word_count = sum(count_words(line) for line in lines)

    return token_count, word_count


def main() -> None:
    """Train and compare tokenizers on held-out Arabic and English text."""
    arabic_lines = load_lines(ARABIC_DATA_PATH)

    english_lines = [
        line
        for line in load_lines(ENGLISH_DATA_PATH)
        if contains_english(line) and not contains_arabic(line)
    ]

    if not english_lines:
        raise ValueError("No English-only sentences found in benchmark.txt.")

    arabic_train, arabic_test = split_language_data(arabic_lines)
    english_train, english_test = split_language_data(english_lines)

    train_lines = arabic_train + english_train

    test_sets = {
        "Arabic": arabic_test,
        "English": english_test,
    }

    print(f"Arabic articles: {len(arabic_lines):,}")
    print(f"English sentences: {len(english_lines):,}")
    print(f"Training sentences: {len(train_lines):,}")
    print(f"Testing sentences: {len(arabic_test) + len(english_test):,}")

    print("\nTraining custom Rust-backed BPE tokenizer...")

    custom_tokenizer = FastByteLevelBPE.train(
        train_lines,
        vocab_size=VOCAB_SIZE,
    )

    results: dict[str, dict[str, tuple[int, int]]] = {}

    for name, model_name in TOKENIZERS.items():
        print(f"\nLoading {name} tokenizer...")

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
        f"- Arabic articles: {len(arabic_lines):,}",
        f"- English sentences: {len(english_lines):,}",
        f"- Arabic training articles: {len(arabic_train):,}",
        f"- English training sentences: {len(english_train):,}",
        f"- Arabic testing articles: {len(arabic_test):,}",
        f"- English testing sentences: {len(english_test):,}",
        f"- Training split: {(1 - TEST_RATIO) * 100:.0f}%",
        f"- Testing split: {TEST_RATIO * 100:.0f}%",
        f"- Random seed: {RANDOM_SEED}",
        f"- Custom BPE vocabulary size: {custom_tokenizer.vocab_size}",
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
                f"| {tokenizer_name} | {language} | {word_count:,} "
                f"| {token_count:,} | {ratio:.3f} |"
            )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n".join(report))
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
