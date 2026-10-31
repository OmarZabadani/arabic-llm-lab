from pathlib import Path

from transformers import AutoTokenizer

from lab.tokenizer.bpe import ByteLevelBPE

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "benchmark.txt"
REPORT_PATH = PROJECT_ROOT / "reports" / "tokenizer_benchmark.md"

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


def main() -> None:
    """Compare tokenization efficiency for Arabic and English."""
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

    custom_tokenizer = ByteLevelBPE.train(lines, vocab_size=512)

    results: dict[str, dict[str, tuple[int, int]]] = {}

    for name, model_name in TOKENIZERS.items():
        print(f"Loading {name} tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        for language, language_lines in (
            ("Arabic", arabic_lines),
            ("English", english_lines),
        ):
            token_count = sum(
                len(tokenizer.encode(line, add_special_tokens=False))
                for line in language_lines
            )
            word_count = sum(count_words(line) for line in language_lines)

            results.setdefault(name, {})[language] = (token_count, word_count)

    for language, language_lines in (
        ("Arabic", arabic_lines),
        ("English", english_lines),
    ):
        token_count = sum(len(custom_tokenizer.encode(line)) for line in language_lines)
        word_count = sum(count_words(line) for line in language_lines)

        results.setdefault("Custom BPE", {})[language] = (
            token_count,
            word_count,
        )

    report = [
        "# Tokenizer Benchmark",
        "",
        f"- Arabic sentences: {len(arabic_lines)}",
        f"- English sentences: {len(english_lines)}",
        f"- Custom BPE vocabulary size: {len(custom_tokenizer.vocab)}",
        "",
        "| Tokenizer | Language | Words | Tokens | Tokens per word |",
        "|---   |---    |---:   |---:   |---:   |",
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
