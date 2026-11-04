import argparse

from lab.tokenizer.benchmark import main as run_tokenizer_benchmark


def main() -> None:
    """Run the Arabic LLM Lab command-line interface."""
    parser = argparse.ArgumentParser(
        prog="lab",
        description="Arabic LLM Lab command-line tools.",
    )

    subparsers = parser.add_subparsers(dest="command")

    tokenize_parser = subparsers.add_parser(
        "tokenize",
        help="Tokenizer operations.",
    )

    tokenize_subparsers = tokenize_parser.add_subparsers(
        dest="tokenize_command",
    )

    tokenize_subparsers.add_parser(
        "report",
        help="Compare tokenizer performance.",
    )

    args = parser.parse_args()

    if args.command == "tokenize" and args.tokenize_command == "report":
        run_tokenizer_benchmark()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
