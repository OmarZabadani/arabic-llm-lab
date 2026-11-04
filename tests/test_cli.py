from unittest.mock import patch

from lab.cli import main


def test_main_without_arguments(capsys) -> None:
    main()
    captured = capsys.readouterr()

    assert "Arabic LLM Lab command-line tools." in captured.out
    assert "tokenize" in captured.out


def test_tokenize_report_runs_benchmark() -> None:
    with (
        patch("sys.argv", ["lab", "tokenize", "report"]),
        patch("lab.cli.run_tokenizer_benchmark") as mock_benchmark,
    ):
        main()

    mock_benchmark.assert_called_once()
