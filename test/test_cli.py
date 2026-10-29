from _pytest import CaptureFixture

from lab.cli import main


def test_main(capsys: CaptureFixture) -> None:
    main()
    captured = capsys.readouterr()
    assert captured.out == "Arabic LLM Lab\n"
