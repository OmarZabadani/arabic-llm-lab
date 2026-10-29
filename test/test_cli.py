from lab.cli import main


def test_main(capsys):
    main()

    captured = capsys.readouterr()
    assert captured.out.strip() == "Arabic LLM Lab"
