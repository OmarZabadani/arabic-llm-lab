from pathlib import Path

import pytest

from lab.tokenizer.fast_bpe import FastByteLevelBPE


@pytest.fixture
def tokenizer() -> FastByteLevelBPE:
    return FastByteLevelBPE.train(
        [
            "hello hello world",
            "hello there",
            "مرحبا بالعالم",
            "مرحبا بكم",
        ],
        vocab_size=300,
    )


@pytest.mark.parametrize(
    "text",
    [
        "Hello world!",
        "مرحبا بالعالم",
        "Hello مرحبا 123!",
        "",
    ],
)
def test_encode_decode_round_trip(
    tokenizer: FastByteLevelBPE,
    text: str,
) -> None:
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_training_creates_vocabulary(
    tokenizer: FastByteLevelBPE,
) -> None:
    assert tokenizer.vocab_size > 256


def test_save_and_load(
    tokenizer: FastByteLevelBPE,
    tmp_path: Path,
) -> None:
    path = tmp_path / "tokenizer.json"

    tokenizer.save(path)
    loaded = FastByteLevelBPE.load(path)

    text = "مرحبا بالعالم"
    assert loaded.decode(loaded.encode(text)) == text


def test_rejects_vocab_size_below_256() -> None:
    with pytest.raises(ValueError):
        FastByteLevelBPE.train(["hello"], vocab_size=100)


def test_rejects_invalid_min_frequency() -> None:
    with pytest.raises(ValueError):
        FastByteLevelBPE.train(["hello"], min_frequency=0)
