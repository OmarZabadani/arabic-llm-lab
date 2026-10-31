"""
checks whether BPE implementation works correctly.
"""

import pytest

from lab.tokenizer.bpe import ByteLevelBPE


@pytest.fixture
def tokenizer() -> ByteLevelBPE:
    texts = [
        "hello hello world",
        "hello there",
        "مرحبا بالعالم",
        "مرحبا بكم",
    ]
    return ByteLevelBPE.train(texts, vocab_size=300)


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
    tokenizer: ByteLevelBPE,
    text: str,
) -> None:
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_training_creates_merges(tokenizer: ByteLevelBPE) -> None:
    assert len(tokenizer.merges) > 0


def test_initial_vocabulary_contains_all_bytes() -> None:
    tokenizer = ByteLevelBPE.train([], vocab_size=256)
    assert len(tokenizer.vocab) == 256


def test_rejects_vocab_size_below_256() -> None:
    with pytest.raises(ValueError):
        ByteLevelBPE.train(["hello"], vocab_size=100)


def test_rejects_unknown_token_id() -> None:
    tokenizer = ByteLevelBPE.train(["hello"], vocab_size=256)

    with pytest.raises(ValueError):
        tokenizer.decode([999])
