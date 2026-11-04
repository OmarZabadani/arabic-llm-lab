from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


class FastByteLevelBPE:
    """Rust-backed byte-level BPE tokenizer."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 512,
        min_frequency: int = 2,
    ) -> "FastByteLevelBPE":
        """Train a byte-level BPE tokenizer using the Rust backend."""
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256.")

        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1.")

        tokenizer = Tokenizer(models.BPE(unk_token=None))

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=[],
            show_progress=True,
        )

        tokenizer.train_from_iterator(texts, trainer=trainer)

        return cls(tokenizer)

    def encode(self, text: str) -> list[int]:
        """Convert text into token IDs."""
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: Iterable[int]) -> str:
        """Convert token IDs back into UTF-8 text."""
        return self.tokenizer.decode(list(token_ids))

    def save(self, path: Path) -> None:
        """Save the tokenizer in JSON format."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))

    @classmethod
    def load(cls, path: Path) -> "FastByteLevelBPE":
        """Load a previously trained tokenizer."""
        return cls(Tokenizer.from_file(str(path)))

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
