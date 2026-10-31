from collections import Counter
from collections.abc import Iterable


class ByteLevelBPE:
    """A minimal byte-level Byte Pair Encoding tokenizer."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[int, int]],
    ) -> None:
        self.vocab = vocab
        self.merges = merges

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 512,
    ) -> "ByteLevelBPE":
        """Train BPE by repeatedly merging the most frequent byte-token pair."""
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256.")

        corpus = [list(text.encode("utf-8")) for text in texts]

        vocab = {i: bytes([i]) for i in range(256)}
        merges: list[tuple[int, int]] = []

        while len(vocab) < vocab_size:
            pair_counts: Counter[tuple[int, int]] = Counter()

            for sequence in corpus:
                pair_counts.update(zip(sequence, sequence[1:], strict=False))

            if not pair_counts:
                break

            # Deterministic tie-breaking makes training reproducible.
            pair, frequency = max(
                pair_counts.items(),
                key=lambda item: (
                    item[1],
                    -item[0][0],
                    -item[0][1],
                ),
            )

            if frequency < 2:
                break

            left, right = pair
            new_id = len(vocab)
            vocab[new_id] = vocab[left] + vocab[right]
            merges.append(pair)

            # Apply the selected merge throughout the corpus.
            updated_corpus = []

            for sequence in corpus:
                merged_sequence = []
                index = 0

                while index < len(sequence):
                    if (
                        index + 1 < len(sequence)
                        and sequence[index] == left
                        and sequence[index + 1] == right
                    ):
                        merged_sequence.append(new_id)
                        index += 2
                    else:
                        merged_sequence.append(sequence[index])
                        index += 1

                updated_corpus.append(merged_sequence)

            corpus = updated_corpus

        return cls(vocab=vocab, merges=merges)

    def encode(self, text: str) -> list[int]:
        """Convert text into token IDs using the learned merges."""
        tokens = list(text.encode("utf-8"))

        for merge_index, (left, right) in enumerate(self.merges):
            new_id = 256 + merge_index
            merged_tokens = []
            index = 0

            while index < len(tokens):
                if (
                    index + 1 < len(tokens)
                    and tokens[index] == left
                    and tokens[index + 1] == right
                ):
                    merged_tokens.append(new_id)
                    index += 2
                else:
                    merged_tokens.append(tokens[index])
                    index += 1

            tokens = merged_tokens

        return tokens

    def decode(self, token_ids: Iterable[int]) -> str:
        """Convert token IDs back into UTF-8 text."""
        token_bytes = []

        for token_id in token_ids:
            if token_id not in self.vocab:
                raise ValueError(f"Unknown token ID: {token_id}")

            token_bytes.append(self.vocab[token_id])

        return b"".join(token_bytes).decode("utf-8")
