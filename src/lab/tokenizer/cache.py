import hashlib
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

CACHE_VERSION = b"1"
BATCH_SIZE = 1024
CACHE_DIR = Path("data/cache")

SCHEMA = pa.schema(
    [
        ("text_bytes", pa.binary()),
    ]
)


def calculate_file_hash(path: Path) -> str:
    """Calculate the SHA-256 hash of a file incrementally."""
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def build_cache(
    corpus_path: Path,
    force: bool = False,
) -> Path:
    """Build a binary Arrow cache or reuse an existing valid cache."""
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = CACHE_DIR / f"{corpus_path.stem}.arrow"
    temporary_path = cache_path.with_suffix(".arrow.tmp")

    corpus_hash = calculate_file_hash(corpus_path)

    if not force and cache_path.exists():
        with pa.memory_map(str(cache_path), "r") as source:
            reader = ipc.open_file(source)
            metadata = reader.schema.metadata or {}

            if (
                metadata.get(b"corpus_hash") == corpus_hash.encode()
                and metadata.get(b"cache_version") == CACHE_VERSION
            ):
                print("Valid cache found. Reusing it.")
                return cache_path

    metadata = {
        b"corpus_hash": corpus_hash.encode(),
        b"cache_version": CACHE_VERSION,
    }

    schema = SCHEMA.with_metadata(metadata)

    batch: list[bytes] = []
    article_count = 0

    try:
        with pa.OSFile(str(temporary_path), "wb") as sink:
            with ipc.new_file(sink, schema) as writer:
                with corpus_path.open("r", encoding="utf-8") as file:
                    for line in file:
                        text = line.strip()

                        if not text:
                            continue

                        batch.append(text.encode("utf-8"))
                        article_count += 1

                        if len(batch) >= BATCH_SIZE:
                            writer.write_batch(
                                pa.record_batch(
                                    [pa.array(batch, type=pa.binary())],
                                    schema=schema,
                                )
                            )
                            batch.clear()

                    if batch:
                        writer.write_batch(
                            pa.record_batch(
                                [pa.array(batch, type=pa.binary())],
                                schema=schema,
                            )
                        )

        temporary_path.replace(cache_path)

    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"Cached articles: {article_count:,}")
    print(f"Cache saved: {cache_path}")

    return cache_path


def load_cached_batches(
    cache_path: Path,
) -> Iterator[pa.RecordBatch]:
    """Yield Arrow record batches without converting them to Python lists."""
    with pa.memory_map(str(cache_path), "r") as source:
        reader = ipc.open_file(source)

        for index in range(reader.num_record_batches):
            yield reader.get_batch(index)
