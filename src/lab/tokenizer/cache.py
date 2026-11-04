import hashlib
import struct
import uuid
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

CACHE_VERSION = b"1"
BATCH_SIZE = 1024
CACHE_DIR = Path("data/cache")

SCHEMA = pa.schema(
    [
        ("text_bytes", pa.large_binary()),
    ]
)


def calculate_file_hash(path: Path) -> str:
    """Calculate a SHA-256 hash without loading the entire file into memory."""
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def is_valid_cache(cache_path: Path, corpus_hash: str) -> bool:
    """Check whether a cache matches the source corpus and format version."""
    try:
        with pa.memory_map(str(cache_path), "r") as source:
            reader = ipc.open_file(source)
            metadata = reader.schema.metadata or {}

            return (
                metadata.get(b"corpus_hash") == corpus_hash.encode("ascii")
                and metadata.get(b"cache_version") == CACHE_VERSION
            )
    except Exception:
        # A damaged or unreadable cache must be rebuilt.
        return False


def build_cache(
    corpus_path: Path,
    force: bool = False,
) -> Path:
    """Build or reuse a memory-mappable Arrow cache."""
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    corpus_hash = calculate_file_hash(corpus_path)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = CACHE_DIR / (f"{corpus_path.stem}-{corpus_hash[:16]}.arrow")

    if not force and cache_path.exists():
        if is_valid_cache(cache_path, corpus_hash):
            print(f"Reusing cache: {cache_path}")
            return cache_path

    metadata = {
        b"corpus_hash": corpus_hash.encode("ascii"),
        b"cache_version": CACHE_VERSION,
    }

    schema = SCHEMA.with_metadata(metadata)

    temporary_path = CACHE_DIR / (f".{cache_path.name}.{uuid.uuid4().hex}.tmp")

    batch: list[bytes] = []
    article_count = 0

    try:
        with pa.OSFile(str(temporary_path), "wb") as sink:
            with ipc.new_file(sink, schema) as writer:
                with corpus_path.open("r", encoding="utf-8") as file:
                    for line in file:
                        text = line.rstrip("\r\n")

                        if not text.strip():
                            continue

                        batch.append(text.encode("utf-8"))
                        article_count += 1

                        if len(batch) >= BATCH_SIZE:
                            writer.write_batch(
                                pa.record_batch(
                                    [pa.array(batch, type=pa.large_binary())],
                                    schema=schema,
                                )
                            )
                            batch.clear()

                    if batch:
                        writer.write_batch(
                            pa.record_batch(
                                [pa.array(batch, type=pa.large_binary())],
                                schema=schema,
                            )
                        )

        temporary_path.replace(cache_path)

    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"Cached articles: {article_count:,}")
    print(f"Cache size: {cache_path.stat().st_size / (1024**2):.2f} MB")
    print(f"Cache saved: {cache_path}")

    return cache_path


class CacheReader:
    """Read cached articles while keeping the memory map alive."""

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self.source: pa.MemoryMappedFile | None = None
        self.reader: ipc.RecordBatchFileReader | None = None

    def __enter__(self) -> "CacheReader":
        self.source = pa.memory_map(str(self.cache_path), "r")
        self.reader = ipc.open_file(self.source)
        return self

    def __exit__(self, *args: object) -> None:
        if self.source is not None:
            self.source.close()

        self.source = None
        self.reader = None

    def iter_batches(self) -> Iterator[pa.RecordBatch]:
        """Yield Arrow batches without converting them to Python lists."""
        if self.reader is None:
            raise RuntimeError("Open CacheReader using a with statement.")

        for index in range(self.reader.num_record_batches):
            yield self.reader.get_batch(index)

    def iter_articles(self) -> Iterator[memoryview]:
        """Yield zero-copy views of individual article bytes."""
        for batch in self.iter_batches():
            array = batch.column(0)
            buffers = array.buffers()

            offsets_buffer = buffers[1]
            values_buffer = buffers[2]

            if offsets_buffer is None or values_buffer is None:
                continue

            offsets = memoryview(offsets_buffer)
            values = memoryview(values_buffer)

            for index in range(len(array)):
                position = (array.offset + index) * 8

                start = struct.unpack_from("<q", offsets, position)[0]
                end = struct.unpack_from("<q", offsets, position + 8)[0]

                yield values[start:end]
