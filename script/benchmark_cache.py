from pathlib import Path
from time import perf_counter

from lab.tokenizer.cache import CacheReader, build_cache

CORPUS_PATH = Path("data/arabic_train_clean.txt")


def main() -> None:
    cache_path = build_cache(CORPUS_PATH)

    start = perf_counter()
    article_count = 0
    total_bytes = 0

    with CacheReader(cache_path) as reader:
        for article in reader.iter_articles():
            article_count += 1
            total_bytes += len(article)

    elapsed = perf_counter() - start

    print("\nCache reading benchmark")
    print("-" * 40)
    print(f"Articles read: {article_count:,}")
    print(f"Bytes read: {total_bytes:,}")
    print(f"Reading time: {elapsed:.3f} seconds")
    print(f"Throughput: {total_bytes / (1024**2) / elapsed:.2f} MB/s")
    print(f"Articles/sec: {article_count / elapsed:,.0f}")


if __name__ == "__main__":
    main()
