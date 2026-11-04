from pathlib import Path

from lab.tokenizer.cache import CacheReader, build_cache


def test_cache_round_trip(tmp_path: Path) -> None:
    corpus = tmp_path / "sample.txt"
    corpus.write_text(
        "مرحبا بالعالم\nHello world\n\nاختبار\n",
        encoding="utf-8",
    )

    cache_path = build_cache(corpus)

    with CacheReader(cache_path) as reader:
        articles = [
            article.tobytes().decode("utf-8") for article in reader.iter_articles()
        ]

    assert articles == [
        "مرحبا بالعالم",
        "Hello world",
        "اختبار",
    ]


def test_cache_reuse(tmp_path: Path) -> None:
    corpus = tmp_path / "sample.txt"
    corpus.write_text("مرحبا\n", encoding="utf-8")

    first_cache = build_cache(corpus)
    second_cache = build_cache(corpus)

    assert first_cache == second_cache
