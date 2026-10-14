import pytest

from prodsearch.analysis import analyze, fold, s_stem


@pytest.mark.parametrize(
    ("word", "stem"),
    [
        ("chairs", "chair"),
        ("stools", "stool"),
        ("pillows", "pillow"),
        ("accessories", "accessory"),
        ("benches", "bench"),
        ("boxes", "box"),
        ("glasses", "glass"),
        ("dishes", "dish"),
        ("tables", "table"),
        ("shoes", "shoe"),
        ("glass", "glass"),
        ("cactus", "cactus"),
        ("bus", "bus"),
        ("modern", "modern"),
    ],
)
def test_s_stem(word: str, stem: str) -> None:
    assert s_stem(word) == stem


def test_fold_strips_accents_and_case() -> None:
    assert fold("Kids Wall Décor") == "kids wall decor"


def test_analyze_drops_stopwords_and_punctuation() -> None:
    assert analyze("Chair and a Half, Recliners!") == ["chair", "half", "recliner"]


def test_analyze_splits_numbers_and_units() -> None:
    assert analyze("8' x 10' Rug") == ["8", "x", "10", "rug"]
