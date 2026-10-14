from pathlib import Path

import numpy as np

from prodsearch.catalog import (
    Catalog,
    Judgements,
    Query,
    Wands,
    load_judgements,
    parse_features,
    split_classes,
)
from prodsearch.filters import AttributeIndex, Filter


def test_sample_catalogue_loads(sample_catalog: Catalog) -> None:
    assert len(sample_catalog) == 40
    assert sample_catalog.names[0] == "oak platform bed with storage drawers"
    assert np.isnan(sample_catalog.average_rating[-1])
    passage = sample_catalog.passage(0)
    assert passage.startswith("oak platform bed with storage drawers. Beds. Furniture /")


def test_parse_features_normalises_and_drops_empty_values() -> None:
    raw = "Color : Natural Oak|seatheight:26| : slow cooker|bad|weight :"
    assert parse_features(raw) == [("color", "natural oak"), ("seatheight", "26")]


def test_split_classes() -> None:
    assert split_classes("Bookcases|Wall Mounted Shelves") == ["Bookcases", "Wall Mounted Shelves"]
    assert split_classes("") == []


def test_field_text_uses_feature_values_only(sample_catalog: Catalog) -> None:
    text = sample_catalog.field_text(0, "features")
    assert "solid wood" in text
    assert "material" not in text


def test_duplicate_judgements_keep_the_highest_grade(tmp_path: Path, sample_catalog) -> None:
    labels = tmp_path / "label.csv"
    labels.write_text(
        "id\tquery_id\tproduct_id\tlabel\n"
        "0\t7\t3\tPartial\n"
        "1\t7\t3\tExact\n"
        "2\t7\t3\tExact\n"
        "3\t7\t4\tIrrelevant\n",
        encoding="utf-8",
    )
    judgements = load_judgements(labels, sample_catalog)
    assert judgements.for_query(7) == {3: 2, 4: 0}
    assert judgements.rows == 4
    assert judgements.conflicts == 1


def test_restrict_keeps_only_judged_products(sample_catalog: Catalog) -> None:
    wands = Wands(
        sample_catalog,
        [Query(1, "bed", "Beds"), Query(2, "sofa", "Sofas")],
        Judgements({1: {0: 2, 1: 1, 30: 0}, 2: {4: 2}}),
    )
    small = wands.restrict(1, docs_per_query=2)
    assert len(small.catalog) == 2
    assert small.catalog.names == sample_catalog.names[:2]
    assert small.judgements.for_query(1) == {0: 2, 1: 1}


def test_filters(sample_catalog: Catalog) -> None:
    attrs = AttributeIndex(sample_catalog)
    assert attrs.mask(Filter()) is None

    def ids(flt: Filter) -> set[int]:
        mask = attrs.mask(flt)
        assert mask is not None
        return set(np.flatnonzero(mask).tolist())

    assert ids(Filter(classes=("bar stools",))) == {23, 24}
    # Multi-valued classes match on any of their values.
    assert 32 in ids(Filter(classes=("Wall Mounted Shelves",)))
    assert ids(Filter(classes=("kids wall decor",))) == {17}
    assert ids(Filter(category="Rugs / Area Rugs")) == {25, 26}
    assert ids(Filter(category="rugs")) == {25, 26, 27, 37}
    assert ids(Filter(category="Rugs / Area")) == set()
    assert ids(Filter(classes=("Area Rugs",), attributes=(("color", "cream"),))) == {25}
    assert ids(Filter(min_rating=5.0)) == {17}
    # Unrated products never pass a rating filter.
    assert 39 not in ids(Filter(min_rating=0.0))
    assert ids(Filter(min_reviews=700)) == {26, 35, 36}
