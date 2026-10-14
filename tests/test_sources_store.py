import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fakes import HashingEncoder
from prodsearch import sources
from prodsearch.store import (
    GRAPH,
    LEXICAL,
    MANIFEST,
    VECTORS,
    HnswParams,
    build_index,
    sample_catalog_path,
)


def test_download_verifies_sha256(tmp_path: Path) -> None:
    payload = b"product_id\tproduct_name\n0\tsofa\n"
    src = tmp_path / "src.tsv"
    src.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    dest = tmp_path / "out" / "copy.tsv"
    sources.download(src.as_uri(), dest, digest)
    assert dest.read_bytes() == payload
    with pytest.raises(sources.ChecksumError):
        sources.download(src.as_uri(), tmp_path / "out" / "bad.tsv", "0" * 64)
    assert not (tmp_path / "out" / "bad.tsv").exists()
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["copy.tsv"]


def test_ensure_skips_verified_files(tmp_path: Path) -> None:
    dest = tmp_path / "f.bin"
    dest.write_bytes(b"abc")
    seen: list[str] = []
    sources.ensure("file:///nonexistent", dest, hashlib.sha256(b"abc").hexdigest(), seen.append)
    assert seen[0].startswith("ok")


def test_every_pinned_file_has_a_full_sha256() -> None:
    hashes = list(sources.WANDS_SHA256.values())
    for model in sources.MODELS:
        hashes += list(model.sha256.values())
        assert model.onnx_file in model.sha256
        assert len(model.revision) == 40
    assert all(len(h) == 64 and int(h, 16) >= 0 for h in hashes)


def test_paths_follow_the_environment(monkeypatch, tmp_path: Path) -> None:
    from prodsearch import paths

    monkeypatch.setenv("PRODSEARCH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert paths.wands_dir() == tmp_path / "wands"
    assert sources.BI_ENCODER.snapshot_dir().is_relative_to(tmp_path / "hf" / "hub")


def test_build_index_writes_every_artifact(tmp_path: Path) -> None:
    manifest = build_index(
        sample_catalog_path(),
        tmp_path,
        encoder=HashingEncoder(),
        hnsw=HnswParams(M=8, ef_construction=32),
        log=lambda _: None,
    )
    for name in (MANIFEST, LEXICAL, VECTORS, GRAPH):
        assert (tmp_path / name).is_file()
    on_disk = json.loads((tmp_path / MANIFEST).read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert manifest["n_docs"] == 40
    assert set(manifest["build_seconds"]) == {"lexical", "embed", "hnsw"}
    assert np.load(tmp_path / VECTORS).shape == (40, 64)
