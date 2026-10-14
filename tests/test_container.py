"""Static checks on the Dockerfile: CI builds and runs the image, these keep its contract."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")


def final_stage() -> str:
    return re.split(r"(?m)^FROM ", DOCKERFILE)[-1]


def test_multi_stage_on_python_312_slim() -> None:
    bases = re.findall(r"(?m)^FROM (\S+)", DOCKERFILE)
    assert len(bases) >= 2
    assert all(b.startswith("python:3.12-slim") for b in bases)


def test_runs_as_non_root_uid_10001() -> None:
    stage = final_stage()
    assert re.search(r"(?m)^USER 10001(:10001)?\s*$", stage)
    assert "useradd" in stage
    assert "10001" in stage


def test_final_stage_has_no_compiler_or_build_tools() -> None:
    stage = final_stage().lower()
    for tool in ("gcc", "g++", "build-essential", "clang", "make ", "uv sync", "pip install"):
        assert tool not in stage, tool


def test_healthcheck_hits_the_service() -> None:
    stage = final_stage()
    assert "HEALTHCHECK" in stage
    assert "/healthz" in stage


def test_copied_paths_exist() -> None:
    for src in re.findall(r"(?m)^COPY (?!--from)(?:--\S+ )*(.+) \S+$", DOCKERFILE):
        for item in src.split():
            assert (ROOT / item).exists(), item
