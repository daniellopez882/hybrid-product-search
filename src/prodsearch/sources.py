"""Pinned downloads: the WANDS judgements and the two ONNX models, each checked against sha256.

WANDS is read from Wayfair's GitHub repository at a fixed commit. The models are read from the
Hugging Face Hub at fixed revisions and stored in the Hub cache layout under `HF_HOME`, so a copy
that another tool already fetched is verified and reused instead of downloaded again.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from prodsearch.paths import hf_home, wands_dir

WANDS_COMMIT = "3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5"
WANDS_URL = "https://raw.githubusercontent.com/wayfair/WANDS/{commit}/dataset/{name}"
WANDS_LICENCE = "MIT (Wayfair LLC)"

WANDS_SHA256: dict[str, str] = {
    "product.csv": "d993926254572e6eba96c8fd87cc549a17fb91ad3748308036eee4cf92b10ac6",
    "query.csv": "63b61660560fecc33ec490804c7e2b81402ee3e7c31a9cbb5e03736639f68e95",
    "label.csv": "c11fe81ad62f17f56f316b0ec9630ebe8fbe1393578cb0ca4f05c17253a180ef",
}


@dataclass(frozen=True)
class HubModel:
    """One ONNX model pinned to a Hub revision, with the hashes observed on first download."""

    key: str
    repo_id: str
    revision: str
    onnx_file: str
    licence: str
    sha256: dict[str, str]

    def snapshot_dir(self) -> Path:
        folder = "models--" + self.repo_id.replace("/", "--")
        return hf_home() / "hub" / folder / "snapshots" / self.revision

    def path(self, filename: str) -> Path:
        return self.snapshot_dir() / filename

    def url(self, filename: str) -> str:
        return f"https://huggingface.co/{self.repo_id}/resolve/{self.revision}/{filename}"


BI_ENCODER = HubModel(
    key="bi-encoder",
    repo_id="BAAI/bge-small-en-v1.5",
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    onnx_file="onnx/model.onnx",
    licence="MIT",
    sha256={
        "onnx/model.onnx": "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35",
        "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
        "config.json": "094f8e891b932f2000c92cfc663bac4c62069f5d8af5b5278c4306aef3084750",
    },
)

CROSS_ENCODER = HubModel(
    key="cross-encoder",
    repo_id="cross-encoder/ms-marco-MiniLM-L6-v2",
    revision="233902d25c440f23af6f7d6e94d2946bac0bee0a",
    onnx_file="onnx/model_quint8_avx2.onnx",
    licence="Apache-2.0",
    sha256={
        "onnx/model_quint8_avx2.onnx": (
            "c80a8b34256ea453093d612e3ac48d3d965a0c0a48c7906709af8b8e28461bf9"
        ),
        "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
        "config.json": "380e02c93f431831be65d99a4e7e5f67c133985bf2e77d9d4eba46847190bacc",
    },
)

MODELS: tuple[HubModel, ...] = (BI_ENCODER, CROSS_ENCODER)


class ChecksumError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, expected_sha256: str) -> None:
    """Stream `url` to a temporary file next to `dest`, check the hash, then move it in place."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "prodsearch-fetch"})
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                shutil.copyfileobj(response, tmp, length=1 << 20)
        except BaseException:
            tmp.close()
            tmp_path.unlink(missing_ok=True)
            raise
    observed = sha256_of(tmp_path)
    if observed != expected_sha256:
        tmp_path.unlink(missing_ok=True)
        raise ChecksumError(f"{url}: sha256 {observed}, expected {expected_sha256}")
    tmp_path.replace(dest)


def ensure(url: str, dest: Path, expected_sha256: str, log: Callable[[str], None]) -> None:
    if dest.is_file():
        if sha256_of(dest) == expected_sha256:
            log(f"ok        {dest}")
            return
        log(f"mismatch  {dest} (downloading again)")
    log(f"download  {url}")
    download(url, dest, expected_sha256)
    log(f"verified  {dest}")


def fetch_wands(log: Callable[[str], None] = print) -> Path:
    target = wands_dir()
    for name, digest in WANDS_SHA256.items():
        ensure(WANDS_URL.format(commit=WANDS_COMMIT, name=name), target / name, digest, log)
    return target


def fetch_model(model: HubModel, log: Callable[[str], None] = print) -> Path:
    for filename, digest in model.sha256.items():
        ensure(model.url(filename), model.path(filename), digest, log)
    return model.snapshot_dir()


def wands_present() -> bool:
    return all((wands_dir() / name).is_file() for name in WANDS_SHA256)


def model_present(model: HubModel) -> bool:
    return all(model.path(name).is_file() for name in model.sha256)
