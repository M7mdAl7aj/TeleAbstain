"""A record of the environment of a judge run, written beside the verdicts.

A judge's identifier records the model, the precision, the decoding settings and the rubric. It
does not record library versions, and it cannot record whether a model ran in the precision its
name implies: `transformers` converts MXFP4 weights to bf16 when the `kernels` package is
absent. This module records both. The record is captured once per run, hashed, and written as a
file beside the verdicts, and the hash goes on every verdict row.

Everything is read from the installed packages and the running process. A field that cannot be
determined is recorded as ``None`` and is never given a default.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field

#: Packages whose version can change a generation result. Kept explicit rather than dumping
#: `pip freeze`, so the hash moves only when something load-bearing moves.
TRACKED = ("torch", "transformers", "tokenizers", "accelerate", "numpy", "kernels",
           "triton", "llama_cpp", "anthropic")


def _version(name: str) -> str | None:
    try:
        mod = importlib.import_module(name)
    except Exception:
        return None
    return getattr(mod, "__version__", None) or "present-version-unknown"


def _cuda() -> dict:
    try:
        import torch
    except Exception:
        return {"available": None, "device_name": None, "cuda_version": None,
                "device_count": None}
    try:
        avail = bool(torch.cuda.is_available())
        return {"available": avail,
                "device_name": torch.cuda.get_device_name(0) if avail else None,
                "cuda_version": getattr(torch.version, "cuda", None),
                "device_count": torch.cuda.device_count() if avail else 0}
    except Exception:
        return {"available": None, "device_name": None, "cuda_version": None,
                "device_count": None}


def mxfp4_state() -> dict:
    """Whether an MXFP4 checkpoint would run as MXFP4 or be converted to bf16.

    `transformers` converts to bf16 when `kernels` is missing, and prints a notice without
    raising, so which path a run took is recorded as a field of its own.
    """
    kernels = _version("kernels")
    return {"kernels_installed": kernels is not None,
            "kernels_version": kernels,
            "mxfp4_path": "dequantized-to-bf16" if kernels is None else "native-mxfp4",
            "note": ("transformers dequantizes MXFP4 to bf16 when `kernels` is absent; this "
                     "field records which path a run took")}


@dataclass
class EnvRecord:
    """One run's environment. ``env_hash`` goes on every verdict row."""

    python: str
    platform: str
    packages: dict
    cuda: dict
    mxfp4: dict
    captured_by: str = "teleabstain.env_record"
    schema: int = 1
    env_hash: str = field(default="", compare=False)

    def compute_hash(self) -> str:
        payload = {"python": self.python, "platform": self.platform,
                   "packages": self.packages, "cuda": self.cuda,
                   "mxfp4": {k: v for k, v in self.mxfp4.items() if k != "note"},
                   "schema": self.schema}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def as_dict(self) -> dict:
        return asdict(self)


def capture() -> EnvRecord:
    """Read the live environment. Cheap enough to call once per run."""
    rec = EnvRecord(
        python=sys.version.split()[0],
        platform=platform.platform(),
        packages={name: _version(name) for name in TRACKED},
        cuda=_cuda(),
        mxfp4=mxfp4_state(),
    )
    rec.env_hash = rec.compute_hash()
    return rec


def write_sidecar(path, record: EnvRecord | None = None) -> EnvRecord:
    """Write ``<path>`` as the environment sidecar and return the record."""
    from pathlib import Path

    rec = record or capture()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rec
