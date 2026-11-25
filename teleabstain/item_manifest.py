"""The item manifest: which items make up the benchmark, and the hash of each prompt.

A manifest names suites and item identifiers, never a directory. The directory that holds the
item files is passed at run time, and `verify_manifest` resolves the manifest against it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from teleabstain.corpus import CorpusError, load_suite


class ManifestError(Exception):
    """The manifest does not resolve against the corpus it is given."""


@dataclass(frozen=True)
class ItemRef:
    suite: str
    item_id: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.suite}/{self.item_id}"


def load_manifest(path: Path | str) -> list[ItemRef]:
    """Read the manifest as an ordered, de-duplicated list of item references."""
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"no manifest at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    refs, seen = [], set()
    for entry in data["items"]:
        ref = ItemRef(entry["suite"], entry["item_id"])
        if ref in seen:
            raise ManifestError(f"duplicate entry {ref}")
        seen.add(ref)
        refs.append(ref)
    return refs


def verify_manifest(corpus_dir: Path | str, path: Path | str) -> list[ItemRef]:
    """Every manifest entry must resolve to an item in ``corpus_dir``.

    Raises instead of filtering: an entry that cannot be found is an error and is never skipped.
    """
    refs = load_manifest(path)
    by_suite: dict[str, set[str]] = {}
    for ref in refs:
        if ref.suite not in by_suite:
            try:
                by_suite[ref.suite] = {i.id for i in load_suite(corpus_dir, ref.suite).items}
            except CorpusError as exc:
                raise ManifestError(f"suite {ref.suite!r} not in {corpus_dir}: {exc}") from exc
        if ref.item_id not in by_suite[ref.suite]:
            raise ManifestError(f"{ref} is not in {corpus_dir}")
    return refs


def manifest_suites(path: Path | str) -> list[str]:
    """The suites the manifest covers, in stable order."""
    seen: list[str] = []
    for ref in load_manifest(path):
        if ref.suite not in seen:
            seen.append(ref.suite)
    return seen


def manifest_item_ids(suite: str, path: Path | str) -> set[str]:
    """The item ids the manifest selects for one suite."""
    return {ref.item_id for ref in load_manifest(path) if ref.suite == suite}


def arms_agree(*item_sets: set[ItemRef] | list[ItemRef]) -> bool:
    """Do all the given item sets hold exactly the same items?"""
    sets = [set(s) for s in item_sets]
    return all(s == sets[0] for s in sets)


# ---------------------------------------------------------------- per-item prompt hashes
#: The hash covers the item's own prompt text, not the text a model sees once a chat template
#: has been applied, so one manifest serves every model whatever its template.
def prompt_sha256(text: str) -> str:
    """sha256 of an item's prompt text, UTF-8, no normalisation and no trimming.

    Deliberately raw: a leading space or a changed newline IS a different prompt to a model, so
    normalising here would hide the rewordings the hash exists to catch.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_sha256(items: "Iterable[tuple[str, str, str]]") -> str:
    """Checksum of the item set: sha256 over sorted ``id\x1fkind\x1fprompt`` lines.

    Sorted, so file order and suite order cannot move it; ``\x1f`` separated, so no id or prompt
    can forge a boundary.
    """
    digest = hashlib.sha256()
    for line in sorted("\x1f".join(t) for t in items):
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def manifest_prompt_hashes(path: Path | str) -> dict[str, str]:
    """``{item_id: prompt_sha256}``. Empty for a manifest that carries no prompt hashes."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {e["item_id"]: e["prompt_sha256"] for e in data["items"] if e.get("prompt_sha256")}


def read_item_ids(path: "Path | str") -> set[str]:
    """Ids from a plain list file: one per line, blank lines and `#` comments ignored.

    A list file is not a manifest: it selects items to run and says nothing about what the
    benchmark holds.
    """
    out = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line)
    return out
