"""Resolve a leaf's type to its ground type and its constraint.

The chain of ``typedef`` statements is followed as far as it goes, and each prefix is resolved
against the module's own ``import`` statements. Where the chain cannot be followed the result
is CANNOT-RESOLVE with the chain shown, never a guessed constraint.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

#: YANG built-in types that end a chain.
BUILTINS = {
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    "decimal64", "string", "boolean", "enumeration", "bits", "binary",
    "leafref", "identityref", "empty", "union", "instance-identifier",
}
NUMERIC = {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "decimal64"}

_IMPORT = re.compile(r"^\s*import\s+([\w.\-]+)\s*\{[^}]*?prefix\s+([\w.\-]+)", re.M | re.S)
_RANGE = re.compile(r'range\s+"([^"]+)"')
_ENUM = re.compile(r'enum\s+"?([\w.\-]+)"?')


@dataclass
class Constraint:
    """What a leaf actually accepts, and how we got there."""

    kind: str                    # 'range' | 'enum' | 'unconstrained' | 'cannot-resolve'
    ground_type: str | None = None
    spec: object = None          # range string, or list of enum values
    chain: list = field(default_factory=list)   # ['file:line type X', ...]

    @property
    def resolved(self) -> bool:
        return self.kind in ("range", "enum")

    @property
    def numeric(self) -> bool:
        return self.ground_type in NUMERIC

    def render(self) -> str:
        head = {"range": f'range "{self.spec}"', "enum": f"enum {self.spec}",
                "unconstrained": f"{self.ground_type} with no range or enumeration",
                "cannot-resolve": "CANNOT-RESOLVE"}[self.kind]
        return f"{head} [{' -> '.join(self.chain)}]"


def _prefix_map(text: str) -> dict:
    return {p: m for m, p in _IMPORT.findall(text)}


def _block_at(text: str, start: int) -> str:
    i = text.index("{", start)
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    return text[i:j + 1]


def _find_typedef(name: str, module: str | None, yang_dir: Path):
    """The typedef block, preferring the module the prefix resolves to."""
    candidates = []
    if module:
        p = yang_dir / f"{module}.yang"
        if p.is_file():
            candidates.append(p)
    candidates += [p for p in sorted(yang_dir.glob("*.yang")) if p not in candidates]
    for p in candidates:
        text = p.read_text(encoding="utf-8", errors="replace")
        m = re.search(rf"^[ \t]*typedef[ \t]+{re.escape(name)}[ \t]*\{{", text, re.M)
        if m:
            line = text[:m.start()].count("\n") + 1
            return p, line, _block_at(text, m.start()), text
    return None, None, None, None


def resolve_constraint(leaf_body: str, module_file: Path, yang_dir: Path,
                       depth: int = 0) -> Constraint:
    """Follow ``type`` from a leaf body to a ground type, collecting the constraint on the way.

    The FIRST range or enumeration met on the way down wins, which is YANG's own rule: a typedef
    may restrict its base, and the leaf may restrict the typedef.
    """
    yang_dir = Path(yang_dir)
    chain: list[str] = []
    body = leaf_body
    text = Path(module_file).read_text(encoding="utf-8", errors="replace")
    seen = set()
    for _ in range(16):                       # chains are short; the bound is a loop guard
        tm = re.search(r"type\s+(?:([\w.\-]+):)?([\w.\-]+)", body)
        if not tm:
            return Constraint("cannot-resolve", None, None, chain + ["no type statement"])
        prefix, tname = tm.group(1), tm.group(2)
        chain.append(f"type {prefix + ':' if prefix else ''}{tname}")
        r = _RANGE.search(body)
        if r:
            return Constraint("range", tname if tname in BUILTINS else None, r.group(1), chain)
        enums = _ENUM.findall(body)
        if enums:
            return Constraint("enum", "enumeration", enums, chain)
        if tname in BUILTINS:
            return Constraint("unconstrained", tname, None, chain)
        if tname in seen:
            return Constraint("cannot-resolve", None, None, chain + ["cycle"])
        seen.add(tname)
        module = _prefix_map(text).get(prefix) if prefix else None
        p, line, block, ntext = _find_typedef(tname, module, yang_dir)
        if block is None:
            return Constraint("cannot-resolve", None, None, chain + [f"typedef {tname} not found"])
        chain[-1] = f"{p.name}:{line} {chain[-1]}"
        body, text = block, ntext
    return Constraint("cannot-resolve", None, None, chain + ["chain too deep"])
