"""The definition of "attribute of the class".

An attribute of the NRCellDU class is a ``leaf`` or ``leaf-list`` declared directly in the
class's own module file, as a child of the class's attribute grouping, at the item's pinned
commit. It is not:

- brought in by ``uses`` from another module,
- nested inside a contained ``list`` or ``container``,
- a ``choice`` or ``case`` name,
- the class ``list`` itself.

This module is the only place where the set of attributes is computed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: Statement keywords that open a lexical scope we must track to know our depth.
_SCOPED = ("grouping", "container", "list", "leaf", "leaf-list", "choice", "case", "augment",
           "rpc", "notification", "action", "input", "output")

_STMT = re.compile(r"^\s*(" + "|".join(_SCOPED) + r")\s+([\w:.\-\"/]+)")

#: The grouping that holds NRCellDU's own attributes. Named explicitly, not inferred: the module
#: also declares nested lists whose leaves are not attributes of the class.
CLASS_GROUPING = {"_3gpp-nr-nrm-nrcelldu": "NRCellDUGrp"}


@dataclass(frozen=True)
class NativeAttr:
    """One native attribute: what it is, and where it is declared."""

    name: str
    statement: str      # 'leaf' or 'leaf-list'
    file: str
    line: int

    @property
    def evidence(self) -> str:
        return f"{self.file}:{self.line} ({self.statement})"


def native_attributes(class_module: str, yang_dir: str | Path) -> dict[str, NativeAttr]:
    """``{name: NativeAttr}`` for the class's own attributes at one pinned state.

    Depth-aware by brace tracking: a ``leaf`` counts only when its immediately enclosing scope is
    the class grouping itself. A leaf one level deeper - inside a nested ``list`` such as
    ``nPNIdentityList`` or ``pLMNInfoList`` - is a member of that list, not an attribute of the
    class, and is excluded.
    """
    grouping = CLASS_GROUPING.get(class_module)
    if grouping is None:
        raise KeyError(f"no class grouping recorded for {class_module!r}")
    path = Path(yang_dir) / f"{class_module}.yang"
    text = path.read_text(encoding="utf-8", errors="replace")

    out: dict[str, NativeAttr] = {}
    # (keyword, name, depth at which the statement opened). Brace depth is NOT stack length:
    # `type x { ... }` and `range "..." { ... }` open braces without being tracked statements, so
    # a length-based pop unwinds the stack at the wrong moment and loses the enclosing scope.
    stack: list[tuple[str, str, int]] = []
    depth = 0
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.split("//")[0]
        m = _STMT.match(stripped)
        opens, closes = stripped.count("{"), stripped.count("}")
        if m and opens:
            kw, name = m.group(1), m.group(2).strip('"')
            # The enclosing scope is what the stack holds BEFORE this statement is pushed.
            if kw in ("leaf", "leaf-list") and stack and stack[-1][:2] == ("grouping", grouping):
                out[name] = NativeAttr(name, kw, path.name, lineno)
            stack.append((kw, name, depth))
        depth += opens - closes
        # A statement opened at depth d closes when depth returns to d.
        while stack and depth <= stack[-1][2]:
            stack.pop()
    return out


def native_names(class_module: str, yang_dir: str | Path) -> set[str]:
    """Just the names, for callers that only need the pool."""
    return set(native_attributes(class_module, yang_dir))


def declarations_elsewhere(name: str, class_module: str, yang_dir: str | Path) -> list[str]:
    """``file:line`` of every ``leaf``/``leaf-list`` declaration of ``name`` in OTHER modules.

    A cross-domain control rests on its name being declared somewhere else in the same pinned
    tree as well; this finds those declarations.
    """
    out = []
    for path in sorted(Path(yang_dir).glob("*.yang")):
        if path.stem == class_module:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        # `[ \t]*`, not `\s*`: `\s` also matches a newline, and a match that starts on a blank
        # line would report the line before the declaration.
        for m in re.finditer(r"^[ \t]*(leaf|leaf-list)[ \t]+" + re.escape(name) + r"[ \t]*\{",
                             text, re.M):
            out.append(f"{path.name}:{text[:m.start()].count(chr(10)) + 1}")
    return out


def declared_anywhere(name: str, yang_dir: str | Path) -> list[str]:
    """``file:line`` of every ``leaf``/``leaf-list`` declaration of ``name``, class module included.

    Used by the unanswerable families, whose keys rest on a name being declared **nowhere** in the
    closure.
    """
    out = []
    for path in sorted(Path(yang_dir).glob("*.yang")):
        text = path.read_text(encoding="utf-8", errors="replace")
        # `[ \t]*`, not `\s*`: `\s` also matches a newline, and a match that starts on a blank
        # line would report the line before the declaration.
        for m in re.finditer(r"^[ \t]*(leaf|leaf-list)[ \t]+" + re.escape(name) + r"[ \t]*\{",
                             text, re.M):
            out.append(f"{path.name}:{text[:m.start()].count(chr(10)) + 1}")
    return out


def enclosing_path(name: str, module_file: str | Path, line: int) -> list[str]:
    """The grouping/container/list chain enclosing the declaration at ``line``.

    Same brace-tracking discipline as :func:`native_attributes`, so the two can never disagree
    about what encloses what.
    """
    text = Path(module_file).read_text(encoding="utf-8", errors="replace").splitlines()
    stack: list[tuple[str, str, int]] = []
    depth = 0
    for lineno, raw in enumerate(text, 1):
        stripped = raw.split("//")[0]
        m = _STMT.match(stripped)
        opens, closes = stripped.count("{"), stripped.count("}")
        if m and opens:
            stack.append((m.group(1), m.group(2).strip('"'), depth))
        depth += opens - closes
        if lineno == line:
            return [f"{k} {n}" for k, n, _ in stack]
        while stack and depth <= stack[-1][2]:
            stack.pop()
    return []
