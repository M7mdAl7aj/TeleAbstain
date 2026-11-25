#!/usr/bin/env python3
"""Facts about a module's data tree, read through pyang's Python API.

For a module at a pinned state, `module_nodes` returns its data nodes, each with its owner:
`OWN` when the node hangs off the module's top managed-object class, or the name of the
contained child class it belongs to. `config` is carried so that read-only nodes are visible.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    name: str
    owner: str          # 'OWN' or the contained child class
    config: bool
    mandatory: bool
    path: str


def _walk(stmt, owner, depth, out, top_seen):
    for child in getattr(stmt, "i_children", []) or []:
        kw = child.keyword
        if kw not in ("container", "list", "leaf", "leaf-list", "choice", "case"):
            continue
        name = child.arg
        # A capitalised list one level in is a contained managed-object class.
        new_owner = owner
        if kw == "list" and name[:1].isupper():
            new_owner = "OWN" if not top_seen else name
        cfg = getattr(child, "i_config", True)
        man = getattr(child, "search_one", lambda *_: None)("mandatory")
        out.append(Node(name, owner if not name[:1].isupper() else new_owner,
                        bool(cfg), bool(man and man.arg == "true"), f"{depth}/{name}"))
        _walk(child, new_owner if name[:1].isupper() else owner, depth + 1, out,
              top_seen or (kw == "list" and name[:1].isupper()))


def module_nodes(module_name: str, yang_dir: str) -> list[Node]:
    """All data nodes of `module_name` at the state rooted at `yang_dir`."""
    from pyang import context, repository
    repo = repository.FileRepository(yang_dir, use_env=False)
    ctx = context.Context(repo)
    path = os.path.join(yang_dir, module_name + ".yang")
    with open(path, encoding="utf-8", errors="replace") as fh:
        mod = ctx.add_module(path, fh.read())
    ctx.validate()
    if mod is None:
        raise RuntimeError(f"pyang could not load {module_name}")
    out: list[Node] = []
    # Top-level data, plus anything this module augments into another tree.
    _walk(mod, "OWN", 0, out, False)
    for sub in mod.substmts:
        if sub.keyword == "augment":
            _walk(sub, "OWN", 0, out, False)
    return out


def owners_by_name(module_name: str, yang_dir: str) -> dict[str, set[str]]:
    d: dict[str, set[str]] = {}
    for n in module_nodes(module_name, yang_dir):
        d.setdefault(n.name, set()).add(n.owner)
    return d
