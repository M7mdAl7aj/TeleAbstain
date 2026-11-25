from __future__ import annotations

from pathlib import Path

import pytest

from teleabstain.native_attributes import native_attributes, native_names

REPO = Path(__file__).resolve().parent.parent
STATES = {
    "Rel-19": REPO / "vendor/MnS/yang-models",
    "Rel-16": None,   # era state is extracted per run; resolved below when present
}
MOD = "_3gpp-nr-nrm-nrcelldu"
pytestmark = pytest.mark.skipif(not (REPO / "vendor/MnS/yang-models").is_dir(),
                                reason="vendor Rel-19 tree not present")


class TestTheCounts:
    def test_rel19_is_exactly_33(self) -> None:
        assert len(native_names(MOD, STATES["Rel-19"])) == 33

    def test_every_attribute_is_a_leaf_or_leaf_list_with_evidence(self) -> None:
        for name, a in native_attributes(MOD, STATES["Rel-19"]).items():
            assert a.statement in ("leaf", "leaf-list")
            assert a.file == f"{MOD}.yang" and a.line > 0
            assert a.evidence.startswith(f"{MOD}.yang:")


class TestEraState:
    """Rel-16 is checked only when the era extraction is available in the scratch dir."""

    def _era_dir(self):
        import os
        sp = os.environ.get("TELEABSTAIN_ERA_DIR")
        return Path(sp) if sp and Path(sp).is_dir() else None

    def test_rel16_is_exactly_20_when_available(self) -> None:
        d = self._era_dir()
        if d is None:
            pytest.skip("era state not extracted; set TELEABSTAIN_ERA_DIR")
        assert len(native_names(MOD, d)) == 20
