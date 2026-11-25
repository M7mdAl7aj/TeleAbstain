from __future__ import annotations

from pathlib import Path

import pytest

from teleabstain.constraints import resolve_constraint
from teleabstain.key_gate import _declared_range, gate_items
from teleabstain.native_attributes import native_attributes

REPO = Path(__file__).resolve().parent.parent
R19 = REPO / "vendor/MnS/yang-models"
MOD = "_3gpp-nr-nrm-nrcelldu"
pytestmark = pytest.mark.skipif(not R19.is_dir(), reason="vendor Rel-19 tree not present")


@pytest.fixture(scope="module")
def nat():
    return native_attributes(MOD, R19)


class TestResolution:
    def test_a_bare_uint32_has_no_machine_readable_range(self, nat) -> None:
        """`rimRSMonitoringOccasionStartingOffset` states its bound in prose, not in YANG."""
        kind, _, _ = _declared_range("rimRSMonitoringOccasionStartingOffset", nat, R19)
        assert kind is None

    def test_a_real_enumeration_resolves_through_its_typedef(self, nat) -> None:
        kind, spec, ev = _declared_range("administrativeState", nat, R19)
        assert kind == "enum" and "LOCKED" in spec
        assert any(e.endswith(tuple("0123456789")) for e in ev), ev

    def test_the_chain_is_reported_when_it_cannot_resolve(self) -> None:
        c = resolve_constraint("leaf x { type nosuch:Type; }", R19 / f"{MOD}.yang", R19)
        assert c.kind == "cannot-resolve" and c.chain
        assert "CANNOT-RESOLVE" in c.render()


class TestGateCheckEight:
    def _item(self, fam, leaf, value, iid="probe"):
        return {"id": iid, "category": fam, "release": "Rel-19", "kind": "answerable",
                "yang_module": MOD,
                "prompt": f"Set '{leaf}' to {value} in the NRCellDU managed object.",
                "check": {"kind": "value_in_range", "leaf": leaf, "value": value}}

    def _v(self, it):
        return gate_items([it], {"Rel-19": R19}, later_pin="Rel-19")[it["id"]]

    def test_a_dn_typed_leaf_with_a_numeric_probe_is_rejected(self) -> None:
        v = self._v(self._item("out-of-range-value", "nRFrequencyRef", -24))
        assert "range_outside_declared" in v.failed_checks

    def test_an_int32_leaf_with_a_string_probe_is_rejected(self) -> None:
        v = self._v(self._item("out-of-range-control", "bSChannelBwSUL", "REDCAP_1RX"))
        assert "range_outside_declared" in v.failed_checks

    def test_an_unresolvable_constraint_fails_both_families(self) -> None:
        for fam in ("out-of-range-value", "out-of-range-control"):
            v = self._v(self._item(fam, "rimRSMonitoringOccasionStartingOffset", 0))
            assert "range_outside_declared" in v.failed_checks, fam

    def test_a_real_enum_probe_inside_passes_control_and_fails_value(self, nat) -> None:
        kind, spec, _ = _declared_range("administrativeState", nat, R19)
        inside = spec[0]
        assert self._v(self._item("out-of-range-control", "administrativeState", inside)).passed
        v = self._v(self._item("out-of-range-value", "administrativeState", inside))
        assert "range_outside_declared" in v.failed_checks

    def test_a_token_outside_the_enum_passes_the_value_family(self) -> None:
        it = self._item("out-of-range-value", "administrativeState", "LOCKED_EXTENDED")
        assert self._v(it).passed
