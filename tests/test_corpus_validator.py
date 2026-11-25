from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.corpus_validator import State

REPO = Path(__file__).resolve().parent.parent
R19 = REPO / "vendor" / "MnS"


def _materialise(repo: Path, rev: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    tar = subprocess.run(["git", "-C", str(repo), "archive", rev, "yang-models"],
                         capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(dest)], input=tar, check=True)
    return dest / "yang-models"


@pytest.fixture(scope="module")
def r19(tmp_path_factory) -> State:
    d = _materialise(R19, "Tag_Rel19_SA112", tmp_path_factory.mktemp("r19"))
    return State("Rel-19", d, "24ffa42e")


NRCELLDU = "_3gpp-nr-nrm-nrcelldu"


class TestDeclarationMatching:
    def test_it_finds_a_plain_leaf(self, r19) -> None:
        assert r19.declares(NRCELLDU, "cellState") is not None

    def test_it_finds_a_leaf_list(self, r19) -> None:
        """`leaf-list x {` does not contain `leaf x {`; searching only the latter misses it."""
        ev = r19.declares("_3gpp-common-managed-function", "allowedNFTypes")
        assert ev is not None and ev.kind == "leaf-list"

    def test_it_does_not_match_a_prefix(self, r19) -> None:
        """`arfcn` must not match the declaration of `arfcnDL`."""
        ev = r19.declares(NRCELLDU, "arfcn")
        assert ev is None or ev.kind in ("leaf", "leaf-list")
        assert r19.declares(NRCELLDU, "arfcnD") is None

    def test_evidence_line_points_at_the_declaration(self, r19) -> None:
        ev = r19.declares(NRCELLDU, "cellState")
        line = (r19.dir / f"{NRCELLDU}.yang").read_text().splitlines()[ev.line - 1]
        assert "cellState" in line and line.strip().startswith(("leaf", "leaf-list"))


class TestKnownArrivals:
    """Known ARRIVAL leaves: not in the module file, reachable in its import closure."""

    @pytest.mark.parametrize("leaf,expect_file", [
        ("collectionPeriodM6NR", "_3gpp-common-trace.yang"),
        ("altitude", "_3gpp-common-yang-types.yang"),
        ("allowedNFTypes", "_3gpp-common-managed-function.yang"),
        ("cAGIdList", "_3gpp-5g-common-yang-types.yang"),
    ])
    def test_arrival_at_rel19(self, r19, leaf, expect_file) -> None:
        f = r19.find(NRCELLDU, leaf)
        assert not f.native_says_exists, f"{leaf} should not be native to nrcelldu"
        assert f.subtree_says_exists, f"{leaf} should resolve in the closure"
        assert any(e.file == expect_file for e in f.subtree), \
            f"{leaf}: expected {expect_file}, got {[e.file for e in f.subtree]}"


class TestClosure:
    def test_the_closure_contains_the_module_itself(self, r19) -> None:
        assert NRCELLDU in r19._closure(NRCELLDU)

    def test_the_closure_is_transitive(self, r19) -> None:
        """trace is not imported by nrcelldu directly; it arrives through the chain."""
        c = r19._closure(NRCELLDU)
        assert "_3gpp-common-trace" in c
        assert "_3gpp-common-trace" not in \
            __import__("re").findall(r"^\s*import\s+([A-Za-z0-9_.-]+)",
                                     r19.text[NRCELLDU], __import__("re").M)

    def test_an_unknown_module_has_an_empty_closure(self, r19) -> None:
        assert r19._closure("_no-such-module") == ()


# --------------------------------------------------------------- ranges and mandatory-ness
class TestRangeParsing:
    """The spec forms this corpus actually uses, parsed from strings, no YANG needed."""

    def test_a_simple_interval(self) -> None:
        from tools.corpus_validator import parse_range
        assert parse_range("0..3279165") == [(0.0, 3279165.0)]

    def test_enumerated_singletons(self) -> None:
        from tools.corpus_validator import parse_range
        assert parse_range("1024|2048|5120|10240") == [
            (1024.0, 1024.0), (2048.0, 2048.0), (5120.0, 5120.0), (10240.0, 10240.0)]

    def test_max_becomes_unbounded(self) -> None:
        from tools.corpus_validator import parse_range
        (lo, hi), = parse_range("1..max")
        assert lo == 1.0 and hi == float("inf")

    def test_containment_over_disjoint_singletons(self) -> None:
        from tools.corpus_validator import RangeFinding, parse_range
        r = RangeFinding(bounds=parse_range("1024|2048"))
        assert r.contains(1024) and r.contains(2048)
        assert not r.contains(1500)

    def test_containment_is_None_when_no_range_was_derived(self) -> None:
        from tools.corpus_validator import RangeFinding
        assert RangeFinding().contains(5) is None


class TestBraceMatching:
    """A leaf body nests; a non-greedy regex would stop inside `type uint32 { range ... }`."""

    def test_the_body_spans_the_whole_leaf(self, r19) -> None:
        from tools.corpus_validator import _decl_body
        line, body = _decl_body(r19.text["_3gpp-nr-nrm-nrcelldu"], "cellLocalId")
        assert body.count("{") == body.count("}")
        assert "mandatory true" in body and "range" in body


class TestRangeFromYang:
    """Hand-checked against the YANG text, quoted in the test so a reader can verify it."""

    def test_arfcn_range_matches_the_file(self, r19) -> None:
        """_3gpp-common-trace.yang, `leaf arfcn { type uint32 { range 0..3279165; }`."""
        r = r19.range_of(NRCELLDU, "arfcn")
        assert r.inline and r.spec.strip() == "0..3279165"
        assert r.bounds == [(0.0, 3279165.0)]
        assert r.file == "_3gpp-common-trace.yang"
        assert r.contains(0) and r.contains(3279165)
        assert not r.contains(3279166)

    def test_cellLocalId_range_is_a_quoted_interval(self, r19) -> None:
        """`leaf cellLocalId { ... type int32 { range "0..16383"; } }`."""
        r = r19.range_of(NRCELLDU, "cellLocalId")
        assert r.inline and r.bounds == [(0.0, 16383.0)]
        assert not r.contains(16384)

    def test_a_typedef_range_is_reported_as_not_inline_not_absent(self, r19) -> None:
        """cellState's type is a typedef; its range, if any, is not in the leaf body.

        Reporting that as 'no range' would manufacture a false finding, so the validator
        distinguishes the two.
        """
        r = r19.range_of(NRCELLDU, "cellState")
        assert r.file is not None, "the leaf itself must still be located"
        assert r.inline is False and r.bounds == []
        assert r.type_ref is not None


class TestMandatoryFromYang:
    """Hand-checked: nrcelldu declares `mandatory true` on cellLocalId and nRPCI."""

    @pytest.mark.parametrize("leaf", ["cellLocalId", "nRPCI"])
    def test_known_mandatory_true(self, r19, leaf) -> None:
        m = r19.mandatory_of(NRCELLDU, leaf)
        assert m.mandatory is True, leaf
        assert m.file == "_3gpp-nr-nrm-nrcelldu.yang"

    def test_a_leaf_without_a_mandatory_statement_reports_None(self, r19) -> None:
        """None means 'no statement', which YANG defaults to optional - not the same as false."""
        m = r19.mandatory_of(NRCELLDU, "administrativeState")
        assert m.mandatory is None
        assert m.file is not None

    def test_an_absent_leaf_reports_nothing_located(self, r19) -> None:
        m = r19.mandatory_of(NRCELLDU, "zzzPhantomOffset")
        assert m.file is None and m.mandatory is None


class TestDuplicateLeafNames:
    def test_name_is_declared_twice_with_different_mandatoryness(self, r19) -> None:
        found = r19.mandatories_of(NRCELLDU, "name")
        assert len(found) == 2, [f.file for f in found]
        assert {f.mandatory for f in found} == {True, None}, "the disagreement must be visible"

    def test_the_convenience_accessor_prefers_the_declaration_that_states_mandatory(self, r19) -> None:
        assert r19.mandatory_of(NRCELLDU, "name").mandatory is True

    def test_reportInterval_is_declared_twice_and_one_carries_the_range(self, r19) -> None:
        found = r19.ranges_of(NRCELLDU, "reportInterval")
        assert len(found) == 2
        assert sum(1 for f in found if f.inline) == 1

    def test_the_convenience_accessor_prefers_the_inline_range(self, r19) -> None:
        r = r19.range_of(NRCELLDU, "reportInterval")
        assert r.inline and r.file == "_3gpp-common-trace.yang"
        assert r.contains(120) and not r.contains(121)

    def test_duplicate_declarations_that_AGREE_are_still_reported_separately(self, r19) -> None:
        """cellLocalId is declared twice with the SAME range. Agreement is not uniqueness, and
        the accessor must not hide the duplication from a caller that wants to check it."""
        found = r19.ranges_of(NRCELLDU, "cellLocalId")
        assert len(found) == 2
        assert {f.spec for f in found} == {"0..16383"}, "these two agree"

    def test_an_undeclared_name_yields_no_findings(self, r19) -> None:
        assert r19.ranges_of(NRCELLDU, "zzzPhantomOffset") == []
        assert r19.mandatories_of(NRCELLDU, "zzzPhantomOffset") == []


IETF_ROOT = REPO / "vendor" / "yang" / "standard" / "ietf" / "RFC"


class TestPathScopedResolution:
    """Hand-checked against the YANG text; the lines are quoted so a reader can verify them.

    `_3gpp-common-managed-function.yang`:
        list operations {
          key name;
          uses OperationGrp ;
        }

        grouping OperationGrp {
          leaf name {
            type string;
            mandatory true;      <- line 66
          }
    """

    def test_operations_container_is_located(self, r19) -> None:
        f = r19.find_container(NRCELLDU, "operations")
        assert f is not None
        assert f[0] == "_3gpp-common-managed-function.yang"

    def test_name_resolves_through_the_uses_grouping(self, r19) -> None:
        hits = r19.resolve_in_container(NRCELLDU, "operations", "name")
        assert hits, "name must resolve inside operations"
        assert all("OperationGrp" in h[2] or h[2] == "direct" for h in hits)

    def test_path_scoping_disambiguates_what_the_bare_search_could_not(self, r19) -> None:
        bare = r19.mandatories_of(NRCELLDU, "name")
        assert len({b.mandatory for b in bare}) > 1, "bare search is ambiguous"
        assert r19.mandatory_in_container(NRCELLDU, "operations", "name").mandatory is True


class TestTypedefChain:
    """`_3gpp-common-yang-types.yang`:
        grouping SAPGrp {
          leaf port {
            type inet:port-number;
            mandatory true;
          }

    and `ietf-inet-types`:
        typedef port-number {
          type uint16 {
            range "0..65535";
    """

    def test_port_has_no_inline_range(self, r19) -> None:
        r = r19.range_of(NRCELLDU, "port")
        assert r.inline is False
        assert r.type_ref == "inet:port-number"

    def test_an_unknown_typedef_resolves_to_nothing(self, r19) -> None:
        assert r19.resolve_typedef_range(NRCELLDU, "zzz:no-such-type", IETF_ROOT).file is None
