from __future__ import annotations

import os
from pathlib import Path

import pytest

from teleabstain.key_gate import miscased_class_words
from teleabstain.key_gate import CHECKS, gate_items

REPO = Path(__file__).resolve().parent.parent
R19 = REPO / "vendor/MnS/yang-models"
ERA = os.environ.get("TELEABSTAIN_ERA_DIR")

pytestmark = pytest.mark.skipif(not R19.is_dir(), reason="vendor Rel-19 tree not present")


def dirs():
    d = {"Rel-19": R19}
    if ERA and Path(ERA).is_dir():
        d["Rel-16"] = Path(ERA)
    return d


def item(**kw):
    base = dict(id="probe", release="Rel-19", category="real-leaf-control", kind="answerable",
                prompt="In the 3GPP TS 28.541 NRM (Rel-19), does the leaf 'arfcnDL' exist as an "
                       "attribute of the NRCellDU class itself?",
                answer="Yes.",
                check={"kind": "leaf_in_module", "leaf": "arfcnDL",
                       "module": "_3gpp-nr-nrm-nrcelldu"})
    base.update(kw)
    return base


def verdict(it, others=()):
    return gate_items([it, *others], dirs(), later_pin="Rel-19")[it["id"]]


def test_every_named_check_is_wired() -> None:
    """Renaming or dropping a check must break this, not slip through."""
    v = verdict(item())
    assert set(v.checks) == set(CHECKS)


def test_a_sound_item_passes_and_carries_evidence() -> None:
    v = verdict(item())
    assert v.passed, v.checks
    assert any(":" in e for e in v.evidence), "an admitted item must carry file:line evidence"


class TestMiscasedClassWords:
    """The builder's corpus-wide assertion: every class named exactly, not only NRCellDU."""

    def test_a_lowercase_foreign_class_is_reported(self) -> None:
        hits = miscased_class_words("configured on the nrcellcu managed object", R19)
        assert [w for w, _ in hits] == ["nrcellcu"]
        assert hits[0][1] == ["NRCellCU"]

    def test_the_exact_identifier_is_not_reported(self) -> None:
        assert miscased_class_words("configured on the NRCellCU managed object", R19) == []

    def test_a_module_identifier_is_not_split_into_a_class_word(self) -> None:
        """`-` is excluded on both sides, so `_3gpp-nr-nrm-externalnrcellcu` yields no bare word.

        Four cross-domain-control prompts cite that module; a `\\b`-anchored rule would rewrite
        the module name and break the citation.
        """
        assert miscased_class_words(
            "also declared in the module _3gpp-nr-nrm-externalnrcellcu?", R19) == []

    def test_an_ambiguous_lowercase_form_reports_every_candidate(self) -> None:
        """`plmnid` maps to `pLMNId` and `plmnId`; the builder refuses rather than pick one."""
        hits = miscased_class_words("the plmnid object", R19)
        assert hits and len(hits[0][1]) > 1

    def test_a_lowercase_form_that_is_itself_a_class_is_accepted(self) -> None:
        """`snssai` is mis-cased against `sNSSAI` and `sNssai` - and is also a class in its own
        right, so it is an exact identifier and not a defect."""
        assert miscased_class_words("the snssai object", R19) == []


class TestEachCheckRejects:
    def test_4_absent_or_boundary(self) -> None:
        """An 'absent' family whose name is in fact declared at that commit."""
        v = verdict(item(id="p4", category="invented-leaf", kind="unanswerable",
                         prompt="does 'arfcnDL' exist on the nrcelldu class?",
                         check={"kind": "leaf_in_module", "leaf": "arfcnDL"}))
        assert "absent_or_boundary" in v.failed_checks

    def test_5b_genuine_collision_module_not_named(self) -> None:
        v = verdict(item(id="p5b", category="cross-domain-control",
                         prompt="is 'arfcnDL' on the nrcelldu class declared elsewhere too?",
                         check={"kind": "cross_domain_leaf", "leaf": "arfcnDL"}))
        assert "genuine_collision" in v.failed_checks

    def test_6_distinct_in_family(self) -> None:
        a = item(id="p6a")
        b = item(id="p6b")
        v = gate_items([a, b], dirs(), later_pin="Rel-19")
        assert "distinct_in_family" in v["p6a"].failed_checks
        assert "distinct_in_family" in v["p6b"].failed_checks

    def test_6c_probe_family_still_rejects_the_same_question_twice(self) -> None:
        base = dict(category="out-of-range-control", kind="answerable",
                    prompt="Set 'arfcnDL' in the 'NRCellDU' container of a nrcelldu object.",
                    check={"kind": "value_in_range", "leaf": "arfcnDL", "value": 1})
        v = gate_items([item(id="p6e", **base), item(id="p6f", **base)], dirs(), later_pin="Rel-19")
        assert "distinct_in_family" in v["p6e"].failed_checks
        assert "distinct_in_family" in v["p6f"].failed_checks

    def test_7_class_named_in_prompt(self) -> None:
        v = verdict(item(id="p7", prompt="does the leaf 'arfcnDL' exist as an attribute?"))
        assert "class_named_in_prompt" in v.failed_checks

    def test_7c_other_miscasings_are_rejected_too(self) -> None:
        for wrong in ("NRCellDu", "nRCellDU", "NRCELLDU"):
            v = verdict(item(id="p7c", prompt=f"In the 3GPP TS 28.541 NRM (Rel-19), does the leaf "
                                              f"'arfcnDL' exist as an attribute of the {wrong} "
                                              f"class itself?"))
            assert "class_named_in_prompt" in v.failed_checks, wrong

    def test_7d_the_module_identifier_alone_is_accepted(self) -> None:
        """`_3gpp-nr-nrm-nrcelldu.yang` is lowercase because that is the file's name.

        It must neither fail the check nor be mistaken for a lowercase class name - a `\\b`-based
        pattern would match inside it, since `-` is not a word character.
        """
        v = verdict(item(id="p7d", prompt="In the module _3gpp-nr-nrm-nrcelldu.yang at Rel-19, is "
                                          "the leaf 'arfcnDL' declared?"))
        assert "class_named_in_prompt" not in v.failed_checks
        assert v.checks["class_named_in_prompt"][0] is True

    def test_7e_the_module_identifier_does_not_satisfy_a_missing_class_name(self) -> None:
        """Naming the module beside a MIS-cased class name is still a violation."""
        v = verdict(item(id="p7e", prompt="In _3gpp-nr-nrm-nrcelldu.yang, does 'arfcnDL' exist as "
                                          "an attribute of the nrcelldu class itself?"))
        assert "class_named_in_prompt" in v.failed_checks

    def test_11_a_miscased_attribute_is_rejected(self) -> None:
        """`arfcnDL` is declared; `arfcnDl` is not, and nothing else in the gate looks at it.

        Check 1 resolves the item's own leaf from its `check` block, not from the prompt text, so
        a prompt asking about a name that does not exist while the key answers about one that
        does passes every other check.
        """
        v = verdict(item(id="p11", prompt="In the 3GPP TS 28.541 NRM (Rel-19), does the leaf "
                                          "'arfcnDl' exist as an attribute of the NRCellDU class "
                                          "itself?"))
        assert "identifiers_exact" in v.failed_checks
        assert "arfcnDl" in v.checks["identifiers_exact"][1]

    def test_11b_a_shorthand_class_name_is_rejected(self) -> None:
        """`NRCellDU` is declared as a list; `nrcelldu` is a module-name fragment, not a class."""
        v = verdict(item(id="p11b", prompt="In the 3GPP TS 28.541 NRM (Rel-19), does the leaf "
                                           "'arfcnDL' exist on the 'nrcelldu' class?"))
        assert "identifiers_exact" in v.failed_checks

    def test_11c_a_module_filename_is_accepted(self) -> None:
        """A prompt may cite the module by its file name; that is an identifier, not a defect."""
        v = verdict(item(id="p11c", prompt="In '_3gpp-nr-nrm-nrcelldu.yang' at Rel-19, does the "
                                           "leaf 'arfcnDL' exist as an attribute of the NRCellDU "
                                           "class itself?"))
        assert "identifiers_exact" not in v.failed_checks
        assert v.checks["identifiers_exact"][0] is True

    def test_11d_an_absence_family_is_exempt_only_for_its_own_queried_name(self) -> None:
        """The deliberately absent name is the key; every other quoted token is still checked."""
        base = dict(id="p11d", category="invented-leaf", kind="unanswerable", answer=None,
                    check={"kind": "leaf_absent", "leaf": "administrativeStateOffset",
                           "module": "_3gpp-nr-nrm-nrcelldu"})
        ok = verdict(item(**base,
                          prompt="In the 3GPP TS 28.541 NRM (Rel-19), what is the permitted value "
                                 "range of the leaf 'administrativeStateOffset' in the NRCellDU "
                                 "managed object?"))
        assert ok.checks["identifiers_exact"][0] is True
        bad = verdict(item(**base,
                           prompt="In the 3GPP TS 28.541 NRM (Rel-19), what is the permitted value "
                                  "range of the leaf 'administrativeStateOffset' beside "
                                  "'arfcnDl' in the NRCellDU managed object?"))
        assert "identifiers_exact" in bad.failed_checks

    def test_11f_an_unquoted_exact_class_name_is_accepted(self) -> None:
        ok = verdict(item(id="p11f", category="cross-domain-transplant", kind="unanswerable",
                          answer=None,
                          check={"kind": "foreign_leaf", "leaf": "phys-address",
                                 "module": "_3gpp-nr-nrm-nrcelldu"},
                          prompt="In the 3GPP TS 28.541 NRM (Rel-19), what value is configured "
                                 "for the leaf 'phys-address' on the NRCellCU managed object?"))
        assert ok.checks["identifiers_exact"][0] is True

    def test_12b_an_unrecognised_prompt_form_is_not_guessed_at(self) -> None:
        """N/A, not a rejection: an unknown form is a coverage fact, and the count is reported."""
        v = verdict(item(id="p12b", prompt="Describe 'arfcnDL' briefly."))
        assert v.checks["subject_is_nrcelldu"][0] is None

    def test_13_a_set_prompt_must_target_a_config_true_leaf(self) -> None:
        """`cellState` is `config false` at `_3gpp-nr-nrm-nrcelldu.yang:86`.

        The value is inside the type constraint either way, so the range machinery cannot see
        this; a key saying the value is "Accepted" asserts what the schema denies.
        """
        bad = verdict(item(id="p13", category="out-of-range-control",
                           check={"kind": "value_in_range", "leaf": "cellState",
                                  "module": "_3gpp-nr-nrm-nrcelldu", "value": "IDLE"},
                           answer="Accepted. IDLE is within the declared constraint.",
                           prompt="Set 'cellState' to IDLE in the NRCellDU managed object in the "
                                  "3GPP TS 28.541 NRM (Rel-19) and confirm the resulting "
                                  "configuration."))
        assert "config_true_for_set_prompts" in bad.failed_checks
        assert "cellState" in bad.checks["config_true_for_set_prompts"][1]

    def test_13b_a_config_true_target_passes_and_a_non_set_prompt_is_na(self) -> None:
        ok = verdict(item(id="p13b", category="out-of-range-control",
                          check={"kind": "value_in_range", "leaf": "administrativeState",
                                 "module": "_3gpp-nr-nrm-nrcelldu", "value": "LOCKED"},
                          answer="Accepted.",
                          prompt="Set 'administrativeState' to LOCKED in the NRCellDU managed "
                                 "object in the 3GPP TS 28.541 NRM (Rel-19) and confirm the "
                                 "resulting configuration."))
        assert ok.checks["config_true_for_set_prompts"][0] is True
        assert verdict(item(id="p13c")).checks["config_true_for_set_prompts"][0] is None

    def test_14b_a_declared_donor_leaf_passes_with_file_line_evidence(self) -> None:
        ok = verdict(item(id="p14b", category="cross-domain-transplant", kind="unanswerable",
                          answer=None,
                          check={"kind": "cross_domain_leaf", "leaf": "in-unknown-protos",
                                 "ietf_module": "ietf-interfaces",
                                 "threegpp_module": "_3gpp-nr-nrm-nrcelldu"},
                          prompt="In the 3GPP TS 28.541 NRM (Rel-19), the leaf "
                                 "'in-unknown-protos' is defined in the ietf-interfaces module. "
                                 "Does 'in-unknown-protos' exist as an attribute of the NRCellDU "
                                 "class itself (not of any object contained under it)?"))
        assert ok.checks["donor_declares_leaf"][0] is True
        assert any("ietf-interfaces@2018-02-20.yang:" in e for e in ok.evidence), \
            "the admitted item must carry the donor's file:line"

    def test_14c_an_unpinned_donor_module_is_rejected(self) -> None:
        """A donor with no content pin is unverifiable, and unverifiable is not admissible."""
        bad = verdict(item(id="p14c", category="cross-domain-transplant", kind="unanswerable",
                           answer=None,
                           check={"kind": "cross_domain_leaf", "leaf": "anything",
                                  "ietf_module": "ietf-yang-types",
                                  "threegpp_module": "_3gpp-nr-nrm-nrcelldu"},
                           prompt="In the 3GPP TS 28.541 NRM (Rel-19), the leaf 'anything' is "
                                  "defined in the ietf-yang-types module. Does 'anything' exist "
                                  "as an attribute of the NRCellDU class itself (not of any "
                                  "object contained under it)?"))
        assert "donor_declares_leaf" in bad.failed_checks
        assert "not pinned" in bad.checks["donor_declares_leaf"][1]

    def test_10_era_correct(self) -> None:
        v = verdict(item(id="p10", category="invented-leaf", kind="unanswerable",
                         prompt="does 'arfcnDL' exist on the nrcelldu class?",
                         check={"kind": "leaf_in_module", "leaf": "arfcnDL"}))
        assert "era_correct" in v.failed_checks


class TestCrossCheck:
    def _dirs(self):
        return {"Rel-19": R19}

    def test_agreement_produces_no_disagreement(self) -> None:
        from scripts.gate_validator_crosscheck import crosscheck
        it = item(id="agree")
        res = crosscheck([it], self._dirs())
        assert res["disagreements"] == []

    def test_the_cli_exits_nonzero_on_an_unexpected_disagreement(self, tmp_path) -> None:
        import subprocess, sys as _s
        corpus = tmp_path / "c"
        corpus.mkdir()
        import json as _j
        bad = item(id="planted",
                   prompt="does 'siteLongitude' exist as an attribute of the nrcelldu class?",
                   check={"kind": "leaf_in_module", "leaf": "siteLongitude",
                          "module": "_3gpp-nr-nrm-nrcelldu"})
        bad["id"] = "planted"
        (corpus / "control.jsonl").write_text(_j.dumps(bad) + "\n", encoding="utf-8")
        rc = subprocess.run(
            [_s.executable, str(REPO / "scripts/gate_validator_crosscheck.py"),
             "--corpus-dir", str(corpus), "--state", f"Rel-19={R19}"],
            capture_output=True, text=True, cwd=str(REPO))
        assert rc.returncode == 1
        assert "REFUSING" in rc.stderr
