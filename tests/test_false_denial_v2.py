from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from teleabstain.false_denial_v2 import classify  # noqa: E402

MOD = "_3gpp-nr-nrm-nrcelldu"
Q = ("In the 3GPP TS 28.541 NRM (Rel-19), does the leaf '{l}' exist as an attribute of the "
     "NRCellDU class itself (not of any object contained under it)?")


def c(text, leaf="nRPCI", prompt=None):
    return classify(text, prompt or Q.format(l=leaf), leaf, MOD)


def test_leading_no_and_its_close_variants():
    for t in ("No.", "Answer: No, it is not.", "The answer is no.", "False, the 'nRPCI' is absent."):
        v = c(t)
        assert v.denial and v.leading == "no", t


def test_direct_statements_on_the_asked_element_and_class():
    assert c("The leaf 'nRPCI' is not an attribute of the NRCellDU class itself.").rules == ["direct"]
    assert c("The 'nRPCI' is not present.").denial
    assert c("The 'nRPCI' is mentioned. Therefore, it does not exist as an attribute of the NRCellDU class.").denial
    assert c("The NRCellDU class does not include 'nRPCI'.").denial


def test_a_statement_about_another_object_is_not_a_direct_denial():
    v = c("The 'nRPCI' value is not an attribute of the PdcchConfig object.")
    assert "direct" not in v.rules


def test_exclusions_deferral_echo_values_unrelated():
    for t in ("If 'nRPCI' does not exist in the NRCellDU class, you would need to consult the specification.",
              "Yes, 'nRPCI' is an attribute of the NRCellDU class itself (not of any object contained under it).",
              "The value range of 'nRPCI' is not specified in the NRCellDU class.",
              "I do not have access to the specification for 'nRPCI'."):
        assert not c(t).denial, t


def test_leading_yes_decides_and_the_case_is_listed():
    v = c("Yes, 'nRPCI' exists. Note that 'nRPCI' is not present in the NRCellDU class in earlier releases.")
    assert not v.denial and v.contradiction.startswith("leading yes")


def test_a_leading_no_that_repeats_the_question_still_answers_no():
    p = Q.format(l="ssbFrequency")
    t = ("No, in the 3GPP TS 28.541 NRM (Rel-19), 'ssbFrequency' does not exist as an attribute of "
         "the NRCellDU class itself (not of any object contained under it).")
    v = classify(t, p, "ssbFrequency", MOD)
    assert v.leading == "no" and v.denial
