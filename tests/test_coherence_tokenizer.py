from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from teleabstain.scoring import GATE_VERSION, MAX_UNBROKEN_RUN, classify_coherence, tokenize


class TestRealLongWordsSurvive:
    """A long word is not a loop. The bound exists so real identifiers are never split."""

    YANG_IDENTIFIERS = (
        "intraRatEsDeactivationCandidateCellsLoadParameters",   # 50, the schema's longest
        "intraRatEsActivationCandidateCellsLoadParameters",
        "interRatEsDeactivationCandidateCellParameters",
        "isEventTriggeredGtpUPathMonitoringSupported",
        "rimRSMonitoringOccasionStartingOffset",               # 37, corpus longest
        "eventListForEventTriggeredMeasurement",
    )

    @pytest.mark.parametrize("ident", YANG_IDENTIFIERS)
    def test_a_schema_identifier_is_never_split(self, ident):
        assert len(ident) <= MAX_UNBROKEN_RUN, (
            f"{ident} is {len(ident)} chars, over the bound of {MAX_UNBROKEN_RUN}")
        assert tokenize(ident) == [ident]

    @pytest.mark.parametrize("ident", YANG_IDENTIFIERS)
    def test_a_sentence_of_identifiers_stays_coherent(self, ident):
        text = (f"The attribute {ident} is defined in the information model. "
                f"Its value range is given by the schema, and {ident} is mandatory "
                "for the managed object class described above.")
        assert classify_coherence(text).coherent is True

    def test_a_german_compound_is_not_split(self):
        """44 ASCII characters, under the bound, one token."""
        word = "Donaudampfschifffahrtsgesellschaftskapitaen"
        assert len(word) <= MAX_UNBROKEN_RUN
        assert tokenize(word) == [word]


class TestTheSplitter:
    def test_an_acronym_followed_by_a_word_splits_correctly(self):
        long_run = "HTTPServer" * 8                  # 80 chars, over the bound
        assert "HTTP" in tokenize(long_run) and "Server" in tokenize(long_run)

    def test_the_bound_is_above_every_measured_legitimate_run(self):
        """39 in the corpus, 50 in the schema. The bound must clear both."""
        assert MAX_UNBROKEN_RUN >= 50


class TestGateVersionIsStamped:
    def test_the_constant_exists_and_is_positive(self):
        assert isinstance(GATE_VERSION, int) and GATE_VERSION >= 1

    def test_the_machine_layer_stamps_it(self):
        sys.path.insert(0, str(REPO / "scripts"))
        from run_judges import machine_layer
        row = machine_layer({"item_id": "x", "response": "I cannot answer that.", "bits": 4}, {})
        assert row["gate_version"] == GATE_VERSION


class TestTheRepairIsMonotone:
    def test_one_long_run_is_still_too_short(self):
        """2,000 characters of one non-repeating run is ONE word, not 500."""
        run = "".join(chr(97 + (i * 7 + i // 26) % 26) for i in range(2000))
        v = classify_coherence(run)
        assert v.coherent is False
        assert v.n_words == 1, "length must be judged on the plain view"
