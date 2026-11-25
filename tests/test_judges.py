
from __future__ import annotations


import pytest

from teleabstain.judges import (
    output_is_complete, CALL_TYPES, JUDGE_REGISTRY, assemble, build_judge, build_prompt,
    parse_field, parse_verdict,
)


# --- the one-line swap -------------------------------------------------------------------


def test_every_named_judge_builds_or_needs_credentials() -> None:
    """Building must not require the model; loading is deferred until first use."""
    for name in JUDGE_REGISTRY:
        judge = build_judge(name)
        assert judge.judge_id, name


# --- the rubric asks for both verdicts ------------------------------------------------------


def test_the_rubric_prompt_asks_for_abstention_and_coherence() -> None:
    filled = "".join(build_prompt(c, "q", "r") for c in CALL_TYPES)
    assert "abstained" in filled and "coherent" in filled
    assert "fluent-sounding but semantically empty" in filled
    assert "Do not answer the question yourself" in filled


# --- parsing --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply,abstained,coherent",
    [
        ('{"abstained": true, "coherent": true}', True, True),
        ('{"abstained": false, "coherent": false}', False, False),
        ('Here you go:\n```json\n{"abstained": true, "coherent": false}\n```', True, False),
        ('  {"abstained": false, "coherent": true}  ', False, True),
    ],
)
def test_verdicts_parse(reply: str, abstained: bool, coherent: bool) -> None:
    verdict = parse_verdict(reply)
    assert verdict.parse_ok
    assert verdict.abstained is abstained
    assert verdict.coherent is coherent


def test_an_unparseable_reply_is_recorded_not_guessed() -> None:
    verdict = parse_verdict("I think the model probably declined, honestly.")
    assert verdict.parse_ok is False
    assert verdict.abstained is False, "the conservative direction: never invent an abstention"
    assert verdict.raw


def test_a_missing_coherence_key_is_none_not_false() -> None:
    verdict = parse_verdict('{"abstained": true}')
    assert verdict.parse_ok and verdict.abstained is True and verdict.coherent is None


# --- rubric versioning ------------------------------------------------------------------


def test_v42_each_prompt_asks_for_exactly_one_field() -> None:
    for call in CALL_TYPES:
        text = build_prompt(call, "q", "r")
        assert f'"{call}": true|false' in text
        assert text.count("true|false") == 1, f"{call} prompt asks for more than one field"


def test_v42_the_hallucination_prompt_carries_its_clause_and_nothing_else() -> None:
    text = " ".join(build_prompt("hallucinated", "q", "r").split())
    assert "ONLY for a SPECIFIC, CHECKABLE, INVENTED FACT" in text
    assert "vague, near-tautological filler" in text
    assert "A false DENIAL is NOT hallucination" in text
    assert "counted separately, by machine, against the schema" in text
    assert "Broken text still counts" in text
    assert "J. FABRICATION INSIDE BROKEN TEXT." in build_prompt("hallucinated", "q", "r")
    # it must not know the response was judged for anything else
    assert "abstain" not in text.lower()
    assert "declin" not in text.lower()


def test_v42_the_coherence_prompt_is_the_cross_check_only() -> None:
    text = " ".join(build_prompt("coherent", "q", "r").split())
    assert "Truncation alone is not incoherence" in text
    assert "fluent-sounding but semantically empty" in text
    assert "abstain" not in text.lower()
    assert "hallucinat" not in text.lower()


def test_v42_the_answer_key_reaches_only_the_hallucination_call() -> None:
    """Structural, not instructed: the other two prompts have no slot for it."""
    fact = "Leaf arfcn exists, type uint32."
    assert fact in build_prompt("hallucinated", "q", "r", fact)
    assert fact not in build_prompt("abstained", "q", "r", fact)
    assert fact not in build_prompt("coherent", "q", "r", fact)
    assert "PROVED FACT" not in build_prompt("hallucinated", "q", "r", None)


def test_v42_an_unanswered_call_makes_the_whole_verdict_unparsed() -> None:
    """A verdict missing one of its three answers is not a verdict.

    Treating it as one would substitute a default for whichever question went unanswered,
    which is exactly the silent-default failure the panel's sit-out rule exists to prevent.
    """
    def call(name, value, parse_ok=True, stop="end_turn"):
        return {"call": name, "value": value, "parse_ok": parse_ok, "stop_reason": stop,
                "attempts": 1, "retried": False, "raw": "x"}

    good = assemble([call("coherent", True), call("abstained", False),
                     call("hallucinated", True)])
    assert good.parse_ok and good.coherent is True and good.hallucinated is True
    assert good.stop_reason == "end_turn" and len(good.calls) == 3

    partial = assemble([call("coherent", True), call("abstained", None, parse_ok=False),
                        call("hallucinated", True)])
    assert not partial.parse_ok, "one unanswered call must fail the whole verdict"


def test_v42_a_capped_or_refused_call_surfaces_in_the_assembled_stop_reason() -> None:
    def call(name, value, stop):
        return {"call": name, "value": value, "parse_ok": value is not None,
                "stop_reason": stop, "attempts": 2, "retried": True, "raw": ""}

    v = assemble([call("coherent", True, "end_turn"), call("abstained", None, "max_new_tokens"),
                  call("hallucinated", True, "end_turn")])
    assert v.stop_reason == "max_new_tokens" and v.retried and v.attempts == 2
    r = assemble([call("coherent", True, "end_turn"), call("abstained", True, "end_turn"),
                  call("hallucinated", None, "refusal")])
    assert r.stop_reason == "refusal"


def test_v42_parse_field_is_strict_about_which_question_was_answered() -> None:
    assert parse_field('{"abstained": true}', "abstained") is True
    assert parse_field('```json\n{"coherent": false}\n```', "coherent") is False
    assert parse_field('Sure. {"hallucinated": true}', "hallucinated") is True
    # answering a different question than the one asked is a parse failure, not a verdict
    assert parse_field('{"hallucinated": true}', "abstained") is None
    assert parse_field("no json here", "coherent") is None


class TestOutputIsComplete:
    def test_missing_file_is_not_complete(self, tmp_path) -> None:
        assert not output_is_complete(tmp_path / "nope.jsonl")

    def test_zero_byte_file_is_not_complete(self, tmp_path) -> None:
        p = tmp_path / "judge.jsonl"
        p.write_text("", encoding="utf-8")
        assert not output_is_complete(p)

    def test_whitespace_only_file_is_not_complete(self, tmp_path) -> None:
        p = tmp_path / "judge.jsonl"
        p.write_text("\n\n   \n", encoding="utf-8")
        assert not output_is_complete(p)

    def test_a_directory_is_not_complete(self, tmp_path) -> None:
        """`is_file()` is false for a directory, and reading one raises - neither may crash."""
        d = tmp_path / "judge.jsonl"
        d.mkdir()
        assert not output_is_complete(d)
