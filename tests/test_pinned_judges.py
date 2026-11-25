from __future__ import annotations


import pytest

from teleabstain.judges import THINK_TRIGGER, UnpinnedJudge, build_judge, template_names_thinking


class _Tok:
    """A tokenizer stand-in: eos, an end-of-turn map, a template, and a call record."""

    def __init__(self, eos=1, eot_map=None, template="plain", unk=0):
        self.eos_token_id, self.unk_token_id = eos, unk
        self._eot = eot_map or {}
        self.chat_template = template
        self.calls = []

    def convert_tokens_to_ids(self, tok):
        return self._eot.get(tok, self.unk_token_id)

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt, **kw):
        self.calls.append(kw)
        return msgs[0]["content"]


class TestEndOfTurn:
    def test_an_end_of_turn_equal_to_eos_adds_nothing(self):
        j = build_judge("qwen3-14b", end_of_turn="<eot>", thinking_switch=True)
        assert j.configure_stops(_Tok(eos=5, eot_map={"<eot>": 5})) == [5]

    def test_the_generation_configs_own_stops_are_kept_not_replaced(self):
        """Passing eos_token_id explicitly replaces the model's own stop list; keep it."""
        class Cfg:
            eos_token_id = [151645, 151643]

        class M:
            generation_config = Cfg()
        j = build_judge("qwen3-14b", end_of_turn="<eot>", thinking_switch=True)
        assert j.configure_stops(_Tok(eos=151645, eot_map={"<eot>": 151645}), M()) == \
            [151645, 151643]
        Cfg.eos_token_id = 1
        assert j.configure_stops(_Tok(eos=1, eot_map={"<eot>": 106}), M()) == [1, 106]

    def test_an_end_of_turn_the_tokenizer_does_not_know_is_refused(self):
        j = build_judge("qwen3-14b", end_of_turn="<nope>", thinking_switch=True)
        with pytest.raises(UnpinnedJudge):
            j.configure_stops(_Tok(eos=5))


class TestThinking:
    def test_the_switch_is_detected_from_the_template_text(self):
        assert template_names_thinking(_Tok(template="{% if enable_thinking %}x{% endif %}"))
        assert not template_names_thinking(_Tok(template="plain"))
        assert not template_names_thinking(_Tok(template=None))

    def _call(self, template, thinking_switch, text_hook=None):
        tok = _Tok(template=template)
        if text_hook:
            tok.apply_chat_template = text_hook
        j = build_judge("qwen3-14b", end_of_turn="<eot>", thinking_switch=thinking_switch,
                        device="cpu")
        j._model, j._tokenizer, j._stop_ids = object(), tok, [1]
        j._generate = lambda text, cap: ('{"abstained": false}', "end_turn")
        return j, tok, j._one_call("abstained", "Q?", "R.", None)

    def test_enable_thinking_false_is_passed_iff_the_template_names_it(self):
        _, tok, out = self._call("... enable_thinking ...", True)
        assert tok.calls == [{"enable_thinking": False}] and out["parse_ok"]
        _, tok, _ = self._call("plain template", False)
        assert tok.calls == [{}]

    def test_a_prompt_carrying_the_think_trigger_is_refused(self):
        with pytest.raises(UnpinnedJudge):
            self._call("plain", False,
                       text_hook=lambda m, tokenize, add_generation_prompt, **kw:
                       THINK_TRIGGER + m[0]["content"])

    def test_the_template_must_agree_with_the_recorded_switch(self):
        j = build_judge("qwen3-14b", end_of_turn="<eot>", thinking_switch=False)
        with pytest.raises(UnpinnedJudge):
            j.check_template(_Tok(template="enable_thinking"))
        j.check_template(_Tok(template="plain"))
