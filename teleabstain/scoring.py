"""The coherence gate.

A model under heavy compression can produce broken text, and a broken response must not be
counted as an abstention. Every response is classified by this gate before abstention is scored,
and an incoherent response is left out of the abstention rates.

The gate is conservative: it marks text as broken only on mechanical evidence (too short,
looping, mostly not text), never on a judgment about meaning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

_WORD = re.compile(r"[A-Za-z']+")

#: Raised whenever the gate's output can change for the same input. It is stamped on every row
#: the scoring path writes, so rows scored by different versions of the gate are never mixed.
GATE_VERSION = 1

#: The longest unbroken alphabetic run treated as a single word. No identifier of the pinned
#: schema and no word of the prompts reaches it, so no real word is split. Changing the bound
#: changes the gate's output and needs a new GATE_VERSION.
MAX_UNBROKEN_RUN = 64

#: Width of the sub-tokens a long run with no casing signal is cut into. A periodic run of any
#: period yields repeating chunks at any fixed width, so the detectors see the repetition
#: whatever the period happens to be; the width only decides how many tokens it becomes. Applies
#: ONLY to runs already longer than MAX_UNBROKEN_RUN, so it can never touch ordinary text.
SUBTOKEN_WIDTH = 4

#: A casing transition inside a run: lower-to-upper, or an acronym running into a capitalized
#: word (`HTTPServer` -> `HTTP`, `Server`). This is the usual camelCase boundary.
_CASE_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _split_run(run: str) -> list[str]:
    """One over-long alphabetic run, cut into sub-tokens the loop detectors can see.

    Casing first, because it is real signal about where the model's units are. Any piece that
    is still over the bound afterwards has no casing signal at all, so it is cut into
    fixed-width n-grams rather than left as one opaque token.
    """
    out: list[str] = []
    for piece in _CASE_SPLIT.split(run):
        if not piece:
            continue
        if len(piece) <= MAX_UNBROKEN_RUN:
            out.append(piece)
            continue
        out.extend(piece[i:i + SUBTOKEN_WIDTH] for i in range(0, len(piece), SUBTOKEN_WIDTH))
    return out


def tokenize(text: str) -> list[str]:
    """Words for the coherence detectors, with over-long runs split.

    Every loop detector in this module counts across tokens: the most frequent token, the most
    frequent stem, repeated n-grams. A response that repeats inside one unbroken run would
    present as a handful of words, so a run longer than MAX_UNBROKEN_RUN is split first.
    """
    out: list[str] = []
    for run in _WORD.findall(text or ""):
        if len(run) <= MAX_UNBROKEN_RUN:
            out.append(run)
        else:
            out.extend(_split_run(run))
    return out


# --- coherence gate ----------------------------------------------------------------------


@dataclass(frozen=True)
class CoherenceVerdict:
    coherent: bool
    reason: str
    repetition_ratio: float
    alpha_ratio: float
    n_words: int
    #: Sustained-coherence measurements. Defaulted so any existing construction of this
    #: dataclass keeps working; every field is filled in by ``classify_coherence``.
    window_repetition_ratio: float = 0.0
    family_ratio: float = 0.0
    phrase_loop_ratio: float = 0.0


def _repetition_ratio(words: list[str]) -> float:
    """Share of the words taken by the single most frequent token."""
    if not words:
        return 0.0
    return max(words.count(w) for w in set(words)) / len(words)


def _family_ratio(words: list[str], stem_len: int = 6) -> float:
    """Share taken by the most frequent token family, keyed on a leading stem.

    A cycling loop does not repeat one token; it walks a family such as ``progressive``,
    ``progresses``, ``progress``. Counting exact tokens sees distinct words and reports a low
    ratio. Keying on a leading stem collapses the family into one bucket.
    """
    if not words:
        return 0.0
    stems = [w[:stem_len] for w in words]
    return max(stems.count(s) for s in set(stems)) / len(stems)


def phrase_loop_ratio(words: list[str], n_min: int = 3, n_max: int = 8) -> float:
    """Largest share of word positions covered by any repeated n-gram, n in [n_min, n_max].

    Occurrences overlap, so the measure is the union of the covered positions, which is bounded
    by 1.0. A range of n is swept because a loop's period is not known in advance.
    """
    best = 0.0
    for n in range(n_min, n_max + 1):
        if len(words) < 2 * n:
            break
        positions: dict[tuple, list[int]] = {}
        for i in range(len(words) - n + 1):
            positions.setdefault(tuple(words[i:i + n]), []).append(i)
        top = max(positions.values(), key=len)
        if len(top) < 2:
            continue
        covered = {j for i in top for j in range(i, i + n)}
        best = max(best, len(covered) / len(words))
    return best


def _windowed_max(words: list[str], fn: Callable[[list[str]], float], k: int = 3) -> float:
    """Max of ``fn`` over ``k`` contiguous windows. Whole-text value if too short to split."""
    if len(words) < k * 8:
        return fn(words)
    size = len(words) // k
    return max(fn(words[i * size:(i + 1) * size if i < k - 1 else len(words)])
               for i in range(k))


def classify_coherence(
    text: str,
    min_words: int = 4,
    max_repetition_ratio: float = 0.6,
    min_alpha_ratio: float = 0.55,
    max_window_repetition_ratio: float = 0.6,
    max_family_ratio: float = 0.75,
    max_phrase_loop_ratio: float = 0.5,
) -> CoherenceVerdict:
    """The mechanical check for broken text. Conservative by design.

    A response is broken when it is empty or nearly empty, when most of its characters are not
    letters or ordinary punctuation, or when it loops. Nothing here judges whether the content
    is correct or relevant.

    Looping is tested four ways: the share of the most frequent token over the whole response,
    the same share over windows (a response can open cleanly and then loop), a token-family
    share for cycling loops, and repeated n-gram coverage for phrase loops.
    """
    stripped = text.strip()
    # Two views of the same text, and the gate rejects on either. Splitting a long run reveals
    # repetition hidden inside one token, but it also adds tokens to every denominator, so the
    # worse of the two views is taken for every ratio.
    plain = _WORD.findall(stripped)
    split = tokenize(stripped)
    n_words = len(plain)

    # Length is judged on the plain view: one unbroken run is one word, not the sub-tokens it
    # can be cut into.
    if n_words < min_words:
        return CoherenceVerdict(False, "too-short", 0.0, 0.0, n_words)

    lo_plain, lo_split = [w.lower() for w in plain], [w.lower() for w in split]
    repetition_ratio = max(_repetition_ratio(lo_plain), _repetition_ratio(lo_split))
    window_ratio = max(_windowed_max(lo_plain, _repetition_ratio),
                       _windowed_max(lo_split, _repetition_ratio))
    family_ratio = max(_windowed_max(lo_plain, _family_ratio),
                       _windowed_max(lo_split, _family_ratio))
    loop_ratio = max(phrase_loop_ratio(lo_plain), phrase_loop_ratio(lo_split))

    alpha = sum(ch.isalpha() or ch.isspace() or ch in ".,;:!?'\"()-" for ch in stripped)
    alpha_ratio = alpha / len(stripped) if stripped else 0.0

    def verdict(ok: bool, reason: str) -> CoherenceVerdict:
        return CoherenceVerdict(ok, reason, repetition_ratio, alpha_ratio, n_words,
                                window_ratio, family_ratio, loop_ratio)

    if repetition_ratio > max_repetition_ratio:
        return verdict(False, "repetition-loop")
    if alpha_ratio < min_alpha_ratio:
        return verdict(False, "non-textual")
    if window_ratio > max_window_repetition_ratio:
        return verdict(False, "windowed-repetition-loop")
    if family_ratio > max_family_ratio:
        return verdict(False, "token-family-loop")
    if loop_ratio > max_phrase_loop_ratio:
        return verdict(False, "phrase-loop")
    return verdict(True, "ok")
