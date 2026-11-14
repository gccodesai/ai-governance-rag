"""Input and output guardrails.

Deterministic, explainable checks: no extra model calls, no cost, easy to test.
Each check returns a GuardrailResult so the decision is visible in the UI,
recorded in the trace, and measurable in the evals.

Input  : length, prompt injection, PII (redacted, not blocked), topic relevance
Output : abstention, citation validity, system-prompt leakage, PII in the answer
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .prompts import ABSTAIN_TOKEN, LEAK_CANARIES

ALLOW, REDACT, WARN, BLOCK = "allow", "redact", "warn", "block"


@dataclass
class GuardrailResult:
    name: str
    action: str           # allow | redact | warn | block
    detail: str = ""

    @property
    def blocked(self) -> bool:
        return self.action == BLOCK

    def as_dict(self) -> dict:
        return {"name": self.name, "action": self.action, "detail": self.detail}


# ---------------------------------------------------------------- prompt injection

INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override)\b.{0,40}\b(instruction|rule|prompt|guideline|direction)s?\b",
    r"\b(reveal|show|print|repeat|leak|tell me)\b.{0,40}\b(system|hidden|initial|original)\s+(prompt|instruction|message)s?\b",
    r"\byour (system )?(prompt|instructions)\b",
    r"\byou are (now|no longer)\b",
    r"\b(developer|god|admin|jailbreak|dan)\s+mode\b",
    r"\bpretend (to be|you are)\b",
    r"\bact as (an? )?(unrestricted|unfiltered|different)\b",
    r"\bdo anything now\b",
    r"</?\s*(system|assistant|instructions?)\s*>",
    r"\bnew instructions?\s*:",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def check_injection(text: str) -> GuardrailResult:
    for pattern in _INJECTION_RE:
        m = pattern.search(text)
        if m:
            return GuardrailResult("prompt_injection", BLOCK, f"matched: '{m.group(0)[:60]}'")
    return GuardrailResult("prompt_injection", ALLOW)


# ---------------------------------------------------------------- PII

PII_PATTERNS = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CARD": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "PHONE": re.compile(r"(?<!\w)(?:\+?\d{1,3}[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b"),
}


def _luhn_ok(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Replace PII with [EMAIL] etc. Returns (clean_text, kinds_found)."""
    found = []

    def make_sub(kind):
        def sub(m):
            if kind == "CARD" and not _luhn_ok(m.group(0)):
                return m.group(0)
            found.append(kind)
            return f"[{kind}]"
        return sub

    for kind in ("EMAIL", "SSN", "CARD", "PHONE"):
        text = PII_PATTERNS[kind].sub(make_sub(kind), text)
    return text, sorted(set(found))


def check_pii(text: str) -> tuple[str, GuardrailResult]:
    clean, kinds = redact_pii(text)
    if kinds:
        return clean, GuardrailResult("pii", REDACT, "redacted " + ", ".join(kinds))
    return text, GuardrailResult("pii", ALLOW)


# ---------------------------------------------------------------- input pipeline

def check_length(text: str, max_chars: int) -> GuardrailResult:
    if not text.strip():
        return GuardrailResult("length", BLOCK, "empty question")
    if len(text) > max_chars:
        return GuardrailResult("length", BLOCK, f"{len(text)} chars > limit {max_chars}")
    return GuardrailResult("length", ALLOW)


def check_relevance(keyword_score: float, dense_score: float | None,
                    min_keyword: float, min_dense: float) -> GuardrailResult:
    """Cheap off-topic gate, run before any LLM call.

    Uses semantic similarity of the best chunk when embeddings are available
    (keyword scores can't tell "weather in Paris" from a real question), and
    falls back to a low keyword-overlap floor otherwise. It is deliberately
    conservative: borderline questions go through, and the LLM's own
    "not in sources" abstention is the second line of defence. Thresholds are
    calibrated with `python -m evals.run_evals --calibrate`.
    """
    detail = f"bm25={keyword_score:.2f}" + (f", dense={dense_score:.3f}" if dense_score is not None else "")
    passed = dense_score >= min_dense if dense_score is not None else keyword_score >= min_keyword
    if passed:
        return GuardrailResult("relevance", ALLOW, detail)
    return GuardrailResult("relevance", BLOCK, "off-topic (" + detail + ")")


# ---------------------------------------------------------------- output checks

CITATION_RE = re.compile(r"\[(\d+)\]")


def check_output(answer: str, n_sources: int) -> tuple[str, list[GuardrailResult]]:
    """Validate the model's answer. Returns (possibly modified answer, results)."""
    results = []

    # 1. System prompt leakage: block outright.
    lowered = answer.lower()
    for canary in LEAK_CANARIES:
        if canary.lower() in lowered:
            results.append(GuardrailResult("prompt_leak", BLOCK, "answer echoed the system prompt"))
            return answer, results
    results.append(GuardrailResult("prompt_leak", ALLOW))

    # 2. Abstention: the reply is just the token. If the model wrote a real answer and
    #    also appended the token, keep the answer and drop the token.
    if is_abstention(answer):
        results.append(GuardrailResult("abstained", WARN, "sources do not cover the question"))
        return ABSTAIN_TOKEN, results
    answer = answer.replace(ABSTAIN_TOKEN, "").strip()

    # 3. Citations must exist and point at real sources.
    cited = [int(n) for n in CITATION_RE.findall(answer)]
    invalid = sorted({n for n in cited if not 1 <= n <= n_sources})
    if invalid:
        answer = CITATION_RE.sub(lambda m: "" if int(m.group(1)) in invalid else m.group(0), answer)
        results.append(GuardrailResult("citations", WARN, f"removed invalid citations {invalid}"))
    elif not cited:
        results.append(GuardrailResult("citations", WARN, "answer has no citations; may be ungrounded"))
    else:
        results.append(GuardrailResult("citations", ALLOW, f"cites {sorted(set(cited))}"))

    # 4. Never echo PII back.
    answer, pii = check_pii(answer)
    if pii.action != ALLOW:
        pii.name = "pii_output"
        results.append(pii)
    return answer, results


def is_abstention(answer: str) -> bool:
    rest = re.sub(r"[\W_]+", " ", answer.replace(ABSTAIN_TOKEN, "")).strip()
    return ABSTAIN_TOKEN in answer and len(rest) < 25


def cited_sources(answer: str) -> list[int]:
    return sorted({int(n) for n in CITATION_RE.findall(answer)})
