"""LLM-as-judge: grade an answer for correctness (vs a reference) and faithfulness (vs sources)."""

from __future__ import annotations

import json
import re

JUDGE_PROMPT = """You are grading an AI assistant's answer about the NIST AI Risk Management Framework.

Question:
{question}

Reference answer (written by a human from the source documents):
{reference}

Sources the assistant was given:
{context}

Assistant's answer:
{answer}

Grade two things:
1. correctness (1-5): does the answer convey the key facts of the reference answer?
   5 = all key facts, nothing wrong; 4 = main point right, minor omission;
   3 = partly right; 2 = mostly wrong or missing; 1 = wrong or irrelevant.
   Extra correct detail is fine. Ignore citation markers like [1].
2. faithfulness ("yes", "partial", "no"): is every claim in the answer supported by the sources?

Reply with JSON only, no code fences:
{{"correctness": <1-5>, "faithfulness": "yes|partial|no", "reason": "<one short sentence>"}}"""


def parse_verdict(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"judge did not return JSON: {text[:200]}")
    data = json.loads(m.group(0))
    correctness = int(data["correctness"])
    faithfulness = str(data["faithfulness"]).lower().strip()
    if not 1 <= correctness <= 5 or faithfulness not in ("yes", "partial", "no"):
        raise ValueError(f"judge returned out-of-range values: {data}")
    return {"correctness": correctness, "faithfulness": faithfulness,
            "reason": str(data.get("reason", ""))[:300]}


def judge_answer(client, question: str, reference: str, answer: str, context: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, reference=reference,
                                 context=context[:12000], answer=answer)
    resp = client.chat([{"role": "user", "content": prompt}], max_tokens=2048)
    try:
        return parse_verdict(resp.text)
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        return {"judge_error": str(e)[:300]}
