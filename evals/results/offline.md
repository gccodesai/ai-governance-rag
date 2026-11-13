## Offline evals (no LLM)

Golden set: 35 answerable questions, 14 adversarial prompts.

### Retrieval

| Mode | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| bm25 | 46% | 71% | 83% | 0.60 |

_Dense retrieval unavailable here: ModuleNotFoundError: No module named 'fastembed'_

### Input guardrails

- Prompt-injection detection: **100%**
- PII redaction: **100%**
- False positives on real questions: **0%**
