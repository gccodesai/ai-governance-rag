## Offline evals (no LLM)

Golden set: 35 answerable questions, 14 adversarial prompts.

### Retrieval

| Mode | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| bm25 | 46% | 74% | 83% | 0.60 |
| dense | 66% | 80% | 89% | 0.74 |
| hybrid | 63% | 83% | 89% | 0.72 |

### Input guardrails

- Prompt-injection detection: **100%**
- PII redaction: **100%**
- False positives on real questions: **0%**
