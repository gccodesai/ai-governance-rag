## Relevance-gate calibration (dense cosine of best chunk)

- In-scope: min 0.646, median 0.803
- Out-of-scope: max 0.705, values [(0.43, 'oos-01'), (0.433, 'oos-04'), (0.441, 'oos-02'), (0.459, 'oos-03'), (0.537, 'oos-05'), (0.538, 'oos-07'), (0.705, 'oos-06')]
- **Suggested MIN_DENSE = 0.626** (set in rag/pipeline.py). Questions that pass the gate but aren't covered are caught by the LLM's abstention.
