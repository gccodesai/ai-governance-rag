## Relevance-gate calibration (dense cosine of best chunk)

- In-scope: min 0.635, median 0.793
- Out-of-scope: max 0.705, values [(0.406, 'oos-01'), (0.425, 'oos-04'), (0.437, 'oos-02'), (0.483, 'oos-03'), (0.536, 'oos-05'), (0.55, 'oos-07'), (0.705, 'oos-06')]
- **Suggested MIN_DENSE = 0.615** (set in rag/pipeline.py). Questions that pass the gate but aren't covered are caught by the LLM's abstention.
