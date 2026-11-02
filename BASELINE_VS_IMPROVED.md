# Baseline versus improved

| Component | Baseline | Improved |
|---|---|---|
| OCR | Same Tesseract adapter | Same Tesseract adapter |
| Layout | Same normalized four-coordinate boxes | Same |
| Vision | Same local patch encoder | Same |
| Query mechanism | 64 fixed learned queries | 64 candidates |
| Query selection | None | Question-aware Top-K |
| Query count | Fixed 64 | Configured selected 16 |
| Projection/decoder | Same local modules | Same local modules |

The baseline is `baseline.model.BaselineDocVLM`, which force-sets `compression_type: fixed`. The extension is `improved.model.ImprovedDocVLM`, which force-sets adaptive mode and uses `src/models/adaptive_query_compression.py:QuestionAwareQuerySelector`. Both runners use `src/demo_runner.py` to enforce the same OCR and preprocessing input format.
