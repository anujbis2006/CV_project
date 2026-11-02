# Current implementation audit

| Component | Status | Paper-faithful? | Evidence | Next step |
|---|---|---|---|---|
| OCR/layout/query topology | Runnable local implementation | Structural only | `OCRLayoutEncoder` | Load DocFormerV2 weights |
| Learned fixed queries | Implemented | Yes, topology | `compression_type: fixed` | Train with paper data |
| Adaptive query extension | Implemented | Proposed extension | `adaptive_query_compression.py` | Evaluate on real data |
| Vision/decoder | Tiny local modules | No | `vision_encoder.py`, `text_encoder.py` | Integrate frozen VLM |
| OCR | Tesseract adapter | Not paper-specified | `scripts/run_ocr.py` | Compare engines/data OCR |
| Dataset | Synthetic runnable, DocVQA manifest-compatible | Synthetic only | `data/processed` | Acquire licensed DocVQA |
| Training | Debug loop | Partial | `trainer.py` | Two-stage/frozen VLM run |
| Evaluation/benchmarks | Local token/latency only | No benchmark claims | benchmark scripts | Run prepared DocVQA |
