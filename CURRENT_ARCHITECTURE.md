# Current architecture audit and separation plan

The local system uses `PatchVisionEncoder`, `OCRLayoutEncoder`, `VLMProjection`, and `TinyCausalDecoder`. `OCRLayoutEncoder` implements the paper-style structural sequence of OCR words, normalized 2D boxes, instruction, and learned queries. `DocVLM` fuses visual tokens with projected query states. The decoder and OCR encoder are intentionally small structural approximations, not pretrained DocFormerV2/VLMs.

| Area | Current file | Baseline ownership | Improved ownership |
|---|---|---|---|
| OCR/layout encoder | `src/models/document_encoder.py` | Shared unchanged | Shared unchanged |
| Fixed queries | `DocVLM` fixed mode | `baseline/model.py` | Not used |
| Adaptive scoring | `adaptive_query_compression.py` | Not imported by wrapper | `improved/model.py` |
| Vision/projection/decoder | `src/models/*` | Shared neutral modules | Shared neutral modules |
| OCR input | `src/ocr/tesseract_engine.py` | Same input adapter | Same input adapter |
| Training/data | `src/training`, `src/data` | Same preprocessing | Same preprocessing |

Missing for paper reproduction: DocFormerV2 weights, compatible frozen VLM, original datasets, two-stage large-scale training, and benchmark evaluation.
