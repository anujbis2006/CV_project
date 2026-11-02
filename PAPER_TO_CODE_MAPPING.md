# Paper-to-code mapping

| Paper component | Code |
|---|---|
| OCR word + 2D layout embeddings | `src/models/document_encoder.py:OCRLayoutEncoder` |
| `Encoder([E_OCR,E_instruction,Q])` | `OCRLayoutEncoder.forward` |
| Retain M learned query states | `OCRLayoutEncoder.forward` |
| Projection to VLM dimension | `src/models/heads.py:VLMProjection` |
| Concatenate OCR and visual tokens | `src/models/model.py:DocVLM.forward` |
| Autoregressive answer objective | `src/losses/losses.py` |
| Two-stage paper settings | `configs/paper.yaml` |
