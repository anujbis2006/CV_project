# Implementation plan and local boundary

Implemented: data manifest/colllation, OCR/layout/query transformer, visual tokens, projection/fusion, causal answer loss, config loading, checkpoints, synthetic generator, smoke training. The production boundary is intentionally documented: downloading DocFormerV2 and a compatible VLM needs model weights, adapters and hardware beyond this machine. Replace `PatchVisionEncoder` and `TinyCausalDecoder` with the chosen frozen VLM interfaces while preserving the projected query concatenation.
