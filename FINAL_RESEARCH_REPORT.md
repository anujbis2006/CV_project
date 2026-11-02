# DocVLM local research extension report

## Abstract

This project preserves a fixed-query DocVLM-style structural implementation and adds a runnable proposed extension: adaptive question-aware query selection. The extension selects a Top-K subset of candidate OCR query states before projection. It is an engineering hypothesis, not a validated scientific contribution.

## Original method and reproduction status

The original paper uses DocFormerV2 OCR/layout encoding, 64 learned queries, a frozen VLM, and two-stage training. This project implements the same high-level fusion ordering but uses small local Transformer modules. It does not reproduce paper checkpoints, training data, VLM adapters, or reported metrics.

## Proposed improvement

`QuestionAwareQuerySelector` scores each candidate query state against pooled question features, selects Top-K states, applies score gating, and forwards them through the existing projection and decoder. Fixed and adaptive modes remain independently selectable by `compression_type`.

## Experimental setup and measured local result

Synthetic records were generated locally. The adaptive debug configuration uses 8 candidate queries and selects 4. The query benchmark measured two runs: 8 candidates selected 4, with 16 visual and 20 total prefix tokens at 1.796 ms; 16 candidates selected 4, also with 20 total prefix tokens at 1.717 ms. These are measured local CPU forward times, not accuracy comparisons.

## Limitations

No DocVQA data is available locally, and no benchmark accuracy was measured. OCR quality on tiny synthetic renderings was weak. Greedy inference executes but produces unusable text because its model was trained only for a few synthetic debug epochs. No conclusion on the adaptive method's accuracy can be drawn.

## Conclusion

The project now supports fixed-query versus question-aware adaptive-query execution and local token/latency measurement. A real research comparison requires licensed DocVQA data, real OCR, a compatible frozen VLM interface, pretrained DocFormerV2, controlled seeds, and suitable compute.
