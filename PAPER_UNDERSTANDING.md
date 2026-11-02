# Paper understanding

DocVLM (Nacson et al., CVPR 2025) makes existing VLMs better document readers without altering their weights. It feeds OCR words and 2D positions plus the user instruction through the text/layout portion of pretrained DocFormerV2 (its visual branch is omitted). M learned query embeddings, normally 64, are included in the same encoder sequence: `Encoder([E_OCR, E_instruction, Q])`. The M resulting query states are projected to the VLM hidden size and concatenated with visual tokens before the frozen LLM decoder.

Training is autoregressive answer cross-entropy with a frozen VLM. Stage I excludes image features, trains queries/projection then unfreezes OCR encoder; stage II adds vision. The supplement reports AdamW, cosine schedule, 1,000 warmup steps, 140K stage-I steps (encoder frozen first 10K) and 100K stage-II steps. Stage-I rates are 1e-4 (queries/projection) and 5e-5 (OCR encoder); stage-II is 5e-6.

Datasets include DocVQA, InfoVQA, ChartQA, TAT-DQA, TextVQA, ST-VQA, OCR-VQA and TextCaps; COCO Caption/VQA-v2 are added in stage II. Metrics include ANLS, VQA score and CIDEr. The full recipe is infeasible locally; this repository's synthetic/debug results are not paper results.
