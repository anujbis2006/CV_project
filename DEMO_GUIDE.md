# Teacher demonstration guide

Generate local test data once:

```bash
python3 scripts/generate_synthetic_data.py --samples 24
```

Run the original/paper-style structural baseline:

```bash
python run_baseline.py --image data/synthetic/0000.png --question "What is the total amount?"
```

Point out `[BASELINE / PAPER IMPLEMENTATION]`, `Query mode: Fixed`, and `Number of queries: 64`. Explain that it uses the OCR/layout encoder plus a fixed learned-query representation, matching the paper's high-level mechanism.

Run the proposed extension:

```bash
python run_improved.py --image data/synthetic/0000.png --question "What is the total amount?"
```

Point out `[IMPROVED / OUR PROPOSED EXTENSION]`, selected query indices, and the smaller selected-query count. The only intended architectural change is question-aware query selection before the shared projection/decoder.

Compare the same input:

```bash
python compare_models.py --image data/synthetic/0000.png --question "What is the total amount?"
```

The command measures actual local token counts, latency and Python allocation peak. Do not present synthetic/random-model answer text, these single-run timings, or this structural implementation as paper benchmark accuracy. See `FINAL_IMPLEMENTATION_STATUS.md` for exact boundaries.
