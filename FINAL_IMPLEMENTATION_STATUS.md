# Final implementation status

## A. Paper/baseline implementation

`baseline/` is an independently runnable fixed-query implementation. `run_baseline.py` forces fixed compression and reports 64 query states by default.

## B. Our proposed improvement

`improved/` is independently runnable. It introduces question-conditioned query scoring and Top-K selection before the existing projection. It is called a proposed extension; no novelty claim is made.

## C. Experiments and D. results

The three top-level commands were run on `data/synthetic/0000.png` with the same question. The baseline reported 64 OCR query tokens and 80 total prefix tokens; the improved runner selected 16 OCR query tokens and 32 total prefix tokens. Latency and memory shown by runners are actual process-local measurements and vary by run. No accuracy was measured.

## E. Limitations

The runners instantiate local untrained structural models unless matching checkpoints are explicitly integrated. Their decoded answers are therefore not meaningful. Tesseract OCR is real, but tiny synthetic text was imperfectly read.

## F/G. Actual reproduction versus approximation

Reproduced structurally: OCR text/layout + instruction + learned queries, retaining query states, projection, and concatenation with visual tokens. Approximate: OCR encoder, vision encoder, decoder, data, pretrained weights, training scale, and evaluation.

## H. Remaining work

Integrate DocFormerV2 and a frozen compatible VLM, acquire licensed DocVQA data, train comparable checkpoints with controlled seeds, then run accuracy ablations and plots. No paper-level reproduction or research-performance claim is currently justified.
