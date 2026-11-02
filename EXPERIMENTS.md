# Experiment tracking

Local measurements are saved under `outputs/benchmarks/query_efficiency/` as JSON and CSV. Each row records candidate/selected query count, visual and total prefix tokens, measured forward latency, and Python allocation peak. These are local CPU measurements, not paper comparisons. No real benchmark experiment has been run because DocVQA data and paper-scale backbones are absent.

Run: `python3 scripts/benchmark_queries.py --config configs/debug.yaml --queries 8,16,32`.
