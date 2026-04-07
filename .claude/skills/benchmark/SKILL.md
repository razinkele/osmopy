---
name: benchmark
description: Run Python vs Java engine benchmark comparison and report timing/parity results
disable-model-invocation: true
---

Run engine benchmarks comparing Python and Java OSMOSE engine performance.

## Arguments

- `config` (optional): "bob" (Bay of Biscay), "eec" (EEC), or "all" (default: "all")
- `years` (optional): number of simulation years (default: 5)

## Steps

1. Run from `/home/razinka/osmose/osmose-python/`

2. **Run benchmark script**:
   ```
   .venv/bin/python scripts/benchmark_engine.py --years {years}
   ```

3. **Report results** as a table:

   | Config | Engine | Time (s) | Speedup |
   |--------|--------|----------|---------|
   | BoB 5yr | Python | X.XXs | — |
   | BoB 5yr | Java | X.XXs | X.Xx |
   | EEC 5yr | Python | X.XXs | — |
   | EEC 5yr | Java | X.XXs | X.Xx |

4. **Flag regressions**: If Python is slower than the baseline (BoB ~2.0s, EEC ~5.2s), warn about the regression and identify which files changed recently that may have caused it:
   ```
   git -C /home/razinka/osmose/osmose-python log --oneline -10 -- osmose/engine/
   ```

5. **Compare parity**: After benchmarking, run the validation to confirm outputs still match:
   ```
   .venv/bin/python scripts/validate_engines.py --years 1
   ```

## Rules

- Always use `.venv/bin/python`, never system python
- Run from `/home/razinka/osmose/osmose-python/`
- Report both timing AND parity — performance gains that break parity are regressions
- Baselines: BoB 5yr ~2.0s, EEC 5yr ~5.2s (Python), Java BoB ~2.3s, EEC ~7.2s
