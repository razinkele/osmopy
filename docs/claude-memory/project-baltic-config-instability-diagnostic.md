---
name: project-baltic-config-instability-diagnostic
description: "ROOT CAUSE — Baltic forage collapse on Java is config instability, NOT a Java/C2 bug (Python collapses it worse)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 18a62785-d85c-4be4-8f3a-e164e19add6a
---

**Investigated 2026-06-30: "Baltic forage species (sprat/smelt/stickleback) go extinct on the Java 4.4.1 run."** ROOT CAUSE = **the bundled `data/baltic` config is dynamically unstable** — it collapses to a 2-species (herring+sprat) degenerate state by ~year 30 on the **Python engine too**, across seeds 0/42/7. NOT a Java bug, NOT the C2 staging, NOT the size-ratio reorder.

**Evidence (PythonEngine `run_in_memory`, 50yr, `osmose_demo('baltic')` config):**
- Focal species alive (>1e-3 of peak): yr5 ≈7/8 → yr10 ≈6-7/8 → yr20 4/8 → **yr30 2/8 → yr49 2/8 (only herring + sprat, which EXPLODE: sprat 3e4→6.9e6)**. Seed-independent.
- Java (user's run): collapsed sprat/smelt/stickleback (3/8) — FEWER than Python's 6/8. The engines lose DIFFERENT species (stochastic RNG/process order) but BOTH collapse.
- → Config holds ~10-15yr then progressively collapses. The `data/baltic` default `simulation.time.nyear=50` outruns its stable horizon.

**Ruled out (don't rebuild these):**
- **Size-ratio "must be reordered" warning is BENIGN.** Baltic stores `sizeratio.min<max`; EEC stores `min>max` (opposite). Python swaps to `[smaller,larger)` either way (`config.py` ~663-670 focal + ~697-700 bg). EEC passes Java-4.4.1 parity → Java's reorder lands on the same window. Predation window identical across engines.
- **C2 staging is NOT the cause.** `data/baltic/predation-accessibility.csv` has NO GreySeal/Cormorant columns → Python defaults their access to **1.0 to ALL prey** (predation.py "-1 not found → 1.0"); my `BG_ACCESS` staging gives them LESS (0.4 herring/sprat, 0.0 smelt/stickleback). So Java's staged bg-predators are GENTLER, yet Python collapses MORE.

**Consistent with known Baltic marginal calibration** (1/8 ICES-in-range; percid overshoots structural / grid under-resolution [[project_percid_overshoot_diagnostic]]). **Fix = recalibrate for long-term stability (a project, structurally hard), OR run Baltic only over its ~10-15yr stable horizon.** No code/Java/staging change warranted. Baltic stays PYTHON-engine-reference [[project-java-live-streaming]] [[project-c2-ui-java-440-background]].
