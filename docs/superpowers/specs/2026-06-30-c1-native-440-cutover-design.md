# C1 — Standardise the bundled Python stack to native OSMOSE 4.4.0 — design

> Status: design (awaiting review) · 2026-06-30
> Make the Python setup genuinely 4.4.x-native (not a 4.3.3 engine behind a permanent shim): convert the
> bundled configs to native 4.4.0 on disk, flip the write/jar/calibration defaults to 4.4.1, and reframe
> the 4.3.3→4.4.0 `canonicalize` path as a *legacy adapter*. The Python engine's **behaviour is unchanged**
> — every conversion is gated by a Python-engine round-trip parity check. Sub-project **C1** of the 4.4.x
> standardisation; **C2** (UI Java-block version-awareness) and **C3** (BoB 365-step forcing) stay separate.
> Builds on sub-project **A** (Baltic background keys, master `072b388`).

## 1. Background (verified against the code)

- The Python engine is **already 4.4.0-native internally**: `OsmoseConfigReader` (`reader.py:215`) and
  `EngineConfig.from_dict` (`config.py:1625`) both run `canonicalize_config` (`aliases.py:265` →
  `migrate_config(target="4.4.0")`), which renames any 4.3.3 keys to 4.4.0 spelling (idempotent on already-
  4.4.0 keys). So the engine reads 4.4.0 keys; the bundled configs being 4.3.3-spelled means the rename
  fires on every load — the "permanent crutch" we are removing.
- **Larval-rate dual semantics**: native-4.4.0 stores the additional-larval-mortality rate as **rate/year**;
  the reader divides it by `ndt` (`reader.py:100`, gated `source_version ≥ 4.4.0`) to recover the per-cohort
  value the engine applies via `1-exp(-rate)`. The write side multiplies by `ndt` (`aliases.py` `_migrate_larva_rate`).
- All 4 in-scope configs (`eec_full`, `minimal`, `baltic`, `baltic_ev`) carry a larval-rate key (Baltic
  across 9 sub-files), so the rate round-trip applies to all.

## 2. Two verified hazards this design MUST handle

**H1 — `to_target_keys` double-scales the rate (confirmed bug).** `to_target_keys` (`aliases.py:17-24`)
applies `_drop_4_4_0_removed_keys`, `_migrate_larva_rate(×ndt)`, and the emits **whenever `target ≥ 4.4.0`,
without checking the source version**. Once a config is native 4.4.0 (rate already `×ndt`), the UI Java path
(`write_temp_config → to_target_keys(4.4.1)`) would multiply by `ndt` **again** → rate `×ndt²`. The reverse
path (4.4.0 source → 4.3.3 jar) also currently fails to `÷ndt` back. **Fix is a core deliverable (§4.1).**

**H2 — dropping `lmax`/`beta` from source breaks the Python engine.** The Python engine reads
`species.lmax.sp{i}` (`config.py:473`; if absent → `lmax = linf`) and `species.beta.sp{i}` (`config.py:1976`;
bioen β, default 0.8). `to_target_keys`'s `_drop_4_4_0_removed_keys` strips these for the **Java** write
(Java 4.4.0 removed `lmax`). The native **source** the Python engine reads must **keep** `lmax`/`beta` —
the conversion is NOT a verbatim `to_target_keys` (§4.2).

## 3. Goal & non-goals

**Goal:** bundled configs (minus BoB) are native `osmose.version 4.4.x` on disk; the write/jar/calibration
defaults are 4.4.1; the legacy 4.3.3-read path is clearly demoted; the Python engine produces parity-
equivalent results before/after; the 4.3.3 jar still works via the reverse-map.

**Non-goals (separate cycles):** BoB conversion (C3 — 365-step blocker); UI Java-block version-awareness so
`nbackground>0` runs on Java 4.4.1 from the UI (C2); removing the `canonicalize` legacy adapter (kept for
old user configs).

## 4. Design

### 4.1 Make `to_target_keys`'s larval-rate transform source-version-aware — CORE, do first
**Only the larval-rate transform** needs the fix (verified): the key RENAMES_440 (forward and inverse) have
disjoint old/new key sets so they are already idempotent, and `_drop_4_4_0_removed_keys` is a no-op when the
key is absent and **must keep firing on the Java write** (Java 4.4.0 needs `lmax`/`beta` gone — see §4.2,
where native *source* keeps them). The emits are `setdefault`-safe. So leave renames/drops/emits as-is and
change ONLY the rate.

Read the input's existing `osmose.version` as the **source version** and scale the rate by the *net* factor
`ndt^[target≥4.4] ÷ ndt^[source≥4.4]`:

| source → target | rate factor | meaning |
|---|---|---|
| 4.3.3 → 4.4.x | `×ndt` | current behaviour (write native) |
| 4.4.x → 4.4.x | `×1` (no-op) | **the fix** — no double-scale |
| 4.4.x → 4.3.3 | `÷ndt` | **the fix** — restore per-cohort for the 4.3.3 jar |
| 4.3.3 → 4.3.3 | `×1` | unchanged |

(`_ndtperyear` missing/0 → already warns + skips, unchanged.) Unit-tested with a full source×target matrix;
this is independently a latent-bug fix and the foundation the native configs stand on.

### 4.2 Config-conversion tool — `scripts/migrate_bundled_to_440.py`
For each in-scope config, produce native 4.4.0 SOURCE that the **Python engine reads identically**:
- Rename keys to 4.4.0 spelling (RENAMES_440 forward), stamp `osmose.version 4.4.1`, scale the larval rate
  `×ndt` (rate/year), emit the resource-forcing keys for resource species (EEC).
- **KEEP `species.lmax`/`species.beta`** (H2) — do NOT apply the Java-only drop.
- **Preserve the multi-file structure**: rewrite each sub-file in place (per-key rename within the file;
  rate scale using the merged `ndt`; append emitted keys to the species/ltl sub-file; stamp version in the
  master). The committed deliverable is the converted `data/<config>/*`.
- **In scope:** `data/eec_full`, `data/minimal`, `data/baltic`, `data/baltic_ev`. **Excluded:** `data/examples`
  (BoB) and any sub-file shared with it (audit for shared includes before writing).
- Baltic: only Python-relevant 4.4.0 keys + `species.multiplier`/`beta` land in source; the Java-only inline
  `species.biomass` + accessibility rows stay in sub-project A's staged-copy harness (decided), NOT source.

### 4.3 Round-trip parity gate — `scripts/native_440_parity.py`
Per config: capture the **pre-conversion** Python-engine outputs (biomass/abundance/yield) from the current
4.3.3 source with **fixed RNG** (`simulation.rng.fixed=true`, fixed seed), short run (recommend **3 years** —
long enough for signal, short enough to bound any ULP-perturbation drift); then run the **post-conversion**
native-4.4.0 source the same way and assert **max relative diff < 1e-9** (tolerance — the `×ndt÷ndt` round-
trip is ~1 ULP, not bit-exact; decided). Bit-exact required for configs/outputs the rate doesn't touch.
Snapshots stored under the script / a baseline dir. This gate is the safety net for every conversion.

### 4.4 Defaults flip to 4.4.1
- `DEFAULT_TARGET_VERSION = "4.4.1"` + `target_version_for_jar(jar_path) -> str` (parse the `\d+\.\d+\.\d+`
  triplet from the jar filename — handles `osmose_4.3.3-…` and `osmose-4.4.1-…`; default 4.4.1) in `aliases.py`.
- Default jar → 4.4.1 at `ui/state.py:42` (both jars stay in the dropdown).
- `ui/pages/run.py:366`: `write_temp_config(…, target_version=target_version_for_jar(jar_path))`.
- `osmose/calibration/problem.py:475`: `to_target_keys(overrides, target_version=target_version_for_jar(self.jar_path))`
  (was hardcoded `"4.3.3"`).
- Flip `to_target_keys` (`aliases.py:229`) / `writer.py:63` / `demo.py:315` defaults to `DEFAULT_TARGET_VERSION`.
- Now native: writing for the 4.4.1 jar is a near-no-op (§4.1 idempotent); the 4.3.3 jar reverse-maps (`÷ndt`).

### 4.5 Engine legacy-framing cleanup (modest, audit-driven)
- Reframe `canonicalize_config` as a documented **legacy 4.3.3 adapter** + emit a **one-time deprecation
  warning** when it actually renames old keys (so loading a legacy config is visibly legacy).
- Update residual 4.3.3 framing where it's now misleading: `osmose/engine/timeseries.py` docstring; finalize
  the `osmose/engine/config_validation.py` 4.4.0-canonical-key allowlist (the PR1/PR2 notes). **Audit-driven
  — only where the semantics are version-stable; no behaviour change** (parity gate catches any slip).

## 5. Validation
- **§4.1 unit matrix**: source×target × {has-rate, has-lmax/beta, resource} — no double-scale, correct
  `÷ndt` reverse, lmax/beta preserved through the native path, emits idempotent.
- **§4.3 round-trip parity** for all 4 configs (<1e-9).
- **`cross_engine_parity_440.py`** (EEC on Java 4.4.1) stays green; add a **legacy-4.3.3 fixture** so the
  `canonicalize` adapter stays tested.
- **Test sweep** (~6–13 tests): tests asserting `osmose.version 4.3.3` or 4.3.3-spelled keys on bundled
  configs flip to 4.4.x; ~23 tests merely *load* bundled configs (most won't break — verify). Enumerate at
  plan time.
- **A-harness regression**: confirm `scripts/baltic_440_smoke.py` (sub-project A) still works — it reads
  `data/baltic` source then `write_temp_config(target=4.4.1)`; with native-4.4.0 Baltic source + §4.1
  idempotency, it must NOT double-scale and must still run + feed (re-run it).

## 6. Risks
- **§4.1 is load-bearing and subtle** — get the source×target rate matrix wrong and every Java run mis-scales.
  Mitigation: the unit matrix + the A-harness re-run + the parity gate.
- **Parity-gate chaos**: a ~1-ULP rate perturbation could amplify over years; 3-yr fixed-RNG run keeps it
  bounded. If a config exceeds 1e-9, investigate (likely a real semantic drift, not float noise) before
  loosening the gate.
- **Shared sub-files**: *verified clear* — EEC and BoB/`examples` share no CSV includes (distinct
  `eec_param-*` vs `osm_param-*` prefixes); Baltic is its own tree. Re-confirm at plan time before writing
  any file; never write a file BoB reads.
- **Resource-forcing emit into EEC source (H3) — benign**: `config.py` does not read `species.biomass.*`,
  so the emitted resource keys are inert extra keys for the Python engine (unlike `lmax`/`beta`); the parity
  gate confirms. (Contrast H2, where the engine *does* read the dropped keys.)

## 7. Plan-time task ordering (recommendation)
1. §4.1 idempotent `to_target_keys` (+ unit matrix) — foundation.
2. §4.3 parity harness + capture pre-conversion baselines.
3. §4.2 convert **EEC first** (canary), gate; then `minimal`, `baltic`, `baltic_ev`.
4. §4.4 defaults flip.
5. §4.5 legacy-framing cleanup.
6. §5 test sweep + A-harness re-run.

## 8. Out of scope
BoB (C3), UI Java-block version-awareness (C2), removing the canonicalize adapter, native rewrite of BoB-
shared files, any Python-engine *behaviour* change (parity-gated to zero).
