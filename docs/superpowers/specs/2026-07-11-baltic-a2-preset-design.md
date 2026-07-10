# Baltic A2-calibrated preset — design spec

**Date:** 2026-07-11
**Status:** approved (design), pending implementation plan
**Related:** `docs/baltic_a2_calibration_results_2026-07-09.md`, `docs/diagnostics/baltic_a2_calibrated_params.json`, `[[project-baltic-chunka2-depletable-plankton]]`

## Goal

Package the converged A2 (depletable-plankton) Baltic calibration as a new bundled demo preset,
**`baltic_a2`**, selectable in the UI model picker and runnable on the Python engine. This makes the
best-achievable Baltic community calibration a first-class, reproducible artifact instead of a set of
numbers in a results JSON.

## Non-goals (YAGNI)

- **Do not** modify the deployed `data/baltic/` config in any way.
- **Do not** re-run the DE, tune further, or attempt to fix the percid overshoot (established structural /
  grid-resolution limit — 9 levers ruled out; see memory).
- **Do not** add a CI gate on emergent simulation outcomes (biomass bands) — real-engine Baltic outcomes
  are non-reproducible across CI runner cores (`[[feedback-ci-fragile-emergent-tests]]`).
- No new engine features — A2 depletion already shipped (`osmose/engine/resources.py`).

## Honest framing (load-bearing)

This preset is the **best community-wide fit achievable with depletable plankton**, *not* a fully
ICES-calibrated config. It **must not** be labeled "ICES-calibrated". The documented multi-seed result
(`a2_on_converged`, objective 2.68 vs A2-off baseline 3.57):

| species | biomass (multi-seed) | ICES band | status |
|---|---|---|---|
| cod | 298 k | 60 k – 250 k | over 2.49× (just above) |
| herring | 1.46 M | 0.8 M – 3 M | **in** |
| sprat | 1.60 M | 0.8 M – 2.5 M | **in** |
| stickleback | 200 k | 50 k – 500 k | **in** |
| flounder | 2.68 M | 20 k – 100 k | over ~53× |
| perch | 2.12 M | 8 k – 50 k | over ~106× |
| pikeperch | 875 k | 4 k – 25 k | over ~88× |
| smelt | 325 k | 20 k – 120 k | over ~5.4× |

**3/8 in-band.** Framing everywhere it is surfaced (DEMO_INFO summary, config header comments, results
doc): A2 compresses the community overshoot from **17–400×** (A2-off) down to **near-band** (pelagics +
stickleback in-band, cod just above, coastal percids structurally over). This is the payoff of the A2
investigation, presented honestly.

## Architecture — DRY overlay (not a full copy)

The existing self-contained demos (`benguela`, `eec_full`) each own a full `data/<name>/` directory.
Copying all of `data/baltic/` would duplicate large NetCDFs (grid, LTL forcing, salinity/predator
climatologies, movement maps) — wasteful and a divergence hazard. Instead, `baltic_a2` is a thin overlay
on `data/baltic/`.

### Include resolution (verified)

`osmose/config/reader.py:124` resolves each `osmose.configuration.*` include as
`master_file.parent / value`, and rejects any include whose resolved path is not
`is_relative_to(config_dir)` (no `../` escapes). Therefore the a2 master **must** reference sub-configs by
bare basename, and every referenced basename **must** exist flat in the generated `config/` directory.

### `data/baltic_a2/` — three small text files only

1. **`baltic_a2_all-parameters.csv`** — a master identical to `data/baltic/baltic_all-parameters.csv`
   except:
   - `osmose.configuration.mortality.additional` → `baltic_a2_param-additional-mortality.csv`
   - one added line: `osmose.configuration.a2.depletion ; baltic_a2_param-depletion.csv`
   - all other keys/includes identical to baltic's, referenced by bare basename. **Do not** change
     `output.file.prefix` — the Java guard keys off `ltl.depletable.enabled` (not the prefix), and each
     run already writes to its own `output_dir`, so a distinct prefix buys nothing and would introduce an
     override-precedence conflict with baltic's `baltic_param-output.csv` include.

2. **`baltic_a2_param-additional-mortality.csv`** — the 16 converged mortality values (replaces baltic's
   R18 values), with provenance comments. Exact values (from `a2_on_converged.params`):

   ```
   mortality.additional.larva.rate.sp0;1.8495054614929225
   mortality.additional.larva.rate.sp1;0.6091614461276307
   mortality.additional.larva.rate.sp2;1.7574285062912955
   mortality.additional.larva.rate.sp3;0.3277205467582994
   mortality.additional.larva.rate.sp4;5.024141712395672
   mortality.additional.larva.rate.sp5;1.1869723413415985
   mortality.additional.larva.rate.sp6;0.3791432328547528
   mortality.additional.larva.rate.sp7;0.27314862986759136
   mortality.additional.rate.sp0;4.288045380663061
   mortality.additional.rate.sp1;0.2636287453341465
   mortality.additional.rate.sp2;0.003071941136699811
   mortality.additional.rate.sp3;0.0045211280482306045
   mortality.additional.rate.sp4;0.005680413608708062
   mortality.additional.rate.sp5;0.855951786667689
   mortality.additional.rate.sp6;0.0036156979635421347
   mortality.additional.rate.sp7;0.19494616193531136
   ```
   (species: sp0 cod, sp1 herring, sp2 sprat, sp3 flounder, sp4 perch, sp5 pikeperch, sp6 smelt,
   sp7 stickleback.)

3. **`baltic_a2_param-depletion.csv`** — the 10 depletion/regrowth keys. Values from
   `enable_a2_base_config` (phyto sp8/sp9 fixed fast = 5.0) and the converged zoo regrowth rate
   (`species.regrowth.rate.zoo` = 1.0580953986747008, applied to sp10–13 = the grouped
   `_ZOO_RESOURCE_INDICES`):

   ```
   ltl.depletable.enabled;true
   ltl.depletable.floor;0.05
   species.regrowth.rate.sp8;5.0
   species.regrowth.rate.sp9;5.0
   species.regrowth.rate.sp10;1.0580953986747008
   species.regrowth.rate.sp11;1.0580953986747008
   species.regrowth.rate.sp12;1.0580953986747008
   species.regrowth.rate.sp13;1.0580953986747008
   ```

### Generator: `_generate_baltic_a2(output_dir)` in `osmose/demo.py`

Mirrors `_generate_baltic`, then overlays the a2 deltas:

1. `data_dir = _bundled_data_dir("baltic")`; `a2_dir = _bundled_data_dir("baltic_a2")`;
   `config_dir = output_dir / "config"`; `sim_output = output_dir / "output"`;
   `sim_output.mkdir(parents=True, exist_ok=True)` (mirrors `_generate_baltic`).
2. `shutil.copytree(data_dir, config_dir, dirs_exist_ok=True)` — brings in grid, maps, forcing, all
   baltic sub-CSVs.
3. `shutil.copytree(a2_dir, config_dir, dirs_exist_ok=True)` — overlays the 3 a2 files.
4. Return `{"config_file": config_dir / "baltic_a2_all-parameters.csv", "output_dir": sim_output}`.

If either `_bundled_data_dir` returns `None` (missing bundle), fall back to the same warn-and-stub
behavior `_generate_baltic` uses (write a minimal non-runnable master). `nyear` stays 15 — inherited from
baltic's `baltic_param-simulation.csv` via step 2, matching the `--years 15` calibration; no override
needed.

**Coupling risk & mitigation:** the a2 master hardcodes baltic's sub-CSV basenames. If a baltic sub-config
is renamed, the a2 master breaks. Mitigated by a unit test asserting the a2 master's non-a2 includes are a
subset of baltic's includes (see Testing).

## Java-engine guard

A2 depletion (`ltl.depletable.enabled`) has no Java-jar equivalent — the jar silently ignores the key and
the community reverts to the non-depletable overshoot (percids 400×). Running `baltic_a2` on Java would
therefore produce wrong (uncalibrated) results silently. Add a guard at the **top** of
`java_engine_block_reason(config, jar_version)` in `osmose/runner.py`, before the nbackground logic:

```python
if str(config.get("ltl.depletable.enabled", "")).strip().lower() == "true":
    return (
        "This configuration uses depletable plankton (ltl.depletable.enabled), a Python-engine "
        "feature with no Java-jar equivalent. Run it on the Python engine."
    )
```

This makes `baltic_a2` Python-gated in the UI run page exactly like `benguela`.

## Registration

- `list_demos()` — append `"baltic_a2"`.
- `DEMO_INFO["baltic_a2"]` — title **"Baltic Sea (A2-calibrated)"**, region "Central/Eastern Baltic",
  species "8 focal species", resources "6 LTL (depletable) + 2 background groups", engine **"Python"**,
  summary framing it as best-achievable-not-fully-calibrated (see Honest framing).
- `osmose_demo`'s `generators` dict — map `"baltic_a2"` → `_generate_baltic_a2`.

The UI (`ui/pages/grid.py`, `ui/pages/scenarios.py`) builds picker choices generically from
`list_demos()`/`demo_info()`, so no UI-page edits are required.

## Testing

Split along the CI-flakiness boundary.

### CI-safe unit tests — `tests/test_baltic_a2_demo.py`

1. `baltic_a2` is in `list_demos()` and has a `DEMO_INFO` entry with engine == "Python".
2. `osmose_demo("baltic_a2", tmp)` returns a config whose flattened (reader-loaded) keys include:
   `ltl.depletable.enabled == "true"`, `ltl.depletable.floor == "0.05"`, and the 8 `species.regrowth.rate`
   keys with the expected values.
3. The 16 converged mortality values are present in the loaded config (spot-check a representative subset
   incl. the extremes: `larva.rate.sp4 == 5.024...`, `rate.sp0 == 4.288...`, `rate.sp2 == 0.00307...`).
4. The generated config loads cleanly through `osmose/config/reader.py` (no missing-include / escape
   errors) — proves the overlay's basename includes resolve.
5. The a2 master's `osmose.configuration.*` keys equal baltic's keys **plus** `a2.depletion`, and every
   include *target* basename referenced by the a2 master exists as a file in the generated `config/`
   (guards the rename-coupling risk; note the `mortality.additional` target intentionally differs —
   it points at the a2 file).
6. `java_engine_block_reason` returns a non-None Python-only reason for the loaded `baltic_a2` config, and
   still returns `None` for a config without depletion (regression guard on the new check).

### Local validation (NOT a CI gate)

Run `osmose_demo("baltic_a2", …)` once on the Python engine (nyear 15, a couple of seeds) to confirm it
executes cleanly and the community lands near the documented bands (pelagics + stickleback in-band, cod
~2.5×, percids over). Record the actual numbers in `docs/baltic_a2_calibration_results_2026-07-09.md`
under a "Deployed as `baltic_a2` preset" section. No emergent-outcome assertion enters CI.

## Success criteria

1. `baltic_a2` appears in the UI model picker with the honest "A2-calibrated" title/summary.
2. It runs cleanly on the Python engine and lands near the documented bands (pelagics + stickleback
   in-band, cod ~2.5×, percids over), validated locally over a couple of seeds.
3. It is blocked on the Java engine with a clear Python-only message.
4. `data/baltic/` is byte-unchanged; no NetCDFs duplicated (`data/baltic_a2/` is 3 small text files).
5. CI-safe unit tests pass; no flaky emergent gate added.
6. Every surface that names the preset frames it as best-achievable, not fully calibrated.
