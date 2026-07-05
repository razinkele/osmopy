# Baltic percid thermal recruitment gate — design

**Date:** 2026-07-05
**Status:** Design (approved in brainstorming) — awaiting user review before writing-plans.
**Author:** brainstorming session, osmopy master `b212a09`.
**Topic:** A configurable, temperature-driven per-year recruitment factor for the two Baltic
percids (perch `sp4`, pikeperch `sp5`), plus the CMEMS data pipeline to feed it, shipped
inert-by-default. Tests whether percid overshoot is recruitment-driven.

---

## 1. Motivation & context

The Baltic OSMOSE config exhibits population-level percid overshoot (×38–96 vs ICES/HELCOM
reference envelope). The 2026-07-03 recruitment literature review
(`docs/baltic_recruitment_literature_review_2026-07-03.md`) identified two self-limiting
mechanisms that percids have in nature but the model lacks: **temperature-gated year-class
strength** and **density-dependent cannibalism**. This spec builds the first.

**Scientific basis.** Pekcan-Hekim et al. (2011, *Ambio* 40(5):447,
doi:10.1007/s13280-011-0143-7): pikeperch year-class strength depends strongly on summer
temperature — mean June–July temperature explained ~40% of year-class-catch variance in the
Gulf of Finland and ~73% (July–August) in the Archipelago Sea; strong year-classes form above
~18.5 °C. Mechanism is first-winter survival (fast summer growth → larger size → higher
overwinter survival). Perch shows the same temperature-gated recruitment (Olin et al. 2019,
*Hydrobiologia*, doi:10.1007/s10750-019-04008-z), jointly controlled by temperature and
pikeperch abundance. The ecological point the model is missing: **strong percid year-classes are
the exception, not the rule** — cool years mostly fail. The current config produces a strong
class every year.

**The RV-gate lesson (design constraint).** The reproductive-volume recruitment gate merged
2026-07-02 (`_load_rv_gate`) *worsened* cod overshoot because a mean-preserving multiplier
injected recruitment variance the model amplifies. Therefore this gate's overshoot-damping mode
must be **mean-reducing** (recruitment thermally limited, most years < 1), not mean-preserving.
Mean-preserving is retained only as a realism option and is explicitly flagged as not an
overshoot fix.

**Honesty up front.** A thermal recruitment gate can only damp overshoot if percid overshoot is
recruitment-driven. If it is instead dominated by too-weak adult mortality (missing cannibalism,
predation, fishing), even a mean-reducing gate will not help. The A/B diagnostic frames this as a
hypothesis test with an honest-negative outcome permitted, exactly as with the recruitment
ceiling (`66b374f`).

---

## 2. Goals & non-goals

**Goals**
- A per-year, per-percid-species recruitment multiplier driven by a summer-temperature index,
  applied in `reproduction()`, **inert-by-default and bit-identical when off**.
- Two configurable modes (the user's "configurable baseline"): `thermal_cap` (mean-reducing,
  the overshoot-damping mode) and `mean_preserving` (realism only).
- A real CMEMS `thetao`-based data pipeline: download → per-year × per-species summer-SST series
  → sidecar CSV, mirroring the RV-gate sidecar.
- An A/B diagnostic reporting percid mean biomass and overshoot ratio, gate off vs on.

**Non-goals**
- No changes to cod or to the RV gate / recruitment ceiling (both cod-only; zero interaction).
- No cannibalism / density-dependent adult mortality — that is the separate second percid lever.
- No spatially-resolved per-cell recruitment — `reproduction()` emits one egg count per species,
  so a per-cell temperature field has nowhere to attach (YAGNI).
- No re-tuning of the wider Baltic calibration.

---

## 3. Architecture overview

Four units, each independently testable, mirroring the RV-gate feature's separation of concerns:

```
CMEMS thetao ──▶ [A] builder ──▶ sidecar CSV ──▶ [B] loader ──▶ EngineConfig fields
 (download)      (per-yr/sp        (year,             (_load_       (thermal_gate_*)
                  summer SST)       temp_sp{idx})      thermal_gate)      │
                                                                          ▼
                                                          [C] pure helper thermal_gate_factor
                                                                          │
                                                                          ▼
                                                    reproduction(): n_eggs[sp] *= factor[sp]
                                                                          │
                                                                  [D] A/B diagnostic
```

- **[A] Data builder** (`osmose/forcing/` + a script): CMEMS `thetao` → per-year × per-species
  summer surface-SST scalar over each species' habitat footprint → sidecar CSV.
- **[B] Config loader** `_load_thermal_gate` (`osmose/engine/config.py`): reads the sidecar +
  config keys, applies the response function and mode normalization, returns precomputed
  `EngineConfig` fields. Fail-fast on any bad input. `(None, …)` when the master switch is off.
- **[C] Pure helper** `thermal_gate_factor(config, step)`
  (`osmose/engine/processes/thermal_gate.py`): engine-state-free; returns a per-species egg
  multiplier for the step, 1.0 for disabled species.
- **[D] A/B diagnostic** (`scripts/baltic_percid_thermal_gate_diagnostic.py`): deterministic
  off-vs-on run, reports percid overshoot.

---

## 4. Component detail

### 4.1 [A] Data builder — CMEMS `thetao` → summer-SST series

- **Source:** CMEMS `baltic_phy` monthly reanalysis (the same product already cached for `so`
  under `data/cmems_cache/cmems_downloads/`), variable `thetao`, **surface layer** (percid
  recruitment is a shallow-coastal phenomenon), reusing the salinity download tooling
  (`osmose/forcing/` + the salinity-forcing scripts).
- **Per-species temporal window** (honoring the source): perch summer ≈ June–July, pikeperch
  summer ≈ July–August. Windows are parameters with these defaults.
- **Spatial footprint:** for each species, average over the cells in that species' movement-map
  footprint (cells with nonzero occupancy) — the coastal habitat where its recruitment happens.
  Whole-domain averaging is explicitly rejected (dilutes the coastal signal with cold open-sea
  water, weakening the very correlation the gate relies on).
- **Output:** one scalar per (year, species) → sidecar CSV
  `data/baltic/forcing/baltic_percid_thermal_series.csv`, columns
  `year,temp_sp4,temp_sp5` (per-species index columns; generalizes if more species are enabled).
  Years must be contiguous ascending (same invariant as the RV series).
- **Loud failure:** if `thetao` is not available (e.g. not downloaded — see Risk R1), the builder
  raises a clear error naming the missing product/variable. No silent fallback to a synthetic
  field.

### 4.2 [B] Config loader `_load_thermal_gate`

Mirrors `_load_rv_gate` (`config.py:1082`). Signature:
`_load_thermal_gate(cfg, n_species, n_dt_per_year, n_year) -> (factor_by_index, enabled_mask, offset)`.

- Off (`reproduction.thermal.gate.enabled != true`) → `(None, None, 0)`.
- Read sidecar; validate rows present, `year` column contiguous/ascending, per-species temp
  columns finite and within a plausible range (e.g. −2…30 °C — reject NaN / absurd values).
- Per-species enable mask from `reproduction.thermal.gate.species.enabled.sp{idx}` (≥1 required).
- **Response function** (applied here, so `EngineConfig` stores the final factor — same pattern
  as RV gate applying the mode formula in the loader): logistic
  `r(T) = 1 / (1 + exp(-(T − t50)/slope))`, with per-species `t50` (default 18.5),
  `slope` (default 1.5). Logistic (not linear) encodes "strong classes are the exception."
- **Mode normalization** → `factor_by_index` shape `(n_years, n_species)` (mirrors the ceiling's
  `(n_cols, n_species)` field):
  - `thermal_cap` (**default; mean-reducing**): `factor = clip(r(T) / r_ref, floor, 1.0)` where
    `r_ref = r(tref)` and per-species `tref` (default 20.0 — a warm reference so most years < 1).
    This is the overshoot-damping mode.
  - `mean_preserving` (**realism only**): `factor = r(T) / mean(r(T) over the run window)` using
    the same multiset-mean-over-sampled-model-years construction as the RV gate. Flagged in the
    spec and a code comment as NOT an overshoot fix.
  - `floor` in `[0,1]` (default 0.0), applied as `max(factor, floor)` like the RV gate.
- `start.year` → `offset = start_year − first_year`, identical to the RV gate, so sim-year → real
  temperature-year mapping is consistent with the existing forcing.

### 4.3 [C] Pure helper `thermal_gate_factor(config, step)`

New module `osmose/engine/processes/thermal_gate.py`, copy of `recruitment_gate.py`'s shape:

```python
def thermal_gate_factor(config, step):
    out = np.ones(config.n_species, dtype=np.float64)
    factor = config.thermal_gate_factor_by_index          # (n_years, n_species) or None
    if factor is None:
        return out
    year = step // config.n_dt_per_year
    idx = (config.thermal_gate_offset + year) % factor.shape[0]
    mask = config.thermal_gate_enabled
    out[mask] = factor[idx, mask]
    return out
```

### 4.4 Engine wiring in `reproduction()`

Insert an independent block after the recruitment-ceiling block (`reproduction.py:170`), guarded
`if config.thermal_gate_factor_by_index is not None:`. Per species, apply
`n_eggs[sp] *= gate[sp]` only when `thermal_gate_enabled[sp] and not seeded_this_step[sp]` —
identical guard structure to the RV gate (`reproduction.py:158-165`). Because the gate is
percid-only and RV-gate/ceiling are cod-only, ordering among the three is immaterial.

### 4.5 [D] A/B diagnostic

`scripts/baltic_percid_thermal_gate_diagnostic.py`, mirroring
`baltic_recruitment_ceiling_diagnostic.py`: two deterministic runs (fixed
`movement.randomseed.fixed` + `stochastic.mortality.randomseed.fixed`, seed 0), gate off vs on
(`thermal_cap`), reporting per-species (perch, pikeperch) mean biomass and the overshoot ratio.
Off ≠ On under identical seed proves the gate fired (silent-non-application excluded, as done for
the ceiling).

---

## 5. Config keys (all lowercase dot-separated; inert defaults)

| Key | Default | Meaning |
|---|---|---|
| `reproduction.thermal.gate.enabled` | `false` | Master switch (off → feature absent, bit-identical). |
| `reproduction.thermal.gate.series.file` | — | Path to the per-year × per-species SST sidecar. |
| `reproduction.thermal.gate.species.enabled.sp{idx}` | `false` | Per-species enable (sp4, sp5). |
| `reproduction.thermal.gate.mode` | `thermal_cap` | `thermal_cap` (mean-reducing) or `mean_preserving`. |
| `reproduction.thermal.gate.t50.sp{idx}` | `18.5` | Logistic midpoint °C (per species). |
| `reproduction.thermal.gate.slope.sp{idx}` | `1.5` | Logistic slope °C (per species). |
| `reproduction.thermal.gate.tref.sp{idx}` | `20.0` | Reference °C for `thermal_cap` normalization (warm year → factor≈1; revisit per open-question 1). |
| `reproduction.thermal.gate.floor` | `0.0` | Lower clamp on the factor, `[0,1]`. |
| `reproduction.thermal.gate.start.year` | first series year | sim-year → real-year offset. |

Config-validation note (per CLAUDE.md): keys built from f-string `sp{sp}` are captured by the
`config_validation` AST walker the same way `reproduction.rv.gate.species.enabled.sp{sp}` is; the
integration test `test_from_dict_warn_mode_clean_on_example_configs` must stay warning-free. New
`EngineConfig` fields: `thermal_gate_factor_by_index` (`(n_years, n_species)` | None),
`thermal_gate_enabled` (`(n_species,)` bool | None), `thermal_gate_offset` (int) — all
`None`/`0` when disabled.

---

## 6. Determinism & parity

- Master switch off → loader returns `None`; the `reproduction()` block is skipped;
  outputs are **bit-identical** to baseline. This is asserted by a parity test (baltic gate-off
  == pre-feature baseline) exactly as the RV gate and ceiling did.
- Gate-on is deterministic under fixed RNG seeds; the factor is a pure function of precomputed
  config, no engine state.

---

## 7. Error handling (fail-fast, no silent fallback)

- Master on but `series.file` empty / missing → `ValueError` / `FileNotFoundError` naming the key.
- Sidecar with no rows, wrong/missing columns, non-contiguous years, NaN or out-of-range temps →
  `ValueError` naming the file.
- No species enabled while master on → `ValueError`.
- Unknown `mode`, `floor` outside `[0,1]`, non-positive/implausible `slope` → `ValueError`.
- Builder: missing `thetao` product/variable → loud error (Risk R1), never a synthetic substitute.

---

## 8. Testing strategy

- **Loader unit** (`tests/`): off → all-None; each fail-fast branch; logistic correctness
  (monotone, `r(t50)=0.5`); `thermal_cap` clip to `[floor,1]`; `mean_preserving` factor mean over
  the run window ≈ 1; offset wraparound.
- **Pure-helper unit**: 1.0 when off / for disabled species; correct per-year indexing incl.
  wraparound over a run longer than the series.
- **Engine integration**: (a) inert-by-default parity (gate-off == baseline, bit-identical);
  (b) gate-on changes percid recruitment deterministically (off ≠ on under fixed seed).
- **Builder unit**: small synthetic `thetao` NetCDF → correct per-year per-species summer means
  over the habitat footprint; surface-layer + per-species window selection; loud failure when
  `thetao` absent.
- **A/B diagnostic**: runs and emits percid overshoot numbers off vs on.

Run with `.venv/bin/python -m pytest`; lint `.venv/bin/ruff check osmose/ tests/` +
`ruff format --check`.

---

## 9. Risks & caveats

- **R1 — CMEMS credential prerequisite.** `thetao` is not cached; downloading it uses the same
  CMEMS product as `so` and therefore the same credentials, which are entangled with the owed
  rotation (see MEMORY security item). This spec does **not** resolve the rotation; the builder
  must fail loudly if the download cannot run, and the mechanism (loader/helper/wiring) is fully
  testable with a synthetic/example sidecar independent of the download.
- **R2 — Overshoot may not be recruitment-driven.** If percid overshoot is dominated by weak
  adult mortality (missing cannibalism/predation), even the mean-reducing mode will not damp it.
  Honest-negative outcome is acceptable and expected to be reported plainly (as with the ceiling).
- **R3 — Direction risk.** `mean_preserving` mode can *worsen* overshoot (RV-gate lesson); it is
  realism-only and must not be presented as the fix. Default is `thermal_cap`.
- **R4 — Short horizon.** `nyear`=15 gives few years for `mean_preserving` normalization
  (small-sample denominator noise). `thermal_cap` (no run-window mean) is less exposed to this.
- **R5 — Footprint sensitivity.** The habitat-cell averaging choice affects the index; it is made
  explicit (movement-map footprint), documented, and unit-tested so it is auditable.

---

## 10. Open questions for user review

1. `tref` default for `thermal_cap` — pick a fixed value (e.g. 19–20 °C) now, or derive it from
   the series (e.g. the observed warm-year percentile) in the builder?
2. Whether the two percids should share one `mode`/`floor` (as specced) or get fully independent
   mode selection. Current spec: shared mode/floor, per-species response params — the minimal
   surface that still honors perch/pikeperch differences.
