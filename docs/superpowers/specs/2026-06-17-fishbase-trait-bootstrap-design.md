# FishBase/SeaLifeBase species-trait bootstrap — design

**Date:** 2026-06-17
**Status:** approved (brainstorming), pending implementation plan
**Backlog item:** "FishBase / SeaLifeBase auto-bootstrap species traits" (MCP / integrations)

## Problem & goal

Setting up a new OSMOSE species means hand-entering life-history traits (von Bertalanffy
growth, length–weight, maturity, longevity) gathered from literature. This is tedious and
error-prone. **Goal:** let a user populate a focal species' trait fields from FishBase
(fish) / SeaLifeBase (non-fish) with a single fetch + review step, as sensible *starting
values* they then calibrate.

Non-goal: authoritative/locality-perfect parameterisation. The bootstrap produces a
defensible first guess (median across studies) with the spread shown; calibration remains
the source of truth.

## Decisions (from brainstorming)

1. **Access:** in-app HTTP client (`osmose/fishbase.py`) that **downloads the FishBase
   parquet-snapshot tables from Source Cooperative and queries them locally**; **not** an
   MCP server (the interactive UI needs an in-app path anyway) and not a bundled in-repo
   snapshot (stale/limited + CC-BY-NC redistribution concerns).
   - **Source correction (verified 2026-06-17):** the old rOpenSci REST API
     (`fishbase.ropensci.org/taxa?...`, per-species JSON) is **deprecated** — it 404s and
     serves a **self-signed cert** (`ssl_verify_result=18`), so a TLS-verifying prod
     client cannot use it. rfishbase 5 moved to **parquet tables on Source Cooperative**.
     Base (valid TLS, HTTP range supported):
     `https://data.source.coop/cboettig/fishbase/fb/v24.07/parquet/<table>.parquet`
     (FishBase) and `.../slb/v24.07/parquet/<table>.parquet` (SeaLifeBase). This honors
     the "live, in-app, not-bundled" intent; only the transport changed (whole-table
     parquet download + local query, vs per-species JSON GET).
   - **License:** the snapshots are CC-BY-NC (Carl Boettiger / FishBase.org). We fetch at
     runtime (no redistribution) and **attribute in the UI** ("Data: FishBase via rOpenSci
     / Source Cooperative, CC-BY-NC").
2. **Resolution:** when FishBase has many estimates per trait, take the **median** and
   surface **count (n) + min–max range** for transparency.
3. **Apply model:** **review panel with per-trait accept** — fetch → table (trait |
   current | FishBase median | n | range | checkbox) → "Apply selected" → write to
   `state.config`. Non-destructive; only ticked rows change.
4. **Coverage:** **FishBase + SeaLifeBase auto** — query FishBase first, fall back to
   SeaLifeBase (a `server` param) when no fish match.
5. **Dependencies:** **stdlib `urllib`** fetches the parquet bytes; **pandas** (already a
   runtime dep) reads them. **pandas does NOT require pyarrow** — it's only present in the
   dev `.venv` transitively via the unmanaged `copernicusmarine` install (the clean-venv
   trap). So **declare `pyarrow>=14` as a runtime dependency** in `pyproject.toml`. The
   prod shiny env already has pyarrow 21 (verified), so no prod break; add pyarrow to
   `deploy.sh`'s ensured-packages defensively. One mockable seam for tests; the clean-venv
   check must *run* the parquet tests (not just `--collect-only`, which wouldn't execute
   the read).
6. **UI placement:** a per-species **"Bootstrap from FishBase"** button on the species
   setup panel, opening a review modal scoped to that species.

## Architecture & components

### `osmose/fishbase.py` (pure, no Shiny import — fully unit-testable)

- `_load_table(table: str, db: str = "fb") -> pd.DataFrame`: the single network seam.
  Builds the URL `https://data.source.coop/cboettig/fishbase/{db}/v24.07/parquet/{table}.parquet`
  (`db` ∈ `{"fb","slb"}`), fetches bytes via stdlib `urllib.request` (timeout, TLS
  verified), and `pd.read_parquet(io.BytesIO(...))`. On-disk cache: the raw `.parquet` is
  saved under a cache dir (TTL); a cache hit skips the network. Cache dir overridable by
  env (`OSMOSE_FISHBASE_CACHE_DIR`) for test isolation (mirrors `OSMOSE_RESULTS_DIR`).
  **All tests monkeypatch this** to return fixture DataFrames — no network in CI. On
  network/HTTP/timeout failure raises `FishBaseUnavailable`.
- `resolve_species(name: str, *, db: str | None = None) -> list[SpecMatch]`
  - Accepts scientific (`"Gadus morhua"`) or common (`"Atlantic cod"`) name; case-insensitive.
  - Queries the `species` table: scientific match on `Genus`+`Species`, common match on
    `FBname`. Returns `SpecMatch(spec_code, scientific_name, common_name, db)` candidates.
  - Tries FishBase (`db="fb"`) first; if no match, tries SeaLifeBase (`db="slb"`).
    Multiple hits → all candidates (UI disambiguates). `db` forces one database.
- `fetch_traits(spec_code: int, db: str) -> dict[str, TraitEstimate]`
  - `TraitEstimate = {value: float (median), n: int, min: float, max: float, unit: str}`.
  - Loads `popgrowth`, `poplw`, `maturity`, `species`, filters each by spec code, and
    aggregates each mapped column to median/n/min–max (see Trait mapping). Traits with no
    data are simply absent (partial coverage is normal).
- `TRAIT_MAP`: ordered mapping of (table, column) → OSMOSE key-pattern stem + unit.
- Exceptions: `FishBaseUnavailable` (network/timeout/HTTP error), `FishBaseNoMatch`
  (name resolves to nothing in either DB).

### UI surface

- A **"Bootstrap from FishBase"** button per species on the setup species panel.
- Opens a modal: scientific-name input (prefilled from `species.name.sp{i}`, editable) →
  **Fetch** → (candidate picker if ambiguous) → **review table** → **Apply selected**.
- Apply writes `state.config[field.resolve_key(i)] = str(value)` for each ticked trait,
  then triggers the existing form-refresh path so inputs update.
- Lives in **`ui/components/fishbase_bootstrap.py`** — a reusable modal builder plus a
  small `fishbase_bootstrap_server(input, output, session, state)` helper invoked once
  from `setup_server`, keeping `setup.py` focused. The per-species buttons share one modal
  parameterised by the species index.

## Data flow

1. User clicks Bootstrap on species *i*; name input prefilled from `species.name.sp{i}`.
2. `resolve_species(name)` → 0 matches → "no match" message; 1 → proceed; >1 → candidate
   picker (scientific + common + db).
3. `fetch_traits(spec_code, db)` → review table rows for each resolved trait:
   `trait label | current config value | FishBase median | n | range | ☑`.
4. User ticks desired traits → **Apply selected** → `state.config` updated → forms refresh.

## Trait mapping (FishBase → OSMOSE)

Column names are **verified against the v24.07 parquet schema** (2026-06-17), with the
*Gadus morhua* (SpecCode 69) median shown as the fixture anchor:

| Source table.column        | OSMOSE key pattern                              | Unit    | cod median |
|----------------------------|-------------------------------------------------|---------|------------|
| `popgrowth.Loo`            | `species.linf.sp{i}`                            | cm      | 110 (n=108) |
| `popgrowth.K`              | `species.k.sp{i}`                               | year⁻¹  | 0.163 (n=108) |
| `popgrowth.to`             | `species.t0.sp{i}`                              | year    | −0.08 (n=47) |
| `species.Length`           | `species.lmax.sp{i}`                            | cm      | 200 |
| `poplw.a`                  | `species.length2weight.condition.factor.sp{i}` | W=a·Lᵇ  | 0.00723 (n=52) |
| `poplw.b`                  | `species.length2weight.allometric.power.sp{i}` | —       | 3.073 (n=52) |
| `maturity.Lm`              | `species.maturity.size.sp{i}`                   | cm      | (per spec) |
| `species.LongevityWild`    | `species.lifespan.sp{i}`                        | year    | 25 |

**Schema quirks (real, must be handled):**
- The `maturity` table keys on **`Speccode`** (lowercase), while `popgrowth`/`poplw`/
  `species` use **`SpecCode`**. The filter must use the right column per table.
- `popgrowth` rows carry a `Sex` column (mixed/male/female/unsexed); v1 takes the median
  across all rows (per the resolution decision) — no sex filtering.

**Unit-convention check (verified):** `W(g)=a·L(cm)ᵇ` matches OSMOSE's
`length2weight.condition.factor`/`allometric.power`. Sanity-checked: cod a=0.00723,
b=3.073 → a 110 cm cod ≈ 13 kg (plausible). A TDD test reasserts this on the recorded
fixture so a future data/format change can't silently break it. Lengths (Loo/Length/Lm)
are cm in both systems.

## Error handling

- `FishBaseUnavailable` → non-blocking UI notification ("FishBase unavailable — try again
  later"); session never crashes.
- `FishBaseNoMatch` → clear "no FishBase/SeaLifeBase record for «name»" message.
- Partial coverage → review table simply omits traits with no data; never blocks Apply.
- Malformed/changed API payload → defensive parsing per field; a field that fails to parse
  is skipped (logged), not fatal.

## Testing

- **Pure client** (`osmose/fishbase.py`): unit tests monkeypatching `_load_table` to
  return **small recorded fixture DataFrames** — tiny parquet slices of the real
  `popgrowth`/`poplw`/`maturity`/`species` tables for *Gadus morhua* (+ one SeaLifeBase
  species), checked into `tests/fixtures/fishbase/`. No network in CI (the clean-venv
  rule). Cover: median/range aggregation, name resolution (scientific, common,
  multi-candidate, none), SeaLifeBase fallback, the `Speccode`-vs-`SpecCode` quirk, trait
  mapping + the a/b weight-at-length assertion, cache hit/miss, `FishBaseUnavailable` on
  injected network error.
- **UI**: controller-level test of fetch→review→apply writing the right config keys; one
  Playwright e2e with the client mocked (no live API).
- Full suite + ruff + pyright clean before merge.

## Out of scope (v1)

- Locality/region filtering of studies (median over all studies only).
- Writing provenance (study citations) into the config.
- Bootstrapping non-trait spatial inputs (distribution/movement maps).
- An MCP server wrapper (can be added later on top of the same client).
