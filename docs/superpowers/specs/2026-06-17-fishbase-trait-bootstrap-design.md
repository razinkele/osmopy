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

1. **Access:** in-app HTTP client (`osmose/fishbase.py`), querying the rOpenSci FishBase
   REST API; **not** an MCP server (the interactive UI needs an in-app path anyway) and
   not a bundled snapshot (stale/limited).
2. **Resolution:** when FishBase has many estimates per trait, take the **median** and
   surface **count (n) + min–max range** for transparency.
3. **Apply model:** **review panel with per-trait accept** — fetch → table (trait |
   current | FishBase median | n | range | checkbox) → "Apply selected" → write to
   `state.config`. Non-destructive; only ticked rows change.
4. **Coverage:** **FishBase + SeaLifeBase auto** — query FishBase first, fall back to
   SeaLifeBase (a `server` param) when no fish match.
5. **HTTP dependency:** use **stdlib `urllib`** (no new prod/runtime dependency) to avoid
   another deploy/env change; one mockable seam for tests.
6. **UI placement:** a per-species **"Bootstrap from FishBase"** button on the species
   setup panel, opening a review modal scoped to that species.

## Architecture & components

### `osmose/fishbase.py` (pure, no Shiny import — fully unit-testable)

- `resolve_species(name: str, *, db: str | None = None) -> list[SpecMatch]`
  - Accepts scientific (`"Gadus morhua"`) or common (`"Atlantic cod"`) name.
  - Returns `SpecMatch(spec_code, scientific_name, common_name, db)` candidates.
  - Tries FishBase first; if no match, tries SeaLifeBase. Multiple hits → all candidates
    (UI disambiguates). `db` forces a specific database.
- `fetch_traits(spec_code: int, db: str) -> dict[str, TraitEstimate]`
  - `TraitEstimate = {value: float (median), n: int, min: float, max: float, unit: str}`.
  - Pulls and aggregates from the FishBase tables (see Trait mapping). Traits with no data
    are simply absent from the dict (partial coverage is normal).
- `TRAIT_MAP`: ordered mapping of FishBase source field → OSMOSE key-pattern stem + unit.
- `_get_json(path, params) -> list[dict] | dict`: the single network seam (stdlib
  `urllib.request`, timeout, JSON parse). **All tests mock this.** On failure raises
  `FishBaseUnavailable`.
- On-disk JSON cache (keyed by path+params) under a cache dir with a TTL, so repeated
  fetches/dev iterations don't re-hit the API. Cache dir overridable by env for test
  isolation (mirrors `OSMOSE_FEEDBACK_FILE`/`OSMOSE_RESULTS_DIR` pattern).
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

| FishBase source (table.field)         | OSMOSE key pattern                              | Unit       |
|---------------------------------------|-------------------------------------------------|------------|
| popgrowth.Loo                         | `species.linf.sp{i}`                            | cm         |
| popgrowth.K                           | `species.k.sp{i}`                               | year⁻¹     |
| popgrowth.to                          | `species.t0.sp{i}`                              | year       |
| species.Length (or popgrowth tmax/Lmax) | `species.lmax.sp{i}`                          | cm         |
| poplw.a                               | `species.length2weight.condition.factor.sp{i}` | W=a·Lᵇ     |
| poplw.b                               | `species.length2weight.allometric.power.sp{i}` | —          |
| maturity.Lm                           | `species.maturity.size.sp{i}`                   | cm         |
| species.LongevityWild (or popgrowth tmax) | `species.lifespan.sp{i}`                    | year       |

**Unit-convention check (implementation gate):** FishBase length–weight is conventionally
`W(g) = a · L(cm)^b`, which matches OSMOSE's `length2weight.condition.factor` /
`allometric.power`. This will be verified with a TDD test against a known species
(*Gadus morhua*): the bootstrapped a/b must reproduce a sane weight-at-length (e.g. a
~80 cm cod ≈ a few kg). Lengths (Loo/Lmax/Lm) are in cm in both. The exact rOpenSci API
field names/endpoints (`/popgrowth`, `/poplw`, `/maturity`, `/species`, `/taxa` or
`/comnames` for resolution; `Loo`,`K`,`to`,`a`,`b`,`Lm`,`Length`,`LongevityWild`) are
confirmed against the live API while recording the test fixtures.

## Error handling

- `FishBaseUnavailable` → non-blocking UI notification ("FishBase unavailable — try again
  later"); session never crashes.
- `FishBaseNoMatch` → clear "no FishBase/SeaLifeBase record for «name»" message.
- Partial coverage → review table simply omits traits with no data; never blocks Apply.
- Malformed/changed API payload → defensive parsing per field; a field that fails to parse
  is skipped (logged), not fatal.

## Testing

- **Pure client** (`osmose/fishbase.py`): unit tests mocking `_get_json` with **recorded
  JSON fixtures** (a real FishBase response for *Gadus morhua* + a SeaLifeBase example).
  No network in CI (the clean-venv rule). Cover: median/range aggregation, name resolution
  (scientific, common, multi-candidate, none), SeaLifeBase fallback, trait mapping +
  units, cache hit/miss, `FishBaseUnavailable` on injected network error.
- **UI**: controller-level test of fetch→review→apply writing the right config keys; one
  Playwright e2e with the client mocked (no live API).
- Full suite + ruff + pyright clean before merge.

## Out of scope (v1)

- Locality/region filtering of studies (median over all studies only).
- Writing provenance (study citations) into the config.
- Bootstrapping non-trait spatial inputs (distribution/movement maps).
- An MCP server wrapper (can be added later on top of the same client).
