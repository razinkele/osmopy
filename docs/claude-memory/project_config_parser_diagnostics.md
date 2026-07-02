---
name: project_config_parser_diagnostics
description: Config parser diagnostics (structured line-located parse issues in OsmoseConfigReader + check_config.py CLI) SHIPPED to origin/master 2026-06-04. Additive-only, parity-safe.
metadata:
  node_type: memory
  type: project
  originSessionId: d8b85e1b-0860-44a3-8a89-b2d35b87a866
---

**Better config parser diagnostics** (a clean DX backlog pick) — `OsmoseConfigReader` now collects structured, line-located parse issues. Merged fast-forward to master + **pushed to origin/master 2026-06-04** (`395cdd8..d335964`, branch `feature/config-parser-diagnostics` deleted, origin synced). 15 new tests; 5 impl commits; additive-only.

## What shipped (all in osmose/config/reader.py + a CLI)
- `@dataclass(frozen=True) ConfigDiagnostic(file, lineno: int|None, line, reason, detail)`; 6 reasons: `unparseable | empty_key | duplicate_key | circular_ref | missing_subconfig | path_escape`. `_ERROR_REASONS` frozenset (errors = unparseable + the 3 recursive; warnings = empty_key + duplicate_key). `diagnostics_have_errors()`, `format_diagnostics()` (grouped-by-file report + summary; lineno-None renders without `:None:`).
- `OsmoseConfigReader` gains `self.diagnostics` (init + reset in `read()`). `read_file` enriched: `enumerate(f,1)` line numbers; emits `unparseable` (len!=2), `empty_key` (key=="" AND post-rstrip value!="" — a `=value` that lost its key), within-file `duplicate_key` (non-empty lowercased keys, `seen_keys` local per file; detail names the EARLIER raw key via key_case_map.get BEFORE overwrite). `_read_recursive` emits `circular_ref`/`path_escape`/`missing_subconfig` (lineno=None) alongside the kept log warnings.
- `scripts/check_config.py` CLI: `--config <master>`, prints `format_diagnostics`, exits 1 iff an ERROR-class diagnostic (warnings exit 0); argparse `p.error` (exit 2) on missing file.
- `tests/test_config_reader_diagnostics.py` (15) incl. parametrized all-5-shipped-masters → `diagnostics==[]`.

## ADDITIVE-ONLY / parity-safe (the core guarantee, verified)
The returned dict + `key_case_map` + `skipped_lines` are **byte-identical** to before (verified by reconstructing the pre-feature loop and diffing on a mixed config). `skipped += 1` stays only on the unparseable branch; empty_key/duplicate don't increment. `result[""]=value` from `,,`/`=value` lines is STILL stored exactly as before (we did NOT fix that latent quirk — fixing it would be a behavior/parity change, out of scope). So no engine-parity risk.

## In-loop review caught the headline issue (2 rounds on spec AND plan)
- **Spec round 1 (BLOCKER):** within-file duplicate detection fired on SHIPPED `eec_full` — `data/eec_full/eec_param-output.csv` has `,,` spacer rows that parse to `key=""` (`["",","]`→rstrip→`""`); repeated `,,` collided on the empty key. My "real config→no diagnostics" guarantee only tested baltic. **Insight:** empty-key is a pre-existing SILENT footgun (`=value` parses len==2, pollutes the dict) → added the `empty_key` category. Fix: empty_key only when value!=""; `,,` blank rows benign; empty keys excluded from dup tracking; storage unchanged. Round 2 confirmed all 5 masters (59 files) diagnostic-free.
- **Plan round 1 (2 MAJORs):** (a) 2 shipped-master test paths wrong — eec/minimal use `osm_all-parameters.csv` NOT `eec_`/`minimal_` (skip-fallback was silently dropping 2/5 coverage). (b) the canned multi-line `ConfigDiagnostic(...)` blocks aren't ruff-format-clean (ruff check passes but `ruff format --check` fails = the green-check/red-format CI trap from [[feedback_ci_lint_is_check_plus_format]]) → added a RUFF FORMAT-FIRST note. Plan round 2 + per-task reviews clean.
- Subagent-driven build; implementer/reviewer note: the `format_diagnostics` conditional `— <line>` (omits trailing " — " when empty) is BETTER than the literal spec and unreachable in practice (all lineno-present diagnostics carry a non-empty line) → accepted, not "fixed".

## Gotchas carried
- `scripts/` is NOT CI-linted (CI lints `osmose/ ui/ tests/`); the CLI is import-tested via `importlib.util.spec_from_file_location` (scripts/ isn't a package). Tests relying on relative `Path("data/...")` need pytest cwd=repo-root (repo convention).
- **Next: pick a fresh backlog item.** See [[project_feature_improvements_backlog]] (open: sensitivity explorer ALREADY EXISTS — don't rebuild; remaining clean picks = config presets, scenario diff, `__slots__`/mutable-SchoolState perf, property-based tests, trophic-network animation).
