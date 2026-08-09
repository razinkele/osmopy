"""Unknown-key validation for OSMOSE EngineConfig.from_dict().

Spec: docs/superpowers/specs/2026-04-19-config-validation-design.md (557ee1b).

Match order (fast -> slow):
  1. Exact literal lookup in KnownKeys.literals (O(1) set).
  2. Normalized-pattern lookup in KnownKeys.patterns (O(1) set) --
     converts user's concrete key (species.linf.sp47) to pattern form
     (species.linf.sp{idx}) segment-by-segment.
  3. Regex match across KnownKeys.regexes (~15 compiled patterns).
  4. Miss -> UnknownKey with optional difflib suggestion (cutoff 0.85
     against normalized pattern-form strings on both sides).
"""

from __future__ import annotations

import ast
import difflib
import re
from dataclasses import dataclass

from osmose.logging import setup_logging

log = setup_logging("osmose.config")

_INDEX_SUFFIXES = (
    ("fsh", re.compile(r"^fsh\d+$")),
    ("mpa", re.compile(r"^mpa\d+$")),
    ("map", re.compile(r"^map\d+$")),
    ("age", re.compile(r"^age\d+$")),
    ("sz", re.compile(r"^sz\d+$")),
    ("sp", re.compile(r"^sp\d+$")),
    ("rsc", re.compile(r"^rsc\d+$")),
)

_VALID_MODES = ("off", "warn", "error")
_SUGGESTION_CUTOFF = 0.85

# Reader-honored keys the AST walker cannot resolve statically, split into two buckets (#123):
#   _ALLOWLIST_PY_HONORED  — the Python engine reads it, OR reader-injected osmose.* metadata.
#   _ALLOWLIST_JAVA_ONLY   — a real OSMOSE/Java key the Python engine provably does NOT read;
#                            setting it on a Python run has no effect (warned about, see #123).
# The membership of each bucket is proven by tests/test_issue_123_known_but_unread_keys.py
# (read-clearance + metadata-clearance guards) — do NOT edit a key's bucket by eyeballing a
# comment; move it, then let the guard confirm. Their union is byte-identical to the pre-#123
# allowlist, so build_known_keys() and all unknown-key validation are unchanged.
_ALLOWLIST_PY_HONORED: frozenset[str] = frozenset(
    [
        "evolution.trait.{name}.envvar.sp{idx}",
        "evolution.trait.{name}.mean.sp{idx}",
        "evolution.trait.{name}.nlocus.sp{idx}",
        "evolution.trait.{name}.nval.sp{idx}",
        "evolution.trait.{name}.target",
        "evolution.trait.{name}.var.sp{idx}",
        "fisheries.movement.file.map{idx}",
        "ltl.depletable.enabled",
        "ltl.depletable.floor",
        "ltl.regrowth.rate.default",
        "ltl.regrowth.rate.rsc{idx}",
        "module.bioeconomics.enabled",
        "module.population.initialisation.enabled",
        "movement.species.map{idx}",
        "osmose.configuration.a2.depletion",
        "osmose.configuration.background",
        "osmose.configuration.bioen",
        "osmose.configuration.fishing",
        "osmose.configuration.genetics",
        "osmose.configuration.grid",
        "osmose.configuration.initialization",
        "osmose.configuration.ltl",
        "osmose.configuration.migration",
        "osmose.configuration.mortality.additional",
        "osmose.configuration.mortality.fishing",
        "osmose.configuration.mortality.predation",
        "osmose.configuration.mortality.starvation",
        "osmose.configuration.movement",
        "osmose.configuration.output",
        "osmose.configuration.oxygen",
        "osmose.configuration.plankton",
        "osmose.configuration.predation",
        "osmose.configuration.reproduction",
        "osmose.configuration.simulation",
        "osmose.configuration.species",
        "osmose.version",
        "output.size.enabled",
        "output.ssb.enabled",
        "output.ssb.netcdf.enabled",
        "output.tl.enabled",
        "output.yield.abundance.enabled",
        "output.yield.abundance.netcdf.enabled",
        "oxygen.factor",
        "oxygen.filename",
        "oxygen.nsteps.year",
        "oxygen.offset",
        "oxygen.varname",
        "simulation.incoming.flux.enabled",
        "species.age.sp{idx}",
        "species.biomass.nsteps.year",
        "species.biomass.total.sp{idx}",
        "species.length.sp{idx}",
        "species.lw.allpower.sp{idx}",
        "species.lw.condition.factor.sp{idx}",
        "species.nclass.sp{idx}",
        "species.regrowth.rate.sp{idx}",
        "species.size.proportion.sp{idx}",
        "species.tl.sp{idx}",
        "species.trophic.level.sp{idx}",
        "species.type.sp{idx}",
    ]
)

_ALLOWLIST_JAVA_ONLY: frozenset[str] = frozenset(
    [
        "economic.output.stage",
        "fisheries.check.enabled",
        "fisheries.movement.fishery.map{idx}",
        "fisheries.name.fsh{idx}",
        "fisheries.period.number.fsh{idx}",
        "fisheries.period.start.fsh{idx}",
        "fisheries.rate.byperiod.fsh{idx}",
        "grid.java.classname",
        "grid.lowright.lat",
        "grid.lowright.lon",
        "grid.mask.file",
        "grid.upleft.lat",
        "grid.upleft.lon",
        "ltl.conversion2tons.rsc{idx}",
        "ltl.java.classname",
        "ltl.nstep",
        "mortality.fishing.recruitment.age.sp{idx}",
        "mortality.fishing.recruitment.size.sp{idx}",
        "output.abundance.age1.enabled",
        "output.abundance.bytl.enabled",
        "output.abundance.byweight.enabled",
        "output.abundance.enabled",
        "output.age.at.death.enabled",
        "output.bioen.enet.enabled",
        "output.biomass.bytl.enabled",
        "output.biomass.byweight.enabled",
        "output.biomass.enabled",
        "output.csv.separator",
        "output.cutoff.enabled",
        "output.diet.composition.byage.enabled",
        "output.diet.composition.bysize.enabled",
        "output.diet.composition.netcdf.enabled",
        "output.diet.pressure.byage.enabled",
        "output.diet.pressure.bysize.enabled",
        "output.diet.pressure.enabled",
        "output.diet.pressure.netcdf.enabled",
        "output.diet.stage.structure",
        "output.diet.stage.threshold.sp{idx}",
        "output.diet.success.enabled",
        "output.dir.path",
        "output.file.prefix",
        "output.fisheries.byage.enabled",
        "output.fisheries.bysize.enabled",
        "output.fisheries.enabled",
        "output.flush.enabled",
        "output.meansize.byage.enabled",
        "output.meantl.byage.enabled",
        "output.meantl.bysize.enabled",
        "output.meanweight.byage.enabled",
        "output.mortality.additional.byage.enabled",
        "output.mortality.additional.bysize.enabled",
        "output.mortality.additionaln.byage.enabled",
        "output.mortality.additionaln.bysize.enabled",
        "output.mortality.enabled",
        "output.mortality.perspecies.byage.enabled",
        "output.mortality.perspecies.bysize.enabled",
        "output.nschool.enabled",
        "output.number.of.eggs.bysize.enabled",
        "output.restart.recordfrequency.ndt",
        "output.restart.spinup",
        "output.size.catch.enabled",
        "output.spatial.egg.enabled",
        "output.spatial.fisheries.enabled",
        "output.spatial.ltl.enabled",
        "output.spatial.size.enabled",
        "output.spatial.yield.abundance.enabled",
        "output.start.year",
        "output.tl.catch.enabled",
        "output.weight.enabled",
        "output.yield.abundance.byage.enabled",
        "output.yield.abundance.bysize.enabled",
        "output.yield.biomass.byage.enabled",
        "output.yield.biomass.bysize.enabled",
        "output.yield.biomass.enabled",
        "population.initialization.method.sp{idx}",
        "predation.accessibility.stage.structure",
        "predation.accessibility.stage.threshold.sp{idx}",
        "simulation.ncpu",
        "simulation.nsimulation",
        "simulation.restart.enabled",
        "simulation.restart.file",
        "simulation.restart.recordfrequency.ndt",
        "simulation.restart.spinup.nyear",
        "species.conversion2tons.sp{idx}",
        "species.first.feeding.age.sp{idx}",
        "temperature.factor",
        "temperature.filename",
        "temperature.nsteps.year",
        "temperature.offset",
        "temperature.varname",
    ]
)

_SUPPLEMENTARY_ALLOWLIST: frozenset[str] = _ALLOWLIST_PY_HONORED | _ALLOWLIST_JAVA_ONLY

# #120 already warns on these two restart keys with a targeted message (config.py) — exclude them
# from #123's summary to avoid double-warning. They remain in _ALLOWLIST_JAVA_ONLY for partition
# completeness (they ARE java-only). Re-verify #120's warn-set at rebase (spec §"#120 overlap").
_RESTART_HANDLED_BY_120: frozenset[str] = frozenset(
    {"simulation.restart.file", "simulation.restart.enabled"}
)


def java_only_keys_set(cfg: dict) -> list[str]:
    """Real OSMOSE keys present in `cfg` that the Python engine does not read (inert on a Python
    run). Canonicalizes first (mirrors validate()); matches _ALLOWLIST_JAVA_ONLY literals + {idx}/
    {name} patterns; excludes the #120-owned restart keys. Returns a sorted list."""
    from osmose.config.aliases import canonicalize_config

    cfg, _ = canonicalize_config(cfg)
    java_only = _ALLOWLIST_JAVA_ONLY - _RESTART_HANDLED_BY_120
    literals = frozenset(p for p in java_only if "{idx}" not in p and "{name}" not in p)
    regexes = tuple(
        _compile_regex_for_pattern(p) for p in java_only if "{idx}" in p or "{name}" in p
    )
    hits = [k for k in cfg if k in literals or any(rx.match(k) for rx in regexes)]
    return sorted(hits)


# Deduped-once-per-process, like #120's _WARNED_UNSUPPORTED_RESTART. Cleared by an autouse test
# fixture (tests/test_issue_123_known_but_unread_keys.py). Placed at the Python-run seam
# (PythonEngine._prepare_run), NOT in from_dict — see spec §3 / Global Constraints.
_WARNED_JAVA_ONLY_KEYS: set[str] = set()
_MAX_NAMED_JAVA_ONLY_KEYS = (
    10  # cap the listed keys; the rest are counted (bundled demos set ~20-44)
)


def warn_unread_java_only_keys(cfg: dict) -> list[str]:
    """If `cfg` sets java-only keys, emit ONE deduped summary warning naming (up to
    _MAX_NAMED_JAVA_ONLY_KEYS of) them. Returns the full key list (empty if none). Call only from
    the Python-engine run seam."""
    keys = java_only_keys_set(cfg)
    if not keys:
        return keys
    # Dedup on the FULL key set so two configs differing only in the un-listed tail warn distinctly.
    fingerprint = ",".join(keys)
    if fingerprint not in _WARNED_JAVA_ONLY_KEYS:
        _WARNED_JAVA_ONLY_KEYS.add(fingerprint)
        shown = keys[:_MAX_NAMED_JAVA_ONLY_KEYS]
        more = (
            "" if len(keys) <= _MAX_NAMED_JAVA_ONLY_KEYS else f", and {len(keys) - len(shown)} more"
        )
        log.warning(
            "%d config key(s) are valid OSMOSE keys the Python engine does not implement; on this "
            "engine they have no effect. Use the Java engine if you need them: %s%s (see issue #123).",
            len(keys),
            ", ".join(shown),
            more,
        )
    return keys


@dataclass(frozen=True)
class UnknownKey:
    key: str
    suggestion: str | None


@dataclass(frozen=True)
class KnownKeys:
    patterns: frozenset[str]
    literals: frozenset[str]
    regexes: tuple[tuple[str, re.Pattern], ...]


def _normalize_key_to_pattern(key: str) -> str:
    """Convert a concrete user key to its {idx}-pattern form segment-by-segment."""
    segments = key.split(".")
    for i, seg in enumerate(segments):
        for token, pattern in _INDEX_SUFFIXES:
            if pattern.fullmatch(seg):
                segments[i] = f"{token}{{idx}}"
                break
    return ".".join(segments)


def _compile_regex_for_pattern(pattern: str) -> re.Pattern:
    # {idx} -> numeric index segment (sp0, fsh1, ...); {name} -> free-form
    # word segment for keys indexed by a name rather than a number, e.g. the
    # trait name in evolution.trait.<name>.* (engine parses it as \w+).
    escaped = re.escape(pattern).replace(r"\{idx\}", r"\d+").replace(r"\{name\}", r"\w+")
    return re.compile(f"^{escaped}$")


def _read_config_source() -> str:
    """Read osmose/engine/config.py via importlib.resources (test-hookable)."""
    import importlib.resources

    return (
        importlib.resources.files("osmose.engine").joinpath("config.py").read_text(encoding="utf-8")
    )


# Engine modules other than config.py that also call cfg.get(...) directly.
# Extended in C1 (2026-05-05) to fix the silent UI-engine drift on movement
# map keys read by movement_maps.py — the AST walker previously saw only
# config.py and missed every key these modules read. Add new entries here
# whenever a new engine module starts reading config keys directly.
_EXTRA_ENGINE_SOURCES: tuple[str, ...] = (
    "movement_maps.py",
    "background.py",
    "resources.py",
    "grid.py",
    "physical_data.py",
    "_netcdf.py",
    "simulate.py",  # extended 2026-05-08: temperature.value, oxygen.value reads
    "__init__.py",  # extended 2026-05-08: PythonEngine._resolve_grid reads grid.* keys
)


def _read_extra_engine_sources() -> dict[str, str]:
    """Read each `_EXTRA_ENGINE_SOURCES` file; missing files are skipped (test-hookable)."""
    import importlib.resources

    sources: dict[str, str] = {}
    base = importlib.resources.files("osmose.engine")
    for filename in _EXTRA_ENGINE_SOURCES:
        try:
            sources[filename] = base.joinpath(filename).read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            continue
    return sources


def _extract_literal_keys_from_config_py(tree: ast.AST) -> set[str]:
    """Walk an AST and extract OSMOSE config-key literals."""
    helper_names = {
        "_get",
        "_enabled",
        "_species_float",
        "_species_float_optional",
        "_species_int",
        "_species_int_optional",
        "_species_str",
        "_species_str_optional",
    }
    out: set[str] = set()

    def _capture_string(s: str) -> None:
        out.add(re.sub(r"\{(i|fsh|sp|idx)\}", "{idx}", s))

    def _render_fstring(joined: ast.JoinedStr) -> str | None:
        pieces: list[str] = []
        for part in joined.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                pieces.append(part.value)
            elif isinstance(part, ast.FormattedValue):
                if isinstance(part.value, ast.Name):
                    pieces.append("{idx}")
                else:
                    return None
            else:
                return None
        return "".join(pieces)

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
        ):
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                _capture_string(first.value)
            elif isinstance(first, ast.JoinedStr):
                rendered = _render_fstring(first)
                if rendered is not None:
                    _capture_string(rendered)
            elif (
                isinstance(first, ast.Call)
                and isinstance(first.func, ast.Attribute)
                and first.func.attr == "format"
                and isinstance(first.func.value, ast.Constant)
                and isinstance(first.func.value.value, str)
            ):
                _capture_string(first.func.value.value)

        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in helper_names and len(node.args) >= 2:
                second = node.args[1]
                if isinstance(second, ast.Constant) and isinstance(second.value, str):
                    _capture_string(second.value)
                elif isinstance(second, ast.JoinedStr):
                    rendered = _render_fstring(second)
                    if rendered is not None:
                        _capture_string(rendered)
            for kw in node.keywords:
                if kw.arg == "key" and isinstance(kw.value, ast.Constant):
                    if isinstance(kw.value.value, str):
                        _capture_string(kw.value.value)

        elif isinstance(node, ast.Subscript):
            sl = node.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                _capture_string(sl.value)

        elif isinstance(node, ast.Compare) and len(node.ops) == 1:
            if isinstance(node.ops[0], ast.In):
                if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
                    if "." in node.left.value:
                        _capture_string(node.left.value)

        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.JoinedStr):
                rendered = _render_fstring(node.value)
                if rendered is not None and "." in rendered and " " not in rendered:
                    _capture_string(rendered)

    return out


_KNOWN_KEYS_CACHE: dict[str, KnownKeys] = {}


def build_known_keys() -> KnownKeys:
    """Union of ParameterRegistry field patterns + AST-extracted reader keys.

    The full (schema + AST + supplementary) build is memoized; the degraded
    (schema-only + supplementary) fallback is NOT memoized so transient FS
    errors don't stick. Intentional divergence from spec's @functools.cache.
    """
    if "full" in _KNOWN_KEYS_CACHE:
        return _KNOWN_KEYS_CACHE["full"]

    from osmose.schema import build_registry

    reg = build_registry()
    pattern_strs: set[str] = {f.key_pattern for f in reg.all_fields()}
    pattern_strs |= _SUPPLEMENTARY_ALLOWLIST

    ast_ok = False
    try:
        source = _read_config_source()
        tree = ast.parse(source)
        pattern_strs |= _extract_literal_keys_from_config_py(tree)
        for filename, extra_source in _read_extra_engine_sources().items():
            try:
                pattern_strs |= _extract_literal_keys_from_config_py(ast.parse(extra_source))
            except SyntaxError as exc:
                log.info(
                    "config_validation: failed to parse %s (%s: %s); skipping.",
                    filename,
                    type(exc).__name__,
                    exc,
                )
        ast_ok = True
    except Exception as exc:
        log.info(
            "config_validation: AST source unavailable or walker failed "
            "(%s: %s); using schema-only allowlist for this call. "
            "Will retry on next call.",
            type(exc).__name__,
            exc,
        )

    patterns = frozenset(pattern_strs)

    def _has_placeholder(p: str) -> bool:
        return "{idx}" in p or "{name}" in p

    literals = frozenset(p for p in patterns if not _has_placeholder(p))
    regex_pairs = tuple((p, _compile_regex_for_pattern(p)) for p in patterns if _has_placeholder(p))
    result = KnownKeys(patterns=patterns, literals=literals, regexes=regex_pairs)

    if ast_ok:
        _KNOWN_KEYS_CACHE["full"] = result
    return result


def _clear_known_keys_cache() -> None:
    """Test hook -- equivalent to functools.cache.cache_clear()."""
    _KNOWN_KEYS_CACHE.clear()


def _suggest(normalized_key: str, patterns: frozenset[str]) -> str | None:
    matches = difflib.get_close_matches(
        normalized_key, list(patterns), n=1, cutoff=_SUGGESTION_CUTOFF
    )
    return matches[0] if matches else None


def _check(cfg_key: str, known: KnownKeys) -> UnknownKey | None:
    """Classify a single cfg key — literal fast-path, normalized-pattern, then regex fallback."""
    if cfg_key in known.literals:
        return None
    normalized = _normalize_key_to_pattern(cfg_key)
    if normalized in known.patterns:
        return None
    for _, compiled in known.regexes:
        if compiled.match(cfg_key):
            return None
    suggestion = _suggest(normalized, known.patterns)
    return UnknownKey(key=cfg_key, suggestion=suggestion)


def validate(cfg: dict, mode: str) -> list[UnknownKey]:
    """Detect unknown config keys and dispatch per mode.

    mode:
      - "off"   : return the list; emit a single info-line nudge if non-empty.
      - "warn"  : return the list; log one warning per unknown (with suggestion).
      - "error" : if any unknowns, collect ALL and raise ValueError.

    Raises ValueError for invalid mode strings (case-sensitive).
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"validation.strict.enabled must be one of {list(_VALID_MODES)!r}; got {mode!r}"
        )

    from osmose.config.aliases import canonicalize_config

    cfg, _ = canonicalize_config(cfg)
    known = build_known_keys()
    unknowns: list[UnknownKey] = []
    for key in cfg:
        result = _check(key, known)
        if result is not None:
            unknowns.append(result)

    if not unknowns:
        return unknowns

    if mode == "error":
        lines = ["Unknown OSMOSE config keys detected:"]
        for uk in unknowns:
            if uk.suggestion:
                lines.append(f"  - {uk.key!r}  (did you mean {uk.suggestion!r}?)")
            else:
                lines.append(f"  - {uk.key!r}")
        raise ValueError("\n".join(lines))

    if mode == "warn":
        for uk in unknowns:
            if uk.suggestion:
                log.warning(
                    "Unknown config key %r -- did you mean %r?",
                    uk.key,
                    uk.suggestion,
                )
            else:
                log.warning("Unknown config key %r", uk.key)
        return unknowns

    log.info(
        "Config has %d unknown keys; set validation.strict.enabled=warn for details.",
        len(unknowns),
    )
    return unknowns
