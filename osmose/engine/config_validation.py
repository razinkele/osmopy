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
)

_VALID_MODES = ("off", "warn", "error")
_SUGGESTION_CUTOFF = 0.85

# Reader-honored keys the AST walker cannot resolve statically -- e.g.,
# reader-injected metadata (osmose.configuration.*, osmose.version),
# legacy aliases, and keys built from caller-arg key_pattern two frames up.
# Populated empirically during the EEC/Baltic/eec_full sweep in Step 10.
# Prefer schema registration over allowlist growth -- allowlist is for keys
# that genuinely belong outside the schema.
_SUPPLEMENTARY_ALLOWLIST: frozenset[str] = frozenset(
    [
        # --- Reader-injected metadata keys (osmose.configuration.* and osmose.version) ---
        # Injected by OsmoseConfigReader when loading sub-config files; never
        # registered in the schema and never read by EngineConfig.from_dict.
        "osmose.version",
        "osmose.configuration.background",
        "osmose.configuration.fishing",
        "osmose.configuration.grid",
        "osmose.configuration.initialization",
        "osmose.configuration.migration",
        "osmose.configuration.mortality.additional",
        "osmose.configuration.mortality.fishing",
        "osmose.configuration.mortality.predation",
        "osmose.configuration.mortality.starvation",
        "osmose.configuration.movement",
        "osmose.configuration.output",
        "osmose.configuration.plankton",
        "osmose.configuration.predation",
        "osmose.configuration.reproduction",
        "osmose.configuration.simulation",
        "osmose.configuration.species",
        # --- Background-species keys (engine reads in background.py) ---
        # The AST walker scans only config.py. These six keys are read by
        # osmose/engine/background.py (line 178-189) when a config declares
        # background species via osmose.configuration.background +
        # simulation.nbackground (e.g. baltic seal sp14, cormorant sp15).
        # Closes C5 from docs/plans/2026-05-05-deep-review-remediation-plan.md.
        "species.nclass.sp{idx}",
        "species.trophic.level.sp{idx}",
        "species.length.sp{idx}",
        "species.size.proportion.sp{idx}",
        "species.age.sp{idx}",
        # --- Legacy species.lw.* aliases ---
        # Shipped configs use species.lw.* as aliases for
        # species.length2weight.*; Java engine reads both forms.
        "species.lw.condition.factor.sp{idx}",
        "species.lw.allpower.sp{idx}",
        # --- Movement map keys: registered in osmose/schema/movement.py
        # as of C1 (2026-05-05). Most of them are auto-detected by the
        # extended AST walker scanning movement_maps.py. The exception
        # is movement.species.map{idx}, which the engine reads via
        # key.startswith("movement.species.map") (a string method, not
        # cfg.get) — invisible to the literal-key walker, so allowlist it.
        "movement.species.map{idx}",
        # --- Fisheries movement/rate keys (Java-side) ---
        "fisheries.movement.file.map{idx}",
        "fisheries.movement.fishery.map{idx}",
        "fisheries.rate.byperiod.fsh{idx}",
        # --- Output configuration keys (Java-side output layer) ---
        # These control the Java engine's output; the Python engine has its
        # own output system and does not parse these.
        "output.byage.enabled",
        "output.bysize.enabled",
        "output.meansize.enabled",
        "output.trophiclevel.enabled",
        "output.frequency.ndtperyear",
        "output.diet.stage.structure",
        "output.diet.stage.threshold.sp{idx}",
        "output.mortality.additionaln.byage.enabled",
        "output.mortality.additionaln.bysize.enabled",
        "output.restart.recordfrequency.ndt",
        "output.restart.spinup",
        # --- Population and simulation restart keys (Java-side) ---
        "population.initialization.method.sp{idx}",
        "simulation.restart.enabled",
        # --- Species biomass time-scale key (Java-side) ---
        "species.biomass.nsteps.year",
    ]
)


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
    escaped = re.escape(pattern).replace(r"\{idx\}", r"\d+")
    return re.compile(f"^{escaped}$")


def _read_config_source() -> str:
    """Read osmose/engine/config.py via importlib.resources (test-hookable)."""
    import importlib.resources

    return (
        importlib.resources.files("osmose.engine")
        .joinpath("config.py")
        .read_text(encoding="utf-8")
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
                if isinstance(node.left, ast.Constant) and isinstance(
                    node.left.value, str
                ):
                    if "." in node.left.value:
                        _capture_string(node.left.value)

        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.JoinedStr):
                rendered = _render_fstring(node.value)
                if (
                    rendered is not None
                    and "." in rendered
                    and " " not in rendered
                ):
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
    literals = frozenset(p for p in patterns if "{idx}" not in p)
    regex_pairs = tuple(
        (p, _compile_regex_for_pattern(p)) for p in patterns if "{idx}" in p
    )
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
            f"validation.strict.enabled must be one of {list(_VALID_MODES)!r}; "
            f"got {mode!r}"
        )

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
