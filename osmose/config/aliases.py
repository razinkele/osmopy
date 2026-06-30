"""Authoritative OSMOSE 4.4.0 config-key rename set.

Faithful port of the v4.4.0 release entry (``$15``) in the Java engine's
``Releases.java`` (``fr.ird.osmose.util.version.Releases``). That entry's
``updateParameters()`` calls ``updateKey(OLD, NEW)`` for every key that was
renamed between 4.3.x and 4.4.0.

Semantics ported here:

* ``updateKey(OLD, NEW)`` renames ``OLD`` -> ``NEW`` and **skips if NEW is
  already defined** ("already defined" => merge/keep-existing). The Python
  applier mirrors this skip-if-exists behaviour.
* Config keys are compared **lowercased** (the reader lowercases keys), so the
  map below is stored fully lowercase even though the Java source uses
  camelCase in a few keys (e.g. ``output.restart.recordFrequency.ndt``).
* A few renames in the Java source are **prefix** renames applied per focal
  species index (``...spN``). Those are stored here as the prefix (stopping
  *before* the ``.sp`` / sub-key segment); the ``migrate_config`` applier
  matches ``k == old or k.startswith(old + ".")``, so indexed ``...spN`` and
  ``species.maturity.<r|m0|m1|eta>.spN`` keys are caught via the ``.``
  separator.

The Java source contains duplicate and chained ``updateKey`` calls (e.g.
``fisheries.enabled`` is first promoted to ``process.multispecies.fisheries.enabled``
and then to ``module.multispecies.fisheries.enabled``). This map records the
**net** result for a fresh 4.3.x config, which is what a forward migration
needs. The transient intermediate key is therefore intentionally omitted.
"""

from __future__ import annotations

import re
import warnings

# Larval additional-mortality SCALAR per-species rate. The OSMOPY internal model and the Python
# engine apply this value as a once-per-cohort TOTAL (osmose/engine/processes/natural.py:139 —
# NOT divided by ndt). 4.4.0 Java reinterprets the config value as rate/YEAR and pre-divides by
# nStepYear at load. So multiply by ndtperyear on write (cancels the jar's ÷ndt); divide when
# READING a native-4.4.0 config. Matches ONLY `...rate.spN` (NOT `.bydt.file.spN` nor `.seasonality.file.spN`).
_LARVA_RATE_RE = re.compile(r"^mortality\.additional\.larva\.rate\.sp\d+$")
_NDT_KEY = "simulation.time.ndtperyear"


def _numeric_version(v: str) -> tuple[int, ...]:
    """Parse the numeric prefix of a version, tolerant of -SNAPSHOT/+build suffixes."""
    from osmose.demo import _version_tuple

    return _version_tuple(re.split(r"[-+]", v.strip())[0])


def _ndtperyear(cfg: dict[str, str]) -> float | None:
    raw = cfg.get(_NDT_KEY)
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _scale_rate_value(value: str, factor: float) -> str:
    # ';'-separated per-stage arrays scaled component-wise; :.10g trims float noise.
    # Unset/sentinel components ('', null/none/na/nan) pass through verbatim so a
    # recoverable config isn't turned into an uncaught ValueError.
    def _scale(part: str) -> str:
        s = part.strip()
        if not s or s.lower() in ("null", "none", "na", "nan"):
            return part
        return f"{float(s) * factor:.10g}"

    return ";".join(_scale(p) for p in value.split(";"))


def _migrate_larva_rate(cfg: dict[str, str], factor: float, *, warn_bydt: bool) -> dict[str, str]:
    """Scale per-year larval-rate scalars by `factor` (ndt to write 4.4.0, 1/ndt to read it)."""
    result = dict(cfg)
    has_larva = any(_LARVA_RATE_RE.match(k) for k in result)
    if has_larva and not _ndtperyear(result):  # None or 0 -> cannot migrate; skip + warn
        warnings.warn(
            "larval additional-mortality rate present but simulation.time.ndtperyear is missing/zero; "
            "skipping the 4.4.0 per-year unit migration (config may be mis-scaled for the jar).",
            stacklevel=2,
        )
        return result
    for key, value in list(result.items()):
        if _LARVA_RATE_RE.match(key):
            result[key] = _scale_rate_value(value, factor)
        elif warn_bydt and key.startswith("mortality.additional.larva.rate.bydt.file"):
            warnings.warn(
                f"{key} references a per-time-step larval-rate file that 4.4.0 reads as rate/year; "
                "OSMOPY does not rescale the referenced file — verify it manually.",
                stacklevel=2,
            )
    return result


def _drop_4_4_0_removed_keys(cfg: dict[str, str]) -> dict[str, str]:
    """Drop keys that 4.4.0 removed or reinterprets, to preserve legacy behavior on write."""
    result = dict(cfg)
    # 4.4.0 removed the species.lmax growth cap (no clean migration; documented limitation).
    for key in [k for k in result if k == "species.lmax" or k.startswith("species.lmax.")]:
        del result[key]
    # 4.4.0 species.beta feeds the predation allometric exponent (default 1 == legacy). 4.3.3 read
    # it only under bioen; for a non-bioen config it must not be emitted.
    bioen_on = str(result.get("module.bioenergetics.enabled", "false")).lower() == "true"
    if not bioen_on:
        for key in [k for k in result if k == "species.beta" or k.startswith("species.beta.")]:
            del result[key]
    return result


def _emit_resource_biomass_forcing(cfg: dict[str, str]) -> dict[str, str]:
    """Emit 4.4.x NETCDF_BIOMASS resource-forcing keys for file-forced resource species.

    4.4.x ``ResourceForcing.init()`` (Simulation.initResourceForcing) hard-requires, per
    NetCDF-forced resource species:
      - ``species.biomass.file.spN``    — the NetCDF path (``species.file.spN`` is only an
                                          ``isNull`` guard, NOT what the forcing reads);
      - ``species.biomass.varname.spN`` — the variable name inside the NetCDF, NO engine
                                          default (``getString``→error if absent);
      - ``species.biomass.nsteps.year.spN`` — per-species steps/year (global
                                          ``species.biomass.nsteps.year`` is an accepted fallback);
      - ``species.biomass.mode.spN``    — optional (defaults ``NETCDF_BIOMASS``); emitted for clarity.
    OSMOPY 4.3.x configs carry only ``species.{type,name,file}.spN``, so 4.4.x dies with
    "NETCDF_BIOMASS resource forcing is used but parameters are missing". This fills the keys
    additively. ``varname = species.name.spN`` (verified 1:1 with the EEC NetCDF data vars —
    NOT sniffed from the NetCDF, whose 10 vars are otherwise ambiguous). Cite:
    ``ResourceForcing.java::init()`` @ v4.4.1.
    """
    result = dict(cfg)
    global_nsteps = cfg.get("simulation.time.ndtperyear") or cfg.get("species.biomass.nsteps.year")
    for key, val in cfg.items():
        if not key.startswith("species.type.sp") or str(val).strip().lower() != "resource":
            continue
        idx = key[len("species.type.sp") :]
        if not idx.isdigit() or f"species.file.sp{idx}" not in cfg:
            continue  # only file-forced resources need the NETCDF_BIOMASS keys
        result.setdefault(f"species.biomass.mode.sp{idx}", "NETCDF_BIOMASS")
        result.setdefault(f"species.biomass.file.sp{idx}", cfg[f"species.file.sp{idx}"])
        name = cfg.get(f"species.name.sp{idx}")
        if name:
            result.setdefault(f"species.biomass.varname.sp{idx}", name)
        if global_nsteps:
            result.setdefault(f"species.biomass.nsteps.year.sp{idx}", str(global_nsteps))
    return result


def _emit_background_species_keys(cfg: dict[str, str]) -> dict[str, str]:
    """Emit the source-dir-free 4.4.x keys that BackgroundSpecies.<init> requires.

    For each ``species.type.spN == "background"`` species, emit ``species.multiplier.spN`` (a
    scalar — 4.4.x reads it via getFloat; no per-class form) and ``species.beta.spN`` (the
    predation allometric exponent; legacy default 1). Only fills absent keys (never overwrites).
    The inline ``species.biomass.spN`` array is NOT emitted here — it needs the predator NetCDF,
    materialized at staging time (see scripts/baltic_440_smoke.py); ``species.biomass.{mode,file,
    varname}`` are the *resource* contract and are NOT used by background species.
    """
    result = dict(cfg)
    for key, val in cfg.items():
        if not key.startswith("species.type.sp") or str(val).strip().lower() != "background":
            continue
        idx = key[len("species.type.sp") :]
        result.setdefault(f"species.multiplier.sp{idx}", "1")
        result.setdefault(f"species.beta.sp{idx}", "1")
    return result


# old_prefix -> new_prefix. Ported VERBATIM from Releases.java $15 (v4.4.0), verified Step 1.
# The migrate_config applier matches `k == old or k.startswith(old + ".")`, so indexed
# `...spN` keys are caught via the `.` separator (prefixes deliberately stop before `.sp`).
RENAMES_440: dict[str, str] = {
    # --- module enable flags ---
    "fisheries.enabled": "module.multispecies.fisheries.enabled",
    "simulation.bioen.enabled": "module.bioenergetics.enabled",
    "simulation.genetic.enabled": "module.genetics.enabled",
    "economy.enabled": "module.bioeconomics.enabled",
    "population.initialization.relativebiomass.enabled": "module.population.initialisation.enabled",
    # --- restart parameters ---
    "output.restart.enabled": "simulation.restart.enabled",
    "output.restart.spinup": "simulation.restart.spinup.nyear",
    "output.restart.recordfrequency.ndt": "simulation.restart.recordfrequency.ndt",
    # --- fishery output flags ---
    "output.fishery.enabled": "output.fisheries.enabled",
    "output.fishery.byage.enabled": "output.fisheries.byage.enabled",
    "output.fishery.bysize.enabled": "output.fisheries.bysize.enabled",
    "output.spatial.fishery.enabled": "output.spatial.fisheries.enabled",
    "output.fecundity.bysize.enabled": "output.number.of.eggs.bysize.enabled",
    # --- bioenergetics species/predation prefixes (applied per focal spN) ---
    "species.bioen.maturity": "species.maturity",  # catches .r/.m0/.m1/.eta.spN
    "predation.ingestion.rate.max.bioen": "predation.ingestion.rate.max",  # .spN; merge/skip-if-exists
    "predation.coef.ingestion.rate.max.larvae.bioen": "predation.larval.ingestion.rate.increase.ratio",  # .spN
}


# new_prefix -> old_prefix (inverse of RENAMES_440).
#
# Omissions (intentional):
#   "predation.ingestion.rate.max.bioen" -> "predation.ingestion.rate.max"  is LOSSY
#       (the 4.4.0 unified key equals the pre-existing legacy key, so no inverse key exists).
#   "species.bioen.maturity" -> "species.maturity"  fans out to 4 explicit leaf entries
#       below to avoid corrupting the pre-existing species.maturity.size / .age growth keys.
_INVERSE_440: dict[str, str] = {
    # module enable flags
    "module.multispecies.fisheries.enabled": "fisheries.enabled",
    "module.bioenergetics.enabled": "simulation.bioen.enabled",
    "module.genetics.enabled": "simulation.genetic.enabled",
    "module.bioeconomics.enabled": "economy.enabled",
    "module.population.initialisation.enabled": "population.initialization.relativebiomass.enabled",
    # restart parameters
    "simulation.restart.enabled": "output.restart.enabled",
    "simulation.restart.spinup.nyear": "output.restart.spinup",
    "simulation.restart.recordfrequency.ndt": "output.restart.recordfrequency.ndt",
    # fishery output flags
    "output.fisheries.enabled": "output.fishery.enabled",
    "output.fisheries.byage.enabled": "output.fishery.byage.enabled",
    "output.fisheries.bysize.enabled": "output.fishery.bysize.enabled",
    "output.spatial.fisheries.enabled": "output.spatial.fishery.enabled",
    "output.number.of.eggs.bysize.enabled": "output.fecundity.bysize.enabled",
    # bioenergetics maturity — leaf-scoped (4 leaves) so .size/.age growth keys are never touched
    "species.maturity.eta": "species.bioen.maturity.eta",
    "species.maturity.r": "species.bioen.maturity.r",
    "species.maturity.m0": "species.bioen.maturity.m0",
    "species.maturity.m1": "species.bioen.maturity.m1",
    # larvae ingestion ratio
    "predation.larval.ingestion.rate.increase.ratio": "predation.coef.ingestion.rate.max.larvae.bioen",
}


def to_target_keys(cfg: dict[str, str], target_version: str = "4.3.3") -> dict[str, str]:
    """Emit config keys for a target engine version (inverse of canonicalize).

    target 4.4.0 -> identity + version stamp. target 4.3.x -> reverse the 4.4.0 renames
    (longest new-prefix first so leaf keys win over shorter prefixes), set osmose.version.
    Reverse is per-key/prefix and leaf-scoped; keys not in _INVERSE_440 (incl. the pre-existing
    species.maturity.size/age growth keys) pass through. The ingestion merge is non-invertible:
    the unified value reverses to the legacy predation.ingestion.rate.max.sp{idx} (NOT .bioen).
    """
    from osmose.demo import _version_tuple

    result = dict(cfg)
    # Suffix-tolerant: any 4.4.x-or-higher target (incl. "4.4.0-SNAPSHOT", "4.4.1") takes the
    # native (identity) branch. A bare _version_tuple returns (0,) for suffixed strings, so the
    # suffix MUST be stripped (via _numeric_version) before the compare or it would fall through
    # to the reverse branch and corrupt a native config back to 4.3.x key names.
    # Source-AND-target-aware larval-rate transform: the rate KEY name is version-stable (not in
    # RENAMES_440); only its unit changes (per-cohort <-> rate/year). Net factor =
    # ndt^[target>=4.4] / ndt^[source>=4.4], so: 4.3.3->4.4 x ndt; 4.4->4.4 = 1 (no double-scale
    # when a native config is staged for the jar); 4.4->4.3.3 = / ndt (restore for the 4.3.3 jar);
    # 4.3.3->4.3.3 = 1. Read the source version from the input config's existing stamp.
    src_ge = _numeric_version(cfg.get("osmose.version", "4.3.3")) >= _version_tuple("4.4.0")
    tgt_ge = _numeric_version(target_version) >= _version_tuple("4.4.0")
    ndt = _ndtperyear(result) or 1.0  # falsy ndt -> 1.0; _migrate_larva_rate still warns if missing
    rate_factor = (ndt if tgt_ge else 1.0) / (ndt if src_ge else 1.0)

    if tgt_ge:
        result = _drop_4_4_0_removed_keys(result)
        # Called unconditionally so the missing-ndt warning still fires (factor 1.0 is a no-op).
        result = _migrate_larva_rate(result, rate_factor, warn_bydt=True)
        result = _emit_resource_biomass_forcing(result)  # 4.4.x NETCDF_BIOMASS required keys
        result = _emit_background_species_keys(result)  # 4.4.x background-species keys
        result["osmose.version"] = target_version  # stamp the ACTUAL target, not a hardcoded 4.4.0
        return result
    # reverse branch (target < 4.4.0): un-scale a native source before reversing the renames.
    if rate_factor != 1.0:  # native 4.4 -> 4.3.3 jar: / ndt restores the per-cohort value
        result = _migrate_larva_rate(result, rate_factor, warn_bydt=False)
    for new_prefix in sorted(_INVERSE_440, key=len, reverse=True):
        old_prefix = _INVERSE_440[new_prefix]
        for key in [k for k in result if k == new_prefix or k.startswith(new_prefix + ".")]:
            reversed_key = old_prefix + key[len(new_prefix) :]
            value = result.pop(key)  # always drop the NEW-named key
            if reversed_key not in result:  # keep an existing OLD value (base wins)
                result[reversed_key] = value
    result["osmose.version"] = target_version
    return result


def canonicalize_config(cfg: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """Migrate a config dict to canonical 4.4.0 keys; return (new_cfg, deprecated_old_keys).

    ``deprecated_old_keys`` = the OLD keys from RENAMES_440 present in the input (for
    one-time deprecation logging by callers). Idempotent on already-4.4.0 configs (NEW
    keys are never in RENAMES_440's OLD set, so they pass through untouched).
    """
    from osmose.demo import migrate_config

    deprecated = sorted(
        k for k in cfg if any(k == old or k.startswith(old + ".") for old in RENAMES_440)
    )
    return migrate_config(cfg, target_version="4.4.0"), deprecated
