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
    result = dict(cfg)
    if target_version == "4.4.0":
        result["osmose.version"] = "4.4.0"
        return result
    for new_prefix in sorted(_INVERSE_440, key=len, reverse=True):
        old_prefix = _INVERSE_440[new_prefix]
        for key in [k for k in result if k == new_prefix or k.startswith(new_prefix + ".")]:
            reversed_key = old_prefix + key[len(new_prefix) :]
            if reversed_key not in result:
                result[reversed_key] = result.pop(key)
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
