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
