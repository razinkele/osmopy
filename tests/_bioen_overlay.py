"""Shared bioenergetics config overlay for tests that need a *material* bioen budget.

``BIOEN_OVERLAY`` + ``apply_overlay`` turn a non-bioen focal config (e.g. the
``baltic``/``baltic_ev`` demo configs) into a minimal bioenergetics-enabled one, using
CANONICAL config key spellings (``species.maturity.{eta,r,m0,m1}.sp{i}``,
``predation.ingestion.rate.max.sp{i}``, ``species.bioen.mobilized.{tp,e.mobi,e.d}.sp{i}``,
``species.bioen.maint.{e.maint,energy.c_m}.sp{i}`` -- see ``EngineConfig.from_dict``,
``osmose/engine/config.py:2489-2530``) so ``from_dict`` parses it without any legacy-key
migration.

Content is copied from the (now-superseded) parent plan's Task 4b overlay
(``docs/superpowers/plans/2026-08-30-baltic-c3-bioen-stage1.md``), with one deliberate
change: ``c_m`` (``species.bioen.maint.energy.c_m.sp{i}``). ``data/baltic_ev``'s production
value (~0.001) combined with this overlay's other parameters was measured to make
maintenance ~1e-8 of gross intake -- ``E_net == E_gross`` at every abundance, which makes
bioen starvation (behaviour 3 of the parent bioen-Numba-kernel plan) impossible to exercise
(``.superpowers/sdd/2026-08-30-baltic-c3-bioen-stage1/task-6-carried-items.md``, item A).

``C_M = 1.0e12`` is chosen instead so maintenance is a MATERIAL fraction of intake. It is
pinned against ``tests/test_engine_bioen_budget_parity.py::_three_schools`` (school 0/1:
1 kg adults; school 2: 10 g juvenile) via ``energy_terms``, at the SAME two temperatures
that matter here:

  * T = 10.0 degC (the ``_three_schools`` tests' own convention, and the provenance number
    quoted in the carried-items doc): e_maint/e_gross = 0.81277.../0.81277.../10.20798...
    for schools 0/1/2. This is the exact value this module is pinned against -- see
    ``tests/test_bioen_overlay.py``.
  * T = 7.0 degC (``BIOEN_OVERLAY["temperature.value"]``, this overlay's own operating
    temperature): e_maint/e_gross = 0.61111.../0.61111.../7.67516... -- lower (Arrhenius:
    maintenance falls with temperature) but still comfortably material, and school 2's
    ratio stays deep in starvation territory either way. Do NOT raise ``C_M`` to chase 0.81
    at 7 degC -- 0.61 already clears the "material fraction" bar this overlay exists to
    guarantee, and the two temperatures are expected to disagree.

Background species handling (ledger ruling R1, ``osmose/engine/processes/bioen_predation.py``
module docstring): ``BioenPredationMortality.getMaxPredationRate`` early-returns for
background predators WITHOUT the ``/nStepYear`` division the focal branch applies, so
``predation.ingestion.rate.max.sp{i}`` for background indices must already be a PER-TIME-STEP
rate, not the per-year "annual turnover" convention the focal branch (and the base configs'
own background rows) use. ``apply_overlay`` converts whatever annual value is already in
``cfg`` for each background index by dividing by ``simulation.time.ndtperyear``. Separately,
``BackgroundSpecies.java:131-133`` reads ``species.beta.sp{i}`` for background species with NO
default (a missing key is a fatal Java config error); the port has no config-driven background
beta today -- ``per_fish_ingestion_cap`` hard-codes 0.8 for ``species_id >= n_species``
regardless of ``species.beta`` -- so setting it here is currently inert on the Python engine,
but is the Java-parity-faithful and forward-compatible thing to author.

FORAGING is deliberately left INERT by this overlay: ``species.bioen.forage.k_for.sp{i}``
is not set here, so ``config.bioen_k_for`` takes the engine default of 0.0
(``osmose/engine/config.py``'s ``_species_float_optional(..., "species.bioen.forage.k_for.sp{i}",
n_sp, 0.0)``) for every focal species, and ``_apply_foraging_for_school`` kills nobody. Neither
``data/baltic`` nor ``data/baltic_ev`` carries a ``k_for`` row, so there is no "real" value to
copy in from a production config the way ``c_m`` was cross-checked. A test that needs FORAGING
to actually fire (bioen-Numba-kernel plan Task 2 Step 2: "``k_for > 0`` so FORAGING is not
inert, with a positive witness ``n_dead[:, int(MortalityCause.FORAGING)].sum() > 0``") must set
``species.bioen.forage.k_for.sp{i}`` itself on top of this overlay -- do not assume BIOEN_OVERLAY
covers it.
"""

from __future__ import annotations

from collections.abc import Sequence

# e_maint/e_gross ~ 0.8 at 10 degC (task-6-carried-items.md item A; matches
# tests/test_engine_bioen_budget_parity.py::_three_schools exactly -- see module docstring
# and tests/test_bioen_overlay.py for the pinned arithmetic).
C_M = 1.0e12

BIOEN_OVERLAY: dict[str, str] = {
    "module.bioenergetics.enabled": "true",
    "simulation.bioen.phit.enabled": "true",
    "simulation.bioen.fo2.enabled": "false",
    "temperature.value": "7.0",
}

# Per-focal-species bioen keys and their overlay values (excluding m0, which is copied
# from the config's own species.maturity.size.sp{i} rather than a fixed literal, and
# c_m, which uses the C_M constant above).
_FOCAL_SPECIES_OVERLAY: dict[str, str] = {
    "species.maturity.m1.sp{i}": "0",
    "species.maturity.r.sp{i}": "0.2",
    "species.maturity.eta.sp{i}": "1",
    "species.beta.sp{i}": "0.8",
    "species.bioen.assimilation.sp{i}": "0.7",
    "species.bioen.mobilized.tp.sp{i}": "10",
    "species.bioen.mobilized.e.mobi.sp{i}": "0.65",
    "species.bioen.mobilized.e.d.sp{i}": "1.5",
    "species.bioen.maint.e.maint.sp{i}": "0.65",
}


def apply_overlay(
    cfg: dict[str, str], n_species: int, background_indices: Sequence[int]
) -> dict[str, str]:
    """Mutate ``cfg`` in place with the bioen overlay, and return it.

    Args:
        cfg: Config dict (as produced by ``OsmoseConfigReader().read(...)``), already
            containing a valid non-bioen focal config -- in particular
            ``species.maturity.size.sp{i}`` for every focal index and
            ``predation.ingestion.rate.max.sp{i}`` for every background index.
        n_species: Number of FOCAL species (indices ``0 .. n_species - 1``). Background
            species are NOT in this range -- see ``background_indices``.
        background_indices: Species indices (``>= n_species``) that are background
            predators (e.g. Baltic: ``[15, 16]``; ``baltic_ev``: ``[14, 15]``).

    Raises:
        ValueError: if ``n_species`` disagrees with ``cfg["simulation.nspecies"]``, or a
            ``background_indices`` entry is not actually a background species in ``cfg``
            (index < n_species, or not typed ``background`` -- e.g. missing, or a copy-paste
            from a different config's layout). Either failure mode would otherwise silently
            leave a real species at the bioen engine's ``c_m`` default of 0.0 (that species
            becomes starvation-free) or write bioen overlay keys onto the wrong index --
            "the two paths agree with each other while both being wrong" (see this module's
            docstring, ledger ruling R1).
        KeyError: if a focal index is missing ``species.maturity.size.sp{i}``, or a
            background index is missing ``predation.ingestion.rate.max.sp{i}`` -- both
            would otherwise silently produce a wrong or zeroed bioen budget.
    """
    cfg_n_species = int(cfg.get("simulation.nspecies", -1))
    if cfg_n_species != n_species:
        raise ValueError(
            f"n_species={n_species} does not match cfg['simulation.nspecies']="
            f"{cfg_n_species}; a mismatch silently zeros bioen params (c_m defaults to 0.0"
            " engine-side) for whichever focal species falls outside the overlaid range."
        )
    for i in background_indices:
        if i < n_species:
            raise ValueError(
                f"background index sp{i} is < n_species={n_species} -- it is a FOCAL"
                " species; background_indices must only name species.type=background rows."
            )
        species_type = cfg.get(f"species.type.sp{i}")
        if species_type != "background":
            raise ValueError(
                f"sp{i} is not typed background in cfg (species.type.sp{i}="
                f"{species_type!r}); background_indices must match this config's own"
                " background species, not another config's index layout."
            )

    cfg.update(BIOEN_OVERLAY)

    for i in range(n_species):
        sp = f"sp{i}"
        cfg[f"species.maturity.m0.{sp}"] = cfg[f"species.maturity.size.{sp}"]
        for pattern, value in _FOCAL_SPECIES_OVERLAY.items():
            cfg[pattern.format(i=i)] = value
        cfg[f"species.bioen.maint.energy.c_m.{sp}"] = str(C_M)

    n_dt_per_year = float(cfg.get("simulation.time.ndtperyear", "24"))
    for i in background_indices:
        sp = f"sp{i}"
        cfg[f"species.beta.{sp}"] = "0.8"
        key = f"predation.ingestion.rate.max.{sp}"
        annual_rate = float(cfg[key])  # KeyError if missing -- see docstring
        cfg[key] = repr(annual_rate / n_dt_per_year)

    return cfg
