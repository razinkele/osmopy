"""Can the C# engine run this config? Fail closed (spec decision D11, plan task 1.1).

Three layers, all pure Python (no CLR):

1. **The catalogue** (``features.json``, shipped with this package and embedded in the C#
   assembly): every optional feature, its phase or status, and whether the C# engine implements
   it. ``core`` is the Phase 1 engine itself and is always required.
2. **Detectors**, one per catalogue feature, decide from the parsed ``EngineConfig`` (and the raw
   dict for the few switches the engine reads outside it) whether a config uses the feature.
   Where the Python engine has a predicate for the switch, the detector calls it rather than
   re-parsing the key, so the two cannot disagree about a spelling.
3. **Key coverage.** Detectors only catch features someone listed. So every config key the Python
   engine reads must also match a rule in ``KEY_RULES``: Phase 1 core, inert, or owned by a named
   feature (whose detector then decides). A set key that the Python engine reads but no rule
   classifies **blocks the run**, naming the key. Keys the Python engine never reads (unknown,
   or Java-only per ``config_validation``) follow Python: they have no effect on either engine.

Spec Q4, as decided: catch-based fishing, by-class rates and selectivity types other than -1, 0
and 1 are refused. They are parsed but not applied by the Python engine.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

CATALOG_PATH = Path(__file__).resolve().parent / "features.json"


@dataclass(frozen=True)
class Feature:
    id: int
    name: str
    phase: int
    status: str  # planned | refused | not_planned
    implemented: bool
    description: str


@cache
def load_catalog() -> dict[str, Feature]:
    """The catalogue, by name, in id order."""
    raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    feats = sorted(
        (Feature(**{k: f[k] for k in Feature.__dataclass_fields__}) for f in raw["features"]),
        key=lambda f: f.id,
    )
    return {f.name: f for f in feats}


def catalog_contract_versions() -> list[int]:
    return list(json.loads(CATALOG_PATH.read_text(encoding="utf-8"))["contract_versions"])


# --------------------------------------------------------------------------- detectors


def _any_set(items: Any) -> bool:
    return items is not None and any(x is not None for x in items)


def _any_true(arr: Any) -> bool:
    return arr is not None and bool(np.any(np.asarray(arr)))


def _focal(c: Any, values: Any) -> np.ndarray:
    """The focal species' entries of a per-species array. Several arrays are padded with
    background-species entries (lmax 0.0, movement "none"), which must not trip a detector:
    background behaviour belongs to the background_species feature."""
    return np.asarray(list(values) if isinstance(values, list) else values)[: c.n_species]


def _selectivity(c: Any) -> tuple[np.ndarray, np.ndarray]:
    # Python applies selectivity in a focal-only loop (_precompute_effective_rates).
    return _focal(c, c.fishing_selectivity_type), _focal(c, c.fishing_selectivity_l50)


def _detect_raw(fn_path: str):
    module, name = fn_path.rsplit(".", 1)

    def detector(_c: Any, raw: Mapping[str, str]) -> bool:
        import importlib

        return bool(getattr(importlib.import_module(module), name)(raw))

    return detector


DETECTORS: dict[str, Any] = {
    "core": lambda c, raw: True,
    "full_outputs": lambda c, raw: not c.output_minimal,
    "background_species": lambda c, raw: c.n_background > 0,
    "lmax_cap": lambda c, raw: bool(np.isfinite(_focal(c, c.lmax)).any()),
    "stock_recruitment": lambda c, raw: any(str(t) != "none" for t in c.recruitment_type),
    "rv_gate": lambda c, raw: _any_true(c.rv_gate_enabled),
    "salinity_gate": lambda c, raw: bool(c.salinity_gate_enabled),
    "oxygen_benthos": _detect_raw("osmose.engine.resources.oxygen_benthos_enabled"),
    "movement_none": lambda c, raw: "none" in _focal(c, c.movement_method).tolist(),
    "depletable_ltl": _detect_raw("osmose.engine.resources.depletable_enabled"),
    # In use, not merely switched on: the switch without a flux file does nothing (the Benguela
    # demo ships so), and a config must not be refused for a feature it does not use.
    "incoming_flux": _detect_raw("osmose.engine.incoming_flux.incoming_flux_in_use"),
    "mpa": lambda c, raw: bool(c.mpa_zones),
    "fishing_rate_by_year": lambda c, raw: _any_set(c.fishing_rate_by_year),
    "additional_rate_by_step": lambda c, raw: _any_set(c.additional_mortality_by_dt),
    "larva_rate_by_step": lambda c, raw: _any_set(c.larva_mortality_by_dt),
    "additional_maps": lambda c, raw: _any_set(c.additional_mortality_spatial),
    "selectivity_logistic": lambda c, raw: (
        c.fishing_enabled and bool(np.any(_selectivity(c)[0] == 1))
    ),
    "selectivity_length": lambda c, raw: (
        c.fishing_enabled and bool(np.any((_selectivity(c)[0] == -1) & (_selectivity(c)[1] > 0)))
    ),
    "recruitment_ceiling": lambda c, raw: _any_true(c.recruitment_ceiling_enabled),
    "thermal_gate": lambda c, raw: _any_true(c.thermal_gate_enabled),
    "depensation_gate": lambda c, raw: _any_true(c.depensation_gate_enabled),
    "rv_spatial": lambda c, raw: _any_true(c.rv_spatial_enabled),
    "functional_response_2_3": lambda c, raw: bool(np.any(np.asarray(c.fr_shape) != 1)),
    "species_accessibility": lambda c, raw: (
        c.stage_accessibility is None and c.accessibility_matrix is not None
    ),
    "dynamic_accessibility": lambda c, raw: bool(c.dynamic_accessibility_enabled),
    "gompertz_growth": lambda c, raw: any(g != "VB" for g in c.growth_class),
    "feeding_stage_metric": lambda c, raw: any(
        m != "size" and int(n) > 1
        for m, n in zip(c.feeding_stage_metric, np.asarray(c.n_feeding_stages), strict=False)
    ),
    "seeding_linear": lambda c, raw: c.seeding_mode == "linear",
    "warm_start": _detect_raw("osmose.engine.initialization._flag_enabled"),
    "spatial_outputs": lambda c, raw: bool(c.output_spatial_enabled),
    "carbon_output": lambda c, raw: bool(c.output_carbon or c.output_carbon_netcdf),
    "bioenergetics": lambda c, raw: bool(c.bioen_enabled),
    "genetics": lambda c, raw: bool(c.genetics_enabled),
    "economics": lambda c, raw: bool(c.economics_enabled),
    "java_compat_rng": lambda c, raw: bool(c.java_compat_rng),
    "fishing_catches": lambda c, raw: (
        (c.fishing_catches is not None and bool(np.any(np.asarray(c.fishing_catches) > 0)))
        or _any_set(c.fishing_catches_by_year)
    ),
    "rate_by_class": lambda c, raw: (
        _any_set(c.fishing_rate_by_dt_by_class) or _any_set(c.additional_mortality_by_dt_by_class)
    ),
    "selectivity_type_2_3": lambda c, raw: (
        c.fishing_enabled and bool(np.any(~np.isin(_selectivity(c)[0], [-1, 0, 1])))
    ),
}


def required_features(config: Any, raw: Mapping[str, str]) -> list[str]:
    """Catalogue features this config needs, in id order (``core`` always first)."""
    return [name for name in load_catalog() if DETECTORS[name](config, raw)]


# --------------------------------------------------------------------------- key coverage

#: Ordered (regex on the pattern-form key, owner) rules. Owner is "phase1" (core semantics the
#: C# engine carries), "inert" (read but no effect on results), "internal" (not a config key:
#: AST noise from config.py's own dict literals), or comma-separated catalogue feature names
#: whose detectors decide whether the key is active. First match wins.
KEY_RULES: tuple[tuple[str, str], ...] = (
    (r"[A-Za-z0-9_]+", "internal"),
    (
        r"_osmose\.config\.dir|osmose\.(configuration\..+|version)|validation\.strict\.enabled",
        "inert",
    ),
    # No effect on the Python engine's results (per-species generators never consumed, 2026-11-17).
    (r"(movement|stochastic\.mortality)\.randomseed\.fixed", "inert"),
    (r"engine\.rng\.java_compat", "java_compat_rng"),
    (r"module\.genetics\.enabled|evolution\..+", "genetics"),
    (r"module\.bioeconomics\.enabled|simulation\.economic\..+", "economics"),
    (
        (
            r"module\.bioenergetics\.enabled|simulation\.bioen\..+|species\.bioen\..+"
            r"|predation\.c\.bioen\..+|species\.beta\..+|species\.zlayer\..+|species\.oxygen\.c[12]\..+"
            r"|species\.maturity\.(eta|m0|m1|r)\..+|species\.larvae\.growth\.threshold\.age\..+"
            r"|predation\.larval\.ingestion\.rate\.increase\.ratio\..+|temperature\..+|output\.bioen\..+"
        ),
        "bioenergetics",
    ),
    # The oxygen forcing is loaded unconditionally but only these two features use it.
    (r"oxygen\..+", "oxygen_benthos,bioenergetics"),
    (r"ltl\.oxygen\.benthos\..+", "oxygen_benthos"),
    (r"module\.population\.initialisation\.enabled", "warm_start"),
    (r"simulation\.incoming\.flux\.enabled|flux\.incoming\..+", "incoming_flux"),
    (r"ltl\.depletable\..+|ltl\.regrowth\..+|species\.regrowth\.rate\..+", "depletable_ltl"),
    (r"movement\.salinity\..+", "salinity_gate"),
    (r"reproduction\.rv\.gate\..+", "rv_gate"),
    (r"reproduction\.rv\.spatial\..+", "rv_spatial"),
    (r"reproduction\.thermal\..+", "thermal_gate"),
    (r"reproduction\.depensation\..+", "depensation_gate"),
    (r"reproduction\.recruitment\.ceiling\..+", "recruitment_ceiling"),
    (r"stock\.recruitment\..+", "stock_recruitment"),
    (r"predation\.accessibility\.dynamic\..+", "dynamic_accessibility"),
    (r"growth\.(gompertz|exponential|java\.classname)\..+", "gompertz_growth"),
    (r"species\.lmax\..+", "lmax_cap"),
    (r"mpa\..+", "mpa"),
    (r"mortality\.fishing\.catches\..+", "fishing_catches"),
    (r"mortality\.fishing\.rate\.byyear\.file\..+", "fishing_rate_by_year"),
    (r"mortality\.fishing\.rate\.(bydt\.by(age|size)|\{idx\})\.file\..+", "rate_by_class"),
    (r"mortality\.additional\.rate\.\{idx\}\.file\..+", "additional_rate_by_step,rate_by_class"),
    (r"mortality\.additional\.larva\.rate\.\{idx\}\.file\..+", "larva_rate_by_step"),
    (r"mortality\.additional\.spatial\.distrib\.file\..+", "additional_maps"),
    (r"output\.spatial\..+", "spatial_outputs"),
    (r"output\.carbon\..+|carbon\..+", "carbon_output"),
    (r"species\.(nclass|length|age|size\.proportion)\..+", "background_species"),
    # ---- Phase 1 core: carried by the bundle builder (task 1.2) ----
    (
        (
            r"simulation\.(time\..+|nsteps\.year|nspecies|nschool(\..+)?|nresource|nbackground"
            r"|nfisheries|fishing\.mortality\.enabled)|mortality\.subdt|grid\..+"
        ),
        "phase1",
    ),
    (
        (
            r"species\.(name|type|file|linf|k|t0|egg\.(size|weight)|length2weight\..+|lw\..+|lifespan"
            r"|maturity\.(size|age)|relativefecundity|sexratio|vonbertalanffy\.threshold\.age"
            r"|delta\.lmax\.factor|size\.(min|max)|tl|trophic\.level|accessibility2fish(\.file)?"
            r"|biomass\..+|multiplier|offset)(\..+)?"
        ),
        "phase1",
    ),
    (r"ltl\..+", "phase1"),
    (
        (
            r"predation\.(accessibility\.file|predprey\..+|ingestion\.rate\.max\..+"
            r"|efficiency\.critical\..+|functional\.response\..+)"
        ),
        "phase1",
    ),
    (
        (
            r"mortality\.(additional\.rate|additional\.larva\.rate|out\.rate|starvation\.rate\.max"
            r"|fishing\.rate|fishing\.spatial\.distrib\.file)\.sp\{idx\}"
        ),
        "phase1",
    ),
    (r"fishing\.(rate|selectivity\.l50|selectivity\.l75)\.sp\{idx\}", "phase1"),
    (r"fisheries\..+|module\.multispecies\.fisheries\.enabled", "phase1"),
    (
        (
            r"movement\.(distribution\.(method|ncell)|file|initialage|lastage|initialyear|lastyear"
            r"|steps|years|species|randomwalk\.range|map\.strict\.coverage)(\..+)?"
        ),
        "phase1",
    ),
    (r"population\.seeding\.(biomass\..+|year\.max|mode)", "phase1"),
    (r"reproduction\.(season\.file\..+|normalisation\.enabled)", "phase1"),
    (r"output\.(minimal\.enabled|cutoff\.age\..+|recordfrequency\.ndt|step0\.include)", "phase1"),
    # The output keys the Python engine reads: each is a family the C# engine collects since
    # task 2.6 (the life-stage split needs no key), the binning of one, or a NetCDF switch of
    # the shared writers. There is deliberately NO catch-all for output.*: with full_outputs
    # implemented, a catch-all would accept a new Python-side output and the C# engine would
    # silently not produce it. A new output key must be classified here by hand, as a family
    # the C# engine collects or as a feature of its own; until then the config is refused and
    # test_every_key_the_python_engine_reads_is_classified fails.
    (
        (
            r"output\.(biomass|abundance)\.(byage|bysize)\.(netcdf\.)?enabled"
            r"|output\.(biomass|abundance|mortality|yield\.biomass)\.netcdf\.enabled"
            r"|output\.(size|ssb|yield\.abundance)\.(netcdf\.)?enabled"
            r"|output\.(tl|meantl|diet\.composition)\.enabled"
            r"|output\.distrib\.bysize\.(min|max|incr)"
        ),
        "full_outputs",
    ),
)

_COMPILED = tuple((re.compile(rx), owner) for rx, owner in KEY_RULES)


def classify_pattern(pattern: str) -> str | None:
    """The owner of a pattern-form key (``species.linf.sp{idx}``), or None if unclassified."""
    for rx, owner in _COMPILED:
        if rx.fullmatch(pattern):
            return owner
    return None


def unclassified_keys(raw: Mapping[str, str]) -> list[str]:
    """Keys set in ``raw`` that the Python engine reads but no rule classifies (fail closed)."""
    from osmose.engine.config_validation import (
        _ALLOWLIST_JAVA_ONLY,
        _check,
        _normalize_key_to_pattern,
        build_known_keys,
    )

    known = build_known_keys()
    out = []
    for key in raw:
        if _check(key, known) is not None:  # unknown to the Python engine: no effect there either
            continue
        pattern = _normalize_key_to_pattern(key)
        # A real OSMOSE key the Python engine never reads. The key itself is looked up too: a
        # Java-only key can have a segment that LOOKS indexed (output.abundance.age1.enabled
        # normalises to ...age{idx}...), and its pattern form is then not in the list.
        if pattern in _ALLOWLIST_JAVA_ONLY or key in _ALLOWLIST_JAVA_ONLY:
            continue
        if classify_pattern(pattern) is None:
            out.append(key)
    return sorted(out)


# --------------------------------------------------------------------------- the decision


def csharp_engine_block_reason(raw: Mapping[str, str]) -> str | None:
    """Why the C# engine cannot run this (raw, reader-produced) config, or None if it can.

    Lists every obstacle at once: unclassified keys, refused or not-planned features, and
    features the engine does not implement yet (naming their phase).
    """
    from osmose.engine.config import EngineConfig

    try:
        config = EngineConfig.from_dict(dict(raw))
    except Exception as exc:  # the Python engine cannot run it either
        return f"the config does not parse: {exc}"
    catalog = load_catalog()
    problems: list[str] = []
    unclassified = unclassified_keys(raw)
    if unclassified:
        problems.append(
            "config keys the C# engine has not classified (it refuses rather than ignore them): "
            + ", ".join(unclassified)
        )
    missing = [catalog[n] for n in required_features(config, raw) if not catalog[n].implemented]
    for status, label in (
        ("refused", "refused (spec Q4)"),
        ("not_planned", "not planned for the C# engine; use the Python engine"),
    ):
        names = [f.name for f in missing if f.status == status]
        if names:
            problems.append(f"{label}: {', '.join(names)}")
    planned = [f for f in missing if f.status == "planned"]
    if planned:
        problems.append(
            "not implemented yet: " + ", ".join(f"{f.name} (phase {f.phase})" for f in planned)
        )
    return "; ".join(problems) if problems else None
