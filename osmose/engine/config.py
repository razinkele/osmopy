"""EngineConfig: typed parameter extraction from flat OSMOSE config dicts.

Converts the flat string key-value config (as read by OsmoseConfigReader)
into typed NumPy arrays indexed by species, ready for vectorized computation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from osmose.engine.accessibility import AccessibilityMatrix
from osmose.engine.background import (
    BackgroundSpeciesInfo,
    _parse_floats,
    parse_background_species,
)
from osmose.engine.path_resolution import resolve_data_path
from osmose.engine.physical_data import PhysicalData
from osmose.logging import setup_logging

if TYPE_CHECKING:
    from osmose.engine.physical_data import PhysicalData

_log = setup_logging("osmose.engine.config")

# Dedupe the parsed-but-unapplied-feature warnings: EngineConfig is rebuilt per
# candidate during calibration, so an un-throttled warning would flood the logs.
# Warn once per distinct message for the process lifetime.
_WARNED_UNSUPPORTED_MORTALITY: set[str] = set()


_GROWTH_MAP: dict[str, str] = {
    # Current canonical classnames
    "fr.ird.osmose.process.growth.VonBertalanffyGrowth": "VB",
    "fr.ird.osmose.process.growth.GompertzGrowth": "GOMPERTZ",
    # Legacy backward compat
    "fr.ird.osmose.growth.VonBertalanffy": "VB",
    "fr.ird.osmose.growth.Gompertz": "GOMPERTZ",
    "fr.ird.osmose.growth.Linear": "VB",  # Linear was never real, map to VB
}

_FR_HALFSAT_SENTINEL = 1.0  # inert: type1 never reads fr_halfsat
_FR_SHAPE_CODE = {"type1": 1, "type2": 2, "type3": 3}


def _get(cfg: dict[str, str], key: str) -> str:
    """Get a config value, raising KeyError with a clear message."""
    val = cfg.get(key)
    if val is None:
        raise KeyError(f"Required OSMOSE config key missing: {key!r}")
    return val


def _species_float(cfg: dict[str, str], pattern: str, n: int) -> NDArray[np.float64]:
    return np.array([float(_get(cfg, pattern.format(i=i))) for i in range(n)])


def _species_int(cfg: dict[str, str], pattern: str, n: int) -> NDArray[np.int32]:
    return np.array([int(_get(cfg, pattern.format(i=i))) for i in range(n)], dtype=np.int32)


def _species_str(cfg: dict[str, str], pattern: str, n: int) -> list[str]:
    return [_get(cfg, pattern.format(i=i)) for i in range(n)]


def _species_float_optional(
    cfg: dict[str, str], pattern: str, n: int, default: float
) -> NDArray[np.float64]:
    """Extract a per-species float array, using default if key is missing."""
    return np.array([float(cfg.get(pattern.format(i=i), str(default))) for i in range(n)])


def _species_int_optional(
    cfg: dict[str, str], pattern: str, n: int, default: int
) -> NDArray[np.int32]:
    """Extract a per-species int array, using default if key is missing."""
    return np.array(
        [int(cfg.get(pattern.format(i=i), str(default))) for i in range(n)], dtype=np.int32
    )


def _species_str_optional(
    cfg: dict[str, str], pattern: str, n: int, default: str, allowed: set[str] | None = None
) -> list[str]:
    """Extract a per-species string array, using default if key is missing.

    Both the default and user-provided values are stripped of whitespace and
    lowercased before any comparison, so callers should pass `allowed` entries
    in lowercase. Returns a list of normalized lowercase strings.

    If `allowed` is given, raise ValueError on any value not in the set.
    """
    out: list[str] = []
    for i in range(n):
        key = pattern.format(i=i)
        val = cfg.get(key, default).strip().lower()
        if allowed is not None and val not in allowed:
            raise ValueError(f"{key}={val!r} is not one of {sorted(allowed)}")
        out.append(val)
    return out


def _resolve_file(file_key: str, config_dir: str = "") -> Path | None:
    """Resolve a relative file path against multiple search directories.

    Returns None for BOTH "key empty" (feature not configured) AND "key set but
    file missing" (user typo or wrong path). Callers that can distinguish these
    two cases should check `file_key` is non-empty before calling this, and
    prefer :func:`_require_file` when a non-empty key implies the file must
    exist.

    Thin wrapper around :func:`resolve_data_path` kept for backward compatibility.
    """
    return resolve_data_path(file_key, config_dir=config_dir)


def _require_file(file_key: str, config_dir: str, context: str) -> Path:
    """Resolve a file path that the caller has determined must exist.

    Unlike :func:`_resolve_file`, this helper raises :class:`FileNotFoundError`
    when ``file_key`` is set but the file cannot be located. Use it in callers
    that have already checked ``if not file_key: continue/skip`` and therefore
    know the user requested this file — a missing file at that point is a
    config error, not a silent "feature not enabled."

    Deep review v3 C-3 through C-7 identified five loaders in this module that
    used ``_resolve_file`` in the pattern::

        file_key = cfg.get("some.file.sp{i}", "")
        if not file_key:
            continue
        path = _resolve_file(file_key, _cfg_dir(cfg))
        if path is None:
            continue  # silently disables the feature for species i

    Callers in that pattern should now use ``_require_file`` and drop the
    defensive ``if path is None`` check — a missing file becomes a loud,
    actionable error instead of a silent wrong-result bug.

    Parameters
    ----------
    file_key:
        The raw value from the config dict (relative or absolute path).
    config_dir:
        The resolved config directory for relative-path lookup.
    context:
        Human-readable context included in the error message, e.g. the
        original config key name — helps users find the typo in their config.
    """
    path = resolve_data_path(file_key, config_dir=config_dir)
    if path is None:
        raise FileNotFoundError(
            f"Configured file {file_key!r} (from {context}) could not be resolved. "
            f"Checked config dir {config_dir!r} and standard data search paths."
        )
    return path


def _enabled(cfg: dict[str, str], key: str) -> bool:
    """Check if a config key is set to 'true' (case-insensitive)."""
    return cfg.get(key, "false").lower() == "true"


def _load_spatial_csv(path: Path) -> np.ndarray:
    """Load a semicolon-separated spatial grid CSV.

    Flips rows (south-to-north to north-to-south).
    """
    df = pd.read_csv(path, sep=";", header=None)
    data = df.values.astype(np.float64)
    return np.flipud(data)


def _cfg_dir(cfg: dict[str, str]) -> str:
    """Extract config directory from the config dict."""
    return cfg.get("_osmose.config.dir", "")


def _accessibility_path_or_none(cfg: dict[str, str]) -> "Path | None":
    """Resolve predation.accessibility.file once, shared by both accessibility loaders.

    Returns None when the key is absent or empty. Raises :class:`FileNotFoundError`
    when the key is set but the file cannot be located — a set-but-missing path is a
    config error, not a silent "feature disabled."
    """
    file_key = cfg.get("predation.accessibility.file", "")
    if not file_key:
        return None
    return _require_file(file_key, _cfg_dir(cfg), "predation.accessibility.file")


def _load_accessibility(cfg: dict[str, str], n_species: int) -> NDArray[np.float64] | None:
    """Load predation accessibility matrix from CSV if available.

    Returns matrix with shape (n_total, n_total) where index [predator, prey] = coefficient.
    Used only when no stage structure is configured.
    """
    path = _accessibility_path_or_none(cfg)
    if path is None:
        return None
    df = pd.read_csv(path, sep=";", index_col=0)
    return df.values.astype(np.float64)


def _load_stage_accessibility(
    cfg: dict[str, str], all_species_names: list[str]
) -> AccessibilityMatrix | None:
    """Load stage-indexed accessibility matrix when age/size stages are used.

    Returns an AccessibilityMatrix instance, or None if no accessibility file exists.
    """
    path = _accessibility_path_or_none(cfg)
    if path is None:
        return None
    return AccessibilityMatrix.from_csv(path, all_species_names)


@dataclass
class MPAZone:
    """A Marine Protected Area definition."""

    grid: NDArray[np.float64]  # 2D spatial grid (1 = protected, 0 = not)
    start_year: int
    end_year: int
    percentage: float  # reduction factor (0-1)

    def __post_init__(self) -> None:
        if not (0.0 <= self.percentage <= 1.0):
            raise ValueError(f"MPAZone percentage must be in [0, 1], got {self.percentage}")
        if self.start_year > self.end_year:
            raise ValueError(f"MPAZone start_year ({self.start_year}) > end_year ({self.end_year})")
        if self.grid.ndim != 2:
            raise ValueError(f"MPAZone.grid must be 2D (shape (ny, nx)), got {self.grid.ndim}D")
        if not np.isin(self.grid, [0.0, 1.0]).all():
            unique = np.unique(self.grid)
            raise ValueError(
                f"MPAZone.grid values must be 0 or 1 (binary protected/unprotected), "
                f"got unique values {unique.tolist()}"
            )
        if self.start_year < 0:
            raise ValueError(f"MPAZone.start_year must be non-negative, got {self.start_year}")


def _parse_fisheries(
    cfg: dict[str, str], species_names: list[str], n_species: int
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int32],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Parse fisheries-based fishing config (OSMOSE v4).

    Returns
    -------
    fishing_rate: NDArray[np.float64]
        Annual fishing rate per species.
    fishing_selectivity_a50: NDArray[np.float64]
        Age at 50% selectivity (years) per species. NaN if not applicable.
    fishing_selectivity_type: NDArray[np.int32]
        0 = age-based (knife-edge), 1 = sigmoidal size, -1 = no fishing.
    fishing_selectivity_l50: NDArray[np.float64]
        Length at 50% selectivity for sigmoidal type. 0 if not applicable.
    fishing_selectivity_slope: NDArray[np.float64]
        Slope of sigmoid selectivity. 0 if not applicable.
    """
    fishing_rate = np.zeros(n_species, dtype=np.float64)
    fishing_a50 = np.full(n_species, np.nan, dtype=np.float64)
    fishing_sel_type = np.full(n_species, -1, dtype=np.int32)
    fishing_l50 = np.zeros(n_species, dtype=np.float64)
    fishing_slope = np.zeros(n_species, dtype=np.float64)

    n_fisheries = int(cfg.get("simulation.nfisheries", "0"))
    if n_fisheries == 0:
        return fishing_rate, fishing_a50, fishing_sel_type, fishing_l50, fishing_slope

    # Read catchability CSV to map species -> fishery.
    # With n_fisheries > 0, the catchability file is required — empty or missing
    # is a config error, not a silent "no fishing" signal. (Deep review v3 C-3)
    catch_file = cfg.get("fisheries.catchability.file", "")
    if not catch_file:
        raise ValueError(
            f"simulation.nfisheries={n_fisheries} but fisheries.catchability.file "
            "is not set. Either set the catchability file or set nfisheries=0."
        )
    catch_path = _require_file(catch_file, _cfg_dir(cfg), "fisheries.catchability.file")

    catch_df = pd.read_csv(catch_path, index_col=0)
    # Row labels = species names, column labels = fishery names
    # Build species_name -> fishery_index mapping
    species_to_fishery: dict[str, int] = {}
    for row_idx in range(len(catch_df)):
        row_name = str(catch_df.index[row_idx]).strip()
        for col_idx in range(len(catch_df.columns)):
            val = float(catch_df.iloc[row_idx, col_idx])
            if val > 0:
                species_to_fishery[row_name.lower()] = col_idx
                break

    # Map species names to their fishing parameters
    for sp_idx in range(n_species):
        sp_name = species_names[sp_idx].strip().lower()
        fsh_idx = species_to_fishery.get(sp_name)
        if fsh_idx is None:
            continue

        # Base rate
        rate_key = f"fisheries.rate.base.fsh{fsh_idx}"
        rate_val = cfg.get(rate_key, "0.0")
        fishing_rate[sp_idx] = float(rate_val)

        # Selectivity type: 0 = knife-edge by age, 1 = sigmoidal size
        sel_type = int(cfg.get(f"fisheries.selectivity.type.fsh{fsh_idx}", "0"))
        if sel_type == 0:
            # Age-based knife-edge
            a50_key = f"fisheries.selectivity.a50.fsh{fsh_idx}"
            a50_val = cfg.get(a50_key, "0.0")
            fishing_a50[sp_idx] = float(a50_val)
            fishing_sel_type[sp_idx] = 0
        elif sel_type == 1:
            # Sigmoidal size selectivity
            fishing_sel_type[sp_idx] = 1
            fishing_l50[sp_idx] = float(cfg.get(f"fisheries.selectivity.l50.fsh{fsh_idx}", "0.0"))
            fishing_slope[sp_idx] = float(
                cfg.get(f"fisheries.selectivity.slope.fsh{fsh_idx}", "1.0")
            )
        else:
            fishing_sel_type[sp_idx] = sel_type

    return fishing_rate, fishing_a50, fishing_sel_type, fishing_l50, fishing_slope


def _load_fishing_seasonality(
    cfg: dict[str, str],
    n_species: int,
    n_dt_per_year: int,
    species_names: list[str] | None = None,
) -> NDArray[np.float64] | None:
    """Load fishing seasonality for each species (v3 sp{i} or v4 fsh{i} format).

    Supports:
    - ``fisheries.seasonality.file.sp{i}`` — per-species CSV file (v3)
    - ``fisheries.seasonality.file.fsh{i}`` — per-fishery CSV file (v4)
    - ``fisheries.seasonality.fsh{i}`` — inline semicolon-separated values (v4)

    Returns array of shape (n_species, n_dt_per_year) with season weights.
    None if no seasonality found.
    """
    seasons = np.ones((n_species, n_dt_per_year), dtype=np.float64) / n_dt_per_year
    found_any = False

    # Build fishery-to-species mapping for v4 fisheries
    fsh_to_sp: dict[int, int] = {}
    catch_file = cfg.get("fisheries.catchability.file", "")
    if catch_file and species_names:
        catch_path = _require_file(catch_file, _cfg_dir(cfg), "fisheries.catchability.file")
        catch_df = pd.read_csv(catch_path, index_col=0)
        for row_idx in range(len(catch_df)):
            row_name = str(catch_df.index[row_idx]).strip().lower()
            for col_idx in range(len(catch_df.columns)):
                if float(catch_df.iloc[row_idx, col_idx]) > 0:
                    for sp_idx, name in enumerate(species_names):
                        if name.strip().lower() == row_name:
                            fsh_to_sp[col_idx] = sp_idx
                            break
                    break

    def _set_season(sp_idx: int, values: NDArray[np.float64]) -> None:
        nonlocal found_any
        if len(values) >= n_dt_per_year:
            vals = values[:n_dt_per_year]
            total = vals.sum()
            if total > 0:
                seasons[sp_idx] = vals / total
                found_any = True

    # Try v3 per-species file keys first
    for i in range(n_species):
        file_key = cfg.get(f"fisheries.seasonality.file.sp{i}", "")
        if not file_key:
            continue
        path = _require_file(file_key, _cfg_dir(cfg), f"fisheries.seasonality.file.sp{i}")
        df = pd.read_csv(path, sep=";")
        _set_season(i, df.iloc[:, 1].values.astype(np.float64))

    # Try v4 per-fishery keys (file or inline)
    n_fisheries = int(cfg.get("simulation.nfisheries", "0"))
    for fsh in range(n_fisheries):
        sp_idx = fsh_to_sp.get(fsh)
        if sp_idx is None:
            continue

        # Try file reference
        file_key = cfg.get(f"fisheries.seasonality.file.fsh{fsh}", "")
        if file_key:
            path = _require_file(file_key, _cfg_dir(cfg), f"fisheries.seasonality.file.fsh{fsh}")
            df = pd.read_csv(path, sep=";")
            _set_season(sp_idx, df.iloc[:, 1].values.astype(np.float64))
            continue

        # Try inline semicolon-separated values
        inline_val = cfg.get(f"fisheries.seasonality.fsh{fsh}", "")
        if inline_val:
            try:
                vals = np.array(
                    [float(v.strip()) for v in inline_val.split(";") if v.strip()],
                    dtype=np.float64,
                )
                _set_season(sp_idx, vals)
            except (ValueError, TypeError) as exc:
                warnings.warn(
                    f"Invalid fisheries.seasonality.fsh{fsh} value: {inline_val!r} — {exc}",
                    stacklevel=2,
                )

    return seasons if found_any else None


def _load_per_species_timeseries(
    cfg: dict[str, str], n_species: int, key_pattern: str, context_prefix: str
) -> list[NDArray[np.float64] | None] | None:
    """Load a per-species time-varying CSV into a list of flattened arrays.

    Shared implementation for fishing rate and additional mortality loaders.
    ``key_pattern`` should contain ``{i}`` which is formatted per species index.
    Returns list of arrays (one per species), or None if no files found.
    """
    result: list[NDArray[np.float64] | None] = [None] * n_species
    found_any = False
    for i in range(n_species):
        file_key = cfg.get(key_pattern.format(i=i), "")
        if not file_key:
            continue
        path = _require_file(file_key, _cfg_dir(cfg), key_pattern.format(i=i))
        values = np.loadtxt(path, dtype=np.float64)
        result[i] = values.flatten()
        found_any = True
    return result if found_any else None


def _load_fishing_rate_by_year(
    cfg: dict[str, str], n_species: int
) -> list[NDArray[np.float64] | None] | None:
    """Load time-varying annual fishing rate CSV for each species."""
    return _load_per_species_timeseries(
        cfg, n_species, "mortality.fishing.rate.byyear.file.sp{i}", "fishing_rate_by_year"
    )


def _parse_growth_params(
    cfg: dict[str, str], n_sp: int, n_dt: int, lifespan_years: NDArray[np.float64]
) -> dict[str, Any]:
    """Parse Von Bertalanffy growth, allometry, lifespan, and additional mortality."""
    linf = _species_float(cfg, "species.linf.sp{i}", n_sp)
    k = _species_float(cfg, "species.k.sp{i}", n_sp)
    t0 = _species_float(cfg, "species.t0.sp{i}", n_sp)
    egg_size = _species_float(cfg, "species.egg.size.sp{i}", n_sp)
    condition_factor = _species_float(cfg, "species.length2weight.condition.factor.sp{i}", n_sp)
    allometric_power = _species_float(cfg, "species.length2weight.allometric.power.sp{i}", n_sp)
    vb_threshold_age = _species_float(cfg, "species.vonbertalanffy.threshold.age.sp{i}", n_sp)
    lifespan_dt = (lifespan_years * n_dt).astype(np.int32)
    delta_lmax_factor = _species_float_optional(
        cfg, "species.delta.lmax.factor.sp{i}", n_sp, default=2.0
    )
    additional_mortality_rate = _species_float_optional(
        cfg, "mortality.additional.rate.sp{i}", n_sp, default=0.0
    )
    lmax = _species_float_optional(cfg, "species.lmax.sp{i}", n_sp, default=0.0)
    for i in range(n_sp):
        if lmax[i] <= 0:
            lmax[i] = linf[i]
    return {
        "focal_linf": linf,
        "focal_k": k,
        "focal_t0": t0,
        "focal_egg_size": egg_size,
        "focal_condition_factor": condition_factor,
        "focal_allometric_power": allometric_power,
        "focal_vb_threshold_age": vb_threshold_age,
        "focal_lifespan_dt": lifespan_dt,
        "focal_delta_lmax_factor": delta_lmax_factor,
        "focal_additional_mortality_rate": additional_mortality_rate,
        "focal_lmax": lmax,
    }


def _parse_reproduction_params(
    cfg: dict[str, str],
    n_sp: int,
    n_dt: int,
    lifespan_years: NDArray[np.float64],
) -> dict[str, Any]:
    """Parse reproduction, seeding, and larva mortality parameters."""
    sex_ratio = _species_float_optional(cfg, "species.sexratio.sp{i}", n_sp, default=0.5)
    # H10: sex ratio is a probability — bounded [0, 1].
    bad_sr = np.where((sex_ratio < 0.0) | (sex_ratio > 1.0))[0]
    if len(bad_sr) > 0:
        i = int(bad_sr[0])
        raise ValueError(f"species.sexratio.sp{i} must be in [0, 1], got {float(sex_ratio[i])}")
    relative_fecundity = _species_float_optional(
        cfg, "species.relativefecundity.sp{i}", n_sp, default=500.0
    )
    # H10: fecundity is a count — non-negative. (Zero is a valid degenerate
    # "no reproduction" case used in tests and some calibration scenarios;
    # negative values are nonsensical and indicate a config error.)
    bad_rf = np.where(relative_fecundity < 0.0)[0]
    if len(bad_rf) > 0:
        i = int(bad_rf[0])
        raise ValueError(
            f"species.relativefecundity.sp{i} must be >= 0, got {float(relative_fecundity[i])}"
        )
    maturity_size = _species_float_optional(cfg, "species.maturity.size.sp{i}", n_sp, default=0.0)
    seeding_biomass = _species_float_optional(
        cfg, "population.seeding.biomass.sp{i}", n_sp, default=0.0
    )
    # Seeding max step: global-only key — verified 2026-04-12 against Java OSMOSE
    # source (ReproductionProcess.java:83,146). Java uses a single `int yearMaxSeeding`
    # for all species, NOT per-species. Python's np.full broadcast is correct parity.
    seeding_max_year_str = cfg.get("population.seeding.year.max", "")
    if seeding_max_year_str:
        seeding_max_years = float(seeding_max_year_str)
        seeding_max_step = np.full(n_sp, int(seeding_max_years * n_dt), dtype=np.int32)
    else:
        seeding_max_step = (lifespan_years * n_dt).astype(np.int32)
    # Warm-start standing-stock init disables egg-seeding: the standing population IS the
    # initialization, so the SSB==0 egg-rescue must not continuously re-inject a suppressed
    # species (which would stop a cod-absent / clupeid-dominated basin from being maintained).
    # Gated on the (default-off) canonical flag, so parity is preserved when warm-start is off.
    if cfg.get("module.population.initialisation.enabled", "false").lower() == "true":
        seeding_max_step = np.zeros(n_sp, dtype=np.int32)
    larva_mortality_rate = _species_float_optional(
        cfg, "mortality.additional.larva.rate.sp{i}", n_sp, default=0.0
    )
    maturity_age_years = _species_float_optional(
        cfg, "species.maturity.age.sp{i}", n_sp, default=0.0
    )
    maturity_age_dt = (maturity_age_years * n_dt).astype(np.int32)
    recruitment_type = _species_str_optional(
        cfg,
        "stock.recruitment.type.sp{i}",
        n_sp,
        default="none",
        allowed={"none", "beverton_holt", "ricker", "hockey_stick", "shepherd"},
    )
    recruitment_ssb_half = _species_float_optional(
        cfg, "stock.recruitment.ssbhalf.sp{i}", n_sp, default=0.0
    )
    recruitment_shepherd_beta = _species_float_optional(
        cfg, "stock.recruitment.shape.sp{i}", n_sp, default=1.0
    )
    for i in range(n_sp):
        if recruitment_type[i] != "none" and recruitment_ssb_half[i] <= 0.0:
            raise ValueError(
                f"stock.recruitment.ssbhalf.sp{i} must be > 0 when "
                f"stock.recruitment.type.sp{i}={recruitment_type[i]!r}"
            )
        if recruitment_type[i] == "shepherd" and recruitment_shepherd_beta[i] <= 0.0:
            raise ValueError(
                f"stock.recruitment.shape.sp{i} must be > 0 when "
                f"stock.recruitment.type.sp{i}={recruitment_type[i]!r}"
            )
    # Predator functional response (post-parity; opt-in, default type1 ≡ existing)
    fr_shape_str = _species_str_optional(
        cfg,
        "predation.functional.response.shape.sp{i}",
        n_sp,
        default="type1",
        allowed={"type1", "type2", "type3"},
    )
    # Validate halfsat presence/range before building the float array so that
    # missing or None values raise a clear domain error rather than a TypeError.
    for i in range(n_sp):
        if fr_shape_str[i] != "type1":
            hv = cfg.get(f"predation.functional.response.halfsat.sp{i}")
            if hv is None:
                raise ValueError(
                    f"predation.functional.response.halfsat.sp{i} is required when "
                    f"predation.functional.response.shape.sp{i} = {fr_shape_str[i]}"
                )
            if not (0.1 <= float(hv) <= 5.0):
                raise ValueError(
                    f"predation.functional.response.halfsat.sp{i} = {hv} out of range [0.1, 5.0]"
                )
    fr_halfsat_focal = _species_float_optional(
        cfg, "predation.functional.response.halfsat.sp{i}", n_sp, default=_FR_HALFSAT_SENTINEL
    )
    fr_shape_focal = np.array([_FR_SHAPE_CODE[s] for s in fr_shape_str], dtype=np.int32)
    return {
        "focal_sex_ratio": sex_ratio,
        "focal_relative_fecundity": relative_fecundity,
        "focal_maturity_size": maturity_size,
        "focal_seeding_biomass": seeding_biomass,
        "focal_seeding_max_step": seeding_max_step,
        "focal_larva_mortality_rate": larva_mortality_rate,
        "focal_maturity_age_dt": maturity_age_dt,
        "focal_recruitment_type": recruitment_type,
        "focal_recruitment_ssb_half": recruitment_ssb_half,
        "focal_recruitment_shepherd_beta": recruitment_shepherd_beta,
        "focal_fr_shape": fr_shape_focal,
        "focal_fr_halfsat": fr_halfsat_focal,
    }


def _parse_predation_params(
    cfg: dict[str, str],
    n_sp: int,
    n_dt: int,
    background_list: list[BackgroundSpeciesInfo],
    focal_fishing_l50_fsh: NDArray[np.float64],
) -> dict[str, Any]:
    """Parse feeding stages, size ratios, predation, and post-predation params."""
    n_bkg = len(background_list)
    focal_ingestion_rate = _species_float(cfg, "predation.ingestion.rate.max.sp{i}", n_sp)
    focal_critical_success_rate = _species_float(cfg, "predation.efficiency.critical.sp{i}", n_sp)

    _VALID_METRICS = {"age", "size", "weight", "tl"}
    global_metric = cfg.get("predation.predprey.stage.structure", "size").strip().lower()
    if global_metric not in _VALID_METRICS:
        raise ValueError(
            f"Unrecognized feeding stage metric: {global_metric!r}. "
            f"Must be one of {sorted(_VALID_METRICS)}."
        )

    all_thresholds: list[list[float]] = []
    all_metrics: list[str] = []
    all_ratio_min: list[list[float]] = []
    all_ratio_max: list[list[float]] = []

    for i in range(n_sp):
        sp_metric = cfg.get(f"predation.predprey.stage.structure.sp{i}", "").strip().lower()
        if not sp_metric:
            sp_metric = global_metric
        elif sp_metric not in _VALID_METRICS:
            raise ValueError(
                f"Unrecognized feeding stage metric for sp{i}: {sp_metric!r}. "
                f"Must be one of {sorted(_VALID_METRICS)}."
            )
        all_metrics.append(sp_metric)

        thresh_raw = cfg.get(f"predation.predprey.stage.threshold.sp{i}", "")
        if not thresh_raw or thresh_raw.strip().lower() == "null":
            sp_thresholds: list[float] = []
        else:
            sp_thresholds = _parse_floats(thresh_raw)
        all_thresholds.append(sp_thresholds)
        n_stages = len(sp_thresholds) + 1

        rmin_raw = cfg.get(f"predation.predprey.sizeratio.min.sp{i}", "1.0")
        rmax_raw = cfg.get(f"predation.predprey.sizeratio.max.sp{i}", "3.5")
        rmin_list = _parse_floats(rmin_raw)
        rmax_list = _parse_floats(rmax_raw)

        if len(rmin_list) != n_stages:
            raise ValueError(
                f"Size ratio min count mismatch for sp{i}: "
                f"got {len(rmin_list)}, expected {n_stages} stages"
            )
        if len(rmax_list) != n_stages:
            raise ValueError(
                f"Size ratio max count mismatch for sp{i}: "
                f"got {len(rmax_list)}, expected {n_stages} stages"
            )

        for s in range(n_stages):
            if rmin_list[s] > rmax_list[s]:
                warnings.warn(
                    f"Swapping size ratios for sp{i} stage {s}: "
                    f"min={rmin_list[s]}, max={rmax_list[s]}",
                    stacklevel=2,
                )
                rmin_list[s], rmax_list[s] = rmax_list[s], rmin_list[s]

        all_ratio_min.append(rmin_list)
        all_ratio_max.append(rmax_list)

    # Background species
    for b in background_list:
        b_idx = b.file_index
        b_metric = cfg.get(f"predation.predprey.stage.structure.sp{b_idx}", "").strip().lower()
        if not b_metric:
            b_metric = global_metric
        all_metrics.append(b_metric)

        thresh_raw = cfg.get(f"predation.predprey.stage.threshold.sp{b_idx}", "")
        if not thresh_raw or thresh_raw.strip().lower() == "null":
            b_thresholds: list[float] = []
        else:
            b_thresholds = _parse_floats(thresh_raw)
        all_thresholds.append(b_thresholds)
        n_stages = len(b_thresholds) + 1

        rmin_list = list(b.size_ratio_min)
        rmax_list = list(b.size_ratio_max)
        if len(rmin_list) == 1 and n_stages > 1:
            rmin_list = rmin_list * n_stages
        if len(rmax_list) == 1 and n_stages > 1:
            rmax_list = rmax_list * n_stages
        for s in range(min(len(rmin_list), len(rmax_list))):
            if rmin_list[s] > rmax_list[s]:
                rmin_list[s], rmax_list[s] = rmax_list[s], rmin_list[s]
        all_ratio_min.append(rmin_list)
        all_ratio_max.append(rmax_list)

    # Build 2D arrays
    n_total = n_sp + n_bkg
    max_stages = max((len(r) for r in all_ratio_min), default=1)
    n_feeding_stages = np.array([len(r) for r in all_ratio_min], dtype=np.int32)
    size_ratio_min_2d = np.zeros((n_total, max_stages), dtype=np.float64)
    size_ratio_max_2d = np.zeros((n_total, max_stages), dtype=np.float64)
    for sp_i in range(n_total):
        n_st = len(all_ratio_min[sp_i])
        for s in range(n_st):
            size_ratio_min_2d[sp_i, s] = all_ratio_min[sp_i][s]
            size_ratio_max_2d[sp_i, s] = all_ratio_max[sp_i][s]
        if n_st > 0 and n_st < max_stages:
            size_ratio_min_2d[sp_i, n_st:] = all_ratio_min[sp_i][-1]
            size_ratio_max_2d[sp_i, n_st:] = all_ratio_max[sp_i][-1]

    # Post-predation variables
    focal_starvation_rate_max = _species_float_optional(
        cfg, "mortality.starvation.rate.max.sp{i}", n_sp, default=0.0
    )
    focal_fishing_selectivity_l50 = _species_float_optional(
        cfg, "fishing.selectivity.l50.sp{i}", n_sp, default=0.0
    )
    for i in range(n_sp):
        if focal_fishing_l50_fsh[i] > 0 and focal_fishing_selectivity_l50[i] == 0:
            focal_fishing_selectivity_l50[i] = focal_fishing_l50_fsh[i]
    focal_movement_method = [
        cfg.get(f"movement.distribution.method.sp{i}", "random") for i in range(n_sp)
    ]
    focal_random_walk_range = _species_int_optional(
        cfg, "movement.randomwalk.range.sp{i}", n_sp, default=1
    )
    focal_out_mortality_rate = _species_float_optional(
        cfg, "mortality.out.rate.sp{i}", n_sp, default=0.0
    )
    focal_n_schools = _species_int_optional(
        cfg,
        "simulation.nschool.sp{i}",
        n_sp,
        default=int(cfg.get("simulation.nschool", "20")),
    )

    return {
        "focal_ingestion_rate": focal_ingestion_rate,
        "focal_critical_success_rate": focal_critical_success_rate,
        "focal_starvation_rate_max": focal_starvation_rate_max,
        "focal_fishing_selectivity_l50": focal_fishing_selectivity_l50,
        "focal_movement_method": focal_movement_method,
        "focal_random_walk_range": focal_random_walk_range,
        "focal_out_mortality_rate": focal_out_mortality_rate,
        "focal_n_schools": focal_n_schools,
        "all_thresholds": all_thresholds,
        "all_metrics": all_metrics,
        "n_feeding_stages": n_feeding_stages,
        "size_ratio_min_2d": size_ratio_min_2d,
        "size_ratio_max_2d": size_ratio_max_2d,
    }


def _merge_focal_background(
    focal: dict[str, Any],
    background_list: list[BackgroundSpeciesInfo],
    focal_species_names: list[str],
    focal_fishing_spatial_maps: list[np.ndarray | None],
    focal_movement_method: list[str],
) -> dict[str, Any]:
    """Concatenate focal-species arrays with background defaults."""
    n_bkg = len(background_list)
    if n_bkg > 0:
        bkg_names = [b.name for b in background_list]
        bkg_ingestion = np.array([b.ingestion_rate for b in background_list])
        bkg_fr_shape = np.array([b.fr_shape for b in background_list], dtype=np.int32)
        bkg_fr_halfsat = np.array([b.fr_halfsat for b in background_list], dtype=np.float64)
        bkg_condition_factor = np.array([b.condition_factor for b in background_list])
        bkg_allometric_power = np.array([b.allometric_power for b in background_list])
        bkg_zeros_f = np.zeros(n_bkg, dtype=np.float64)
        bkg_zeros_i = np.zeros(n_bkg, dtype=np.int32)

        return {
            "all_species_names": focal_species_names + bkg_names,
            "linf": np.concatenate([focal["focal_linf"], bkg_zeros_f]),
            "k": np.concatenate([focal["focal_k"], bkg_zeros_f]),
            "t0": np.concatenate([focal["focal_t0"], bkg_zeros_f]),
            "egg_size": np.concatenate([focal["focal_egg_size"], bkg_zeros_f]),
            "condition_factor": np.concatenate(
                [focal["focal_condition_factor"], bkg_condition_factor]
            ),
            "allometric_power": np.concatenate(
                [focal["focal_allometric_power"], bkg_allometric_power]
            ),
            "vb_threshold_age": np.concatenate([focal["focal_vb_threshold_age"], bkg_zeros_f]),
            "lifespan_dt": np.concatenate([focal["focal_lifespan_dt"], bkg_zeros_i]),
            "ingestion_rate": np.concatenate([focal["focal_ingestion_rate"], bkg_ingestion]),
            "critical_success_rate": np.concatenate(
                [focal["focal_critical_success_rate"], bkg_zeros_f]
            ),
            "delta_lmax_factor": np.concatenate([focal["focal_delta_lmax_factor"], bkg_zeros_f]),
            "additional_mortality_rate": np.concatenate(
                [focal["focal_additional_mortality_rate"], bkg_zeros_f]
            ),
            "sex_ratio": np.concatenate([focal["focal_sex_ratio"], bkg_zeros_f]),
            "relative_fecundity": np.concatenate([focal["focal_relative_fecundity"], bkg_zeros_f]),
            "maturity_size": np.concatenate([focal["focal_maturity_size"], bkg_zeros_f]),
            "seeding_biomass": np.concatenate([focal["focal_seeding_biomass"], bkg_zeros_f]),
            "seeding_max_step": np.concatenate([focal["focal_seeding_max_step"], bkg_zeros_i]),
            "larva_mortality_rate": np.concatenate(
                [focal["focal_larva_mortality_rate"], bkg_zeros_f]
            ),
            "maturity_age_dt": np.concatenate([focal["focal_maturity_age_dt"], bkg_zeros_i]),
            "recruitment_type": (focal["focal_recruitment_type"] + ["none"] * n_bkg),
            "recruitment_ssb_half": np.concatenate(
                [focal["focal_recruitment_ssb_half"], bkg_zeros_f]
            ),
            "recruitment_shepherd_beta": np.concatenate(
                [focal["focal_recruitment_shepherd_beta"], np.ones(n_bkg, dtype=np.float64)]
            ),
            "fr_shape": np.concatenate([focal["focal_fr_shape"], bkg_fr_shape]),
            "fr_halfsat": np.concatenate([focal["focal_fr_halfsat"], bkg_fr_halfsat]),
            "lmax": np.concatenate([focal["focal_lmax"], bkg_zeros_f]),
            "starvation_rate_max": np.concatenate(
                [focal["focal_starvation_rate_max"], bkg_zeros_f]
            ),
            "fishing_rate": np.concatenate([focal["fishing"], bkg_zeros_f]),
            "fishing_selectivity_l50": np.concatenate(
                [focal["focal_fishing_selectivity_l50"], bkg_zeros_f]
            ),
            "fishing_selectivity_a50": np.concatenate(
                [focal["focal_fishing_a50"], np.full(n_bkg, np.nan, dtype=np.float64)]
            ),
            "fishing_selectivity_type": np.concatenate(
                [focal["focal_fishing_sel_type"], np.full(n_bkg, -1, dtype=np.int32)]
            ),
            "fishing_selectivity_slope": np.concatenate(
                [focal["focal_fishing_slope"], bkg_zeros_f]
            ),
            "movement_method": focal_movement_method + ["none"] * n_bkg,
            "random_walk_range": np.concatenate([focal["focal_random_walk_range"], bkg_zeros_i]),
            "out_mortality_rate": np.concatenate([focal["focal_out_mortality_rate"], bkg_zeros_f]),
            "n_schools": np.concatenate([focal["focal_n_schools"], bkg_zeros_i]),
            "fishing_spatial_maps": focal_fishing_spatial_maps + [None] * n_bkg,
        }
    else:
        return {
            "all_species_names": focal_species_names[:],
            "linf": focal["focal_linf"],
            "k": focal["focal_k"],
            "t0": focal["focal_t0"],
            "egg_size": focal["focal_egg_size"],
            "condition_factor": focal["focal_condition_factor"],
            "allometric_power": focal["focal_allometric_power"],
            "vb_threshold_age": focal["focal_vb_threshold_age"],
            "lifespan_dt": focal["focal_lifespan_dt"],
            "ingestion_rate": focal["focal_ingestion_rate"],
            "critical_success_rate": focal["focal_critical_success_rate"],
            "delta_lmax_factor": focal["focal_delta_lmax_factor"],
            "additional_mortality_rate": focal["focal_additional_mortality_rate"],
            "sex_ratio": focal["focal_sex_ratio"],
            "relative_fecundity": focal["focal_relative_fecundity"],
            "maturity_size": focal["focal_maturity_size"],
            "seeding_biomass": focal["focal_seeding_biomass"],
            "seeding_max_step": focal["focal_seeding_max_step"],
            "larva_mortality_rate": focal["focal_larva_mortality_rate"],
            "maturity_age_dt": focal["focal_maturity_age_dt"],
            "recruitment_type": focal["focal_recruitment_type"],
            "recruitment_ssb_half": focal["focal_recruitment_ssb_half"],
            "recruitment_shepherd_beta": focal["focal_recruitment_shepherd_beta"],
            "fr_shape": focal["focal_fr_shape"],
            "fr_halfsat": focal["focal_fr_halfsat"],
            "lmax": focal["focal_lmax"],
            "starvation_rate_max": focal["focal_starvation_rate_max"],
            "fishing_rate": focal["fishing"],
            "fishing_selectivity_l50": focal["focal_fishing_selectivity_l50"],
            "fishing_selectivity_a50": focal["focal_fishing_a50"],
            "fishing_selectivity_type": focal["focal_fishing_sel_type"],
            "fishing_selectivity_slope": focal["focal_fishing_slope"],
            "movement_method": focal_movement_method,
            "random_walk_range": focal["focal_random_walk_range"],
            "out_mortality_rate": focal["focal_out_mortality_rate"],
            "n_schools": focal["focal_n_schools"],
            "fishing_spatial_maps": focal_fishing_spatial_maps,
        }


def _parse_output_flags(cfg: dict[str, str], n_sp: int, n_bkg: int) -> dict[str, Any]:
    """Parse output recording flags and distribution settings."""
    output_record_freq = int(cfg.get("output.recordfrequency.ndt", "1"))
    diet_output = cfg.get("output.diet.composition.enabled", "false").lower() == "true"
    step0 = cfg.get("output.step0.include", "false").lower() == "true"

    # Output cutoff age
    cutoff_vals = []
    found_any = False
    for i in range(n_sp):
        val = cfg.get(f"output.cutoff.age.sp{i}", "")
        if val and val.lower() not in ("null", "none", ""):
            cutoff_vals.append(float(val))
            found_any = True
        else:
            cutoff_vals.append(0.0)
    cutoff_vals.extend([0.0] * n_bkg)
    cutoff_age = np.array(cutoff_vals, dtype=np.float64) if found_any else None

    return {
        "output_record_frequency": output_record_freq,
        "diet_output_enabled": diet_output,
        "output_step0_include": step0,
        "output_cutoff_age": cutoff_age,
        "output_biomass_byage": _enabled(cfg, "output.biomass.byage.enabled"),
        "output_biomass_bysize": _enabled(cfg, "output.biomass.bysize.enabled"),
        "output_abundance_byage": _enabled(cfg, "output.abundance.byage.enabled"),
        "output_abundance_bysize": _enabled(cfg, "output.abundance.bysize.enabled"),
        # #121: read the real upstream name (output.tl.enabled) first; the osmopy-invented
        # output.meantl.enabled remains a back-compat fallback.
        "output_meantl": _enabled(cfg, "output.tl.enabled")
        or _enabled(cfg, "output.meantl.enabled"),
        # Three pre-existing schema keys that were declared but not parsed
        "output_biomass_netcdf": _enabled(cfg, "output.biomass.netcdf.enabled"),
        "output_abundance_netcdf": _enabled(cfg, "output.abundance.netcdf.enabled"),
        "output_yield_biomass_netcdf": _enabled(cfg, "output.yield.biomass.netcdf.enabled"),
        "output_yield_abundance": _enabled(cfg, "output.yield.abundance.enabled"),
        "output_mean_size": _enabled(cfg, "output.size.enabled"),
        "output_yield_abundance_netcdf": _enabled(cfg, "output.yield.abundance.netcdf.enabled"),
        "output_mean_size_netcdf": _enabled(cfg, "output.size.netcdf.enabled"),
        "output_ssb": _enabled(cfg, "output.ssb.enabled"),
        "output_ssb_netcdf": _enabled(cfg, "output.ssb.netcdf.enabled"),
        # Five new keys
        "output_biomass_byage_netcdf": _enabled(cfg, "output.biomass.byage.netcdf.enabled"),
        "output_abundance_byage_netcdf": _enabled(cfg, "output.abundance.byage.netcdf.enabled"),
        "output_biomass_bysize_netcdf": _enabled(cfg, "output.biomass.bysize.netcdf.enabled"),
        "output_abundance_bysize_netcdf": _enabled(cfg, "output.abundance.bysize.netcdf.enabled"),
        "output_mortality_netcdf": _enabled(cfg, "output.mortality.netcdf.enabled"),
        # Spatial output flags (output.spatial.* schema keys)
        "output_spatial_enabled": _enabled(cfg, "output.spatial.enabled"),
        "output_spatial_biomass": _enabled(cfg, "output.spatial.biomass.enabled"),
        "output_spatial_abundance": _enabled(cfg, "output.spatial.abundance.enabled"),
        "output_spatial_yield_biomass": _enabled(cfg, "output.spatial.yield.biomass.enabled"),
        "output_size_min": float(cfg.get("output.distrib.bysize.min", "0")),
        "output_size_max": float(cfg.get("output.distrib.bysize.max", "205")),
        "output_size_incr": float(cfg.get("output.distrib.bysize.incr", "10")),
        "output_bioen_ingest": cfg.get("output.bioen.ingest.enabled", "false").lower() == "true",
        "output_bioen_maint": cfg.get("output.bioen.maint.enabled", "false").lower() == "true",
        "output_bioen_rho": cfg.get("output.bioen.rho.enabled", "false").lower() == "true",
        "output_bioen_sizeinf": (
            cfg.get("output.bioen.sizeinf.enabled", "false").lower() == "true"
        ),
    }


def _parse_mpa_zones(cfg: dict[str, str]) -> list[MPAZone] | None:
    """Parse Marine Protected Area configurations."""
    zones: list[MPAZone] = []
    i = 0
    while True:
        file_key = cfg.get(f"mpa.file.mpa{i}", "")
        if not file_key:
            break
        path = _require_file(file_key, _cfg_dir(cfg), f"mpa.file.mpa{i}")
        grid = _load_spatial_csv(path)
        start_year = int(cfg.get(f"mpa.start.year.mpa{i}", "0"))
        end_year = int(cfg.get(f"mpa.end.year.mpa{i}", "999"))
        percentage = float(cfg.get(f"mpa.percentage.mpa{i}", "1.0"))
        zones.append(
            MPAZone(grid=grid, start_year=start_year, end_year=end_year, percentage=percentage)
        )
        i += 1

    return zones if zones else None


def _load_discard_rates(
    cfg: dict[str, str], species_names: list[str], n_species: int
) -> NDArray[np.float64] | None:
    """Load fishery discard rates from CSV.

    Returns per-species discard rate array, or None if no discard file.
    """
    file_key = cfg.get("fisheries.discards.file", "")
    if not file_key:
        return None
    path = _require_file(file_key, _cfg_dir(cfg), "fisheries.discards.file")

    df = pd.read_csv(path, index_col=0)
    discard_rate = np.zeros(n_species, dtype=np.float64)

    for sp_idx in range(n_species):
        sp_name = species_names[sp_idx].strip()
        if sp_name in df.index:
            row = df.loc[sp_name]
            # Take the first nonzero discard rate (assumes one primary fishery per species)
            vals = row.values.astype(np.float64)
            nonzero = vals[vals > 0]
            if len(nonzero) > 0:
                discard_rate[sp_idx] = nonzero[0]

    return discard_rate


def _load_spawning_seasons(
    cfg: dict[str, str], n_species: int, n_dt_per_year: int
) -> NDArray[np.float64] | None:
    """Load spawning season CSV files for each species.

    Returns array of shape (n_species, n_columns) with season weights.
    n_columns equals n_dt_per_year for single-year data, or n_dt_per_year * n_years
    for multi-year time series.
    """
    normalize = cfg.get("reproduction.normalisation.enabled", "false").lower() == "true"

    # First pass: load all values to determine max column count
    all_values: list[NDArray[np.float64] | None] = [None] * n_species
    max_cols = n_dt_per_year
    found_any = False

    for i in range(n_species):
        file_key = cfg.get(f"reproduction.season.file.sp{i}", "")
        if not file_key:
            continue
        path = _require_file(file_key, _cfg_dir(cfg), f"reproduction.season.file.sp{i}")
        df = pd.read_csv(path, sep=";")
        values = df.iloc[:, 1].values.astype(np.float64)
        if len(values) >= n_dt_per_year:
            all_values[i] = values
            max_cols = max(max_cols, len(values))
            found_any = True
            # H10: spawning-season vectors should sum to 1.0 per year (each
            # entry is the fraction of annual reproduction in that step).
            # If `reproduction.normalisation.enabled=true` the engine will
            # auto-correct downstream; otherwise a non-unit sum produces
            # silently-wrong reproduction. Warn so the config author can fix
            # the input rather than rely on the implicit normaliser.
            if not normalize:
                n_years = max(1, len(values) // n_dt_per_year)
                annual_mean = float(values.sum()) / n_years
                if not np.isclose(annual_mean, 1.0, atol=0.01):
                    warnings.warn(
                        f"reproduction.season.file.sp{i}: per-year sum is {annual_mean:.4f}, "
                        f"expected 1.0 (set reproduction.normalisation.enabled=true to "
                        f"auto-rescale, or fix the input).",
                        stacklevel=2,
                    )

    if not found_any:
        return None

    seasons = np.ones((n_species, max_cols), dtype=np.float64) / n_dt_per_year
    for i in range(n_species):
        if all_values[i] is None:
            continue
        vals: NDArray[np.float64] = cast(NDArray[np.float64], all_values[i])
        n_vals = len(vals)
        if normalize:
            # Normalize each n_dt_per_year-sized chunk independently so
            # every year's weights sum to 1.0, not the whole multi-year array.
            # A trailing partial-year chunk is also normalized to sum 1.0;
            # warn because a non-whole-year file is usually a data error.
            if n_vals % n_dt_per_year != 0:
                _log.warning(
                    "Spawning season file for species %d has %d rows "
                    "which is not a multiple of n_dt_per_year=%d; "
                    "normalizing the %d-row trailing partial chunk to sum 1.0",
                    i,
                    n_vals,
                    n_dt_per_year,
                    n_vals % n_dt_per_year,
                )
            n_chunks = (n_vals + n_dt_per_year - 1) // n_dt_per_year  # ceil
            for yr in range(n_chunks):
                s = yr * n_dt_per_year
                e = min(s + n_dt_per_year, n_vals)
                if vals is None:
                    continue
                chunk_sum = vals[s:e].sum()
                if chunk_sum > 0:
                    vals[s:e] = vals[s:e] / chunk_sum
        seasons[i, :n_vals] = vals
        # Pad remaining columns with uniform if multi-year array is shorter
        if n_vals < max_cols:
            seasons[i, n_vals:] = 1.0 / n_dt_per_year

    return seasons


def _load_rv_gate(
    cfg: dict[str, str], n_species: int, n_dt_per_year: int, n_year: int
) -> tuple[NDArray[np.float64] | None, NDArray[np.bool_] | None, int]:
    """Load the reproductive-volume recruitment gate (spec §3.2/§4/§8).

    Returns (factor_by_index, enabled_mask, offset). factor_by_index has length
    n_years (number of series rows), is indexed by series index, and has the
    mode formula already applied. All three are (None, None, 0) when the master
    switch is off. Raises a clear error on any invalid configuration (fail-fast):
    ValueError for bad content/values, FileNotFoundError for a missing file.
    """
    if cfg.get("reproduction.rv.gate.enabled", "false").lower() != "true":
        return None, None, 0

    file_key = cfg.get("reproduction.rv.gate.series.file", "")
    if not file_key:
        raise ValueError("RV gate enabled but reproduction.rv.gate.series.file is empty.")
    path = _require_file(file_key, _cfg_dir(cfg), "reproduction.rv.gate.series.file")
    df = pd.read_csv(path)
    if df.shape[0] == 0 or "year" not in df.columns or "spawning_rv" not in df.columns:
        raise ValueError(f"RV gate series {path} has no data rows or wrong columns.")
    years = df["year"].to_numpy()
    rv = df["spawning_rv"].to_numpy(dtype=np.float64)
    first_year = int(years[0])
    if not np.array_equal(years, np.arange(first_year, first_year + len(years))):
        raise ValueError(f"RV gate series {path} years must be contiguous and ascending.")
    if np.any(~np.isfinite(rv)) or np.any(rv < 0):
        raise ValueError(f"RV gate series {path} has NaN or negative spawning_rv.")

    enabled = np.zeros(n_species, dtype=np.bool_)
    for sp in range(n_species):
        if cfg.get(f"reproduction.rv.gate.species.enabled.sp{sp}", "false").lower() == "true":
            enabled[sp] = True
    if not enabled.any():
        raise ValueError("RV gate enabled but no species enabled (…species.enabled.sp{idx}).")

    mode = cfg.get("reproduction.rv.gate.mode", "mean_preserving")
    floor = float(cfg.get("reproduction.rv.gate.floor", "0.0"))
    if not (0.0 <= floor <= 1.0):
        raise ValueError(f"reproduction.rv.gate.floor must be in [0,1], got {floor}.")
    start_year = int(cfg.get("reproduction.rv.gate.start.year", str(first_year)))
    n_years = len(rv)
    offset = start_year - first_year

    if mode == "mean_preserving":
        # Multiset mean over the sampled model years y=0..n_year-1 (with repeats).
        window_idx = [(offset + y) % n_years for y in range(n_year)]
        denom = float(np.mean(rv[window_idx]))
        if denom == 0.0:
            raise ValueError("RV gate mean_preserving denominator is 0 over the run window.")
        factor = rv / denom
    elif mode == "raw_cap":
        ref = float(cfg.get("reproduction.rv.gate.ref", "0.20"))
        if ref <= 0.0:
            raise ValueError(f"reproduction.rv.gate.ref must be > 0, got {ref}.")
        factor = np.clip(rv / ref, 0.0, 1.0)
    else:
        raise ValueError(f"unknown reproduction.rv.gate.mode: {mode!r}")

    factor = np.maximum(factor, floor)
    return factor.astype(np.float64), enabled, offset


def _load_rv_spatial(
    cfg: dict[str, str], n_species: int
) -> tuple[PhysicalData | None, NDArray[np.bool_] | None]:
    """Load the spatial RV egg-survival field (spec §5/§6). Returns (field, enable_mask)
    or (None, None) when the master switch is off. Fail-fast on invalid config."""
    from osmose.engine.physical_data import PhysicalData

    if cfg.get("reproduction.rv.spatial.enabled", "false").lower() != "true":
        return None, None

    file_key = cfg.get("reproduction.rv.spatial.field.file", "")
    if not file_key:
        raise ValueError("RV spatial enabled but reproduction.rv.spatial.field.file is empty.")
    path = _require_file(file_key, _cfg_dir(cfg), "reproduction.rv.spatial.field.file")
    varname = cfg.get("reproduction.rv.spatial.field.varname", "reproductive_volume")

    # Expected grid shape + spawning mask from the cod_spawning map (read directly;
    # MovementMapSet is not built at config time). Prefer a map under the config dir;
    # fall back to the bundled Baltic map.
    cfg_dir = Path(_cfg_dir(cfg))
    spawn_path = cfg_dir / "maps" / "cod_spawning.csv"
    if not spawn_path.exists():
        spawn_path = Path("data/baltic/maps/cod_spawning.csv")
    spawn = _load_spatial_csv(spawn_path) > 0  # north-first, (nlat, nlon)

    ref_cfg = float(cfg.get("reproduction.rv.spatial.ref", "-1"))
    import xarray as xr

    with xr.open_dataset(path) as ds:
        if varname not in ds:
            raise ValueError(f"RV field {path} has no variable {varname!r}.")
        grid_shape = ds[varname].shape[-2:]
        attr_ref = ds[varname].attrs.get("RV_ref", None)
    if tuple(grid_shape) != spawn.shape:
        raise ValueError(f"RV field grid {tuple(grid_shape)} != engine grid {spawn.shape}.")
    rv_ref = ref_cfg if ref_cfg > 0 else attr_ref
    if rv_ref is None or float(rv_ref) <= 0:
        raise ValueError("RV_ref not resolvable (ref<=0 and no positive RV_ref attr).")

    field = PhysicalData.from_netcdf_field(path, varname, float(rv_ref))
    n_dt = int(cfg.get("simulation.time.ndtperyear", "24"))
    assert field._data is not None
    tlen = field._data.shape[0]  # NetCDF time length (24 climatology / 696 interannual)
    if tlen <= 0 or tlen % n_dt != 0:
        raise ValueError(
            f"RV field time length {tlen} is not a positive multiple of ndtperyear {n_dt}."
        )
    # Interannual field (tlen > one year): a run longer than the forcing period would silently
    # wrap (step % tlen repeats year 0). Fail fast rather than hindcast against wrapped years.
    if tlen > n_dt:
        nyear = int(float(cfg.get("simulation.time.nyear", "0") or 0))
        if nyear * n_dt > tlen:
            raise ValueError(
                f"RV field is interannual ({tlen} steps = {tlen // n_dt} yr) but the run is "
                f"{nyear} yr ({nyear * n_dt} steps) — it would wrap past the forcing period. "
                f"Set simulation.time.nyear <= {tlen // n_dt}."
            )
    # Spec §6: a NaN at a cod_spawning cell means the field is broken there — fail loudly
    # rather than let the consumer's finite-guard silently no-op at a real spawning cell.
    if np.isnan(field._data[:, spawn]).any():
        raise ValueError(f"RV field {path} has NaN at cod_spawning cells.")

    enabled = np.zeros(n_species, dtype=np.bool_)
    for sp in range(n_species):
        if cfg.get(f"reproduction.rv.spatial.species.enabled.sp{sp}", "false").lower() == "true":
            enabled[sp] = True
    if not enabled.any():
        raise ValueError("RV spatial enabled but no species enabled (…species.enabled.sp{idx}).")
    return field, enabled


def _load_thermal_gate(
    cfg: dict[str, str], n_species: int, n_dt_per_year: int, n_year: int
) -> tuple[NDArray[np.float64] | None, NDArray[np.bool_] | None, int]:
    """Load the percid thermal recruitment gate (spec 2026-07-05).

    Returns (factor_by_index, enabled_mask, offset). factor_by_index has shape
    (n_years, n_species) with the logistic response + mode already applied;
    columns for disabled species are 1.0. All three are (None, None, 0) when the
    master switch is off. Raises a clear error on any invalid configuration.
    """
    from osmose.engine.processes.thermal_gate import logistic_response, normalize_factor

    if cfg.get("reproduction.thermal.gate.enabled", "false").lower() != "true":
        return None, None, 0

    file_key = cfg.get("reproduction.thermal.gate.series.file", "")
    if not file_key:
        raise ValueError("Thermal gate enabled but reproduction.thermal.gate.series.file is empty.")
    path = _require_file(file_key, _cfg_dir(cfg), "reproduction.thermal.gate.series.file")
    df = pd.read_csv(path)
    if df.shape[0] == 0 or "year" not in df.columns:
        raise ValueError(f"Thermal gate series {path} has no data rows or missing 'year' column.")
    years = df["year"].to_numpy()
    first_year = int(years[0])
    if not np.array_equal(years, np.arange(first_year, first_year + len(years))):
        raise ValueError(f"Thermal gate series {path} years must be contiguous and ascending.")
    n_years = len(years)

    enabled = np.zeros(n_species, dtype=np.bool_)
    for sp in range(n_species):
        if cfg.get(f"reproduction.thermal.gate.species.enabled.sp{sp}", "false").lower() == "true":
            enabled[sp] = True
    if not enabled.any():
        raise ValueError(
            "Thermal gate enabled but no species enabled "
            "(reproduction.thermal.gate.species.enabled.sp{idx})."
        )

    mode = cfg.get("reproduction.thermal.gate.mode", "thermal_cap")
    if mode not in ("thermal_cap", "mean_preserving"):
        raise ValueError(f"unknown reproduction.thermal.gate.mode: {mode!r}")
    floor = float(cfg.get("reproduction.thermal.gate.floor", "0.0"))
    if not (0.0 <= floor <= 1.0):
        raise ValueError(f"reproduction.thermal.gate.floor must be in [0,1], got {floor}.")
    start_year = int(cfg.get("reproduction.thermal.gate.start.year", str(first_year)))
    offset = start_year - first_year
    window_idx = [(offset + y) % n_years for y in range(n_year)]

    factor = np.ones((n_years, n_species), dtype=np.float64)
    for sp in range(n_species):
        if not enabled[sp]:
            continue
        col = f"temp_sp{sp}"
        if col not in df.columns:
            raise ValueError(
                f"Thermal gate series {path} missing column {col!r} for enabled sp{sp}."
            )
        temp = df[col].to_numpy(dtype=np.float64)
        if np.any(~np.isfinite(temp)) or np.any(temp < -2.0) or np.any(temp > 30.0):
            raise ValueError(
                f"Thermal gate series {path} column {col} has NaN or out-of-range (-2..30 C) values."
            )
        t50 = float(cfg.get(f"reproduction.thermal.gate.t50.sp{sp}", "18.5"))
        slope = float(cfg.get(f"reproduction.thermal.gate.slope.sp{sp}", "1.5"))
        if slope <= 0.0:
            raise ValueError(f"reproduction.thermal.gate.slope.sp{sp} must be > 0, got {slope}.")
        r = logistic_response(temp, t50, slope)
        if mode == "thermal_cap":
            tref = float(cfg.get(f"reproduction.thermal.gate.tref.sp{sp}", "20.0"))
            r_ref = float(logistic_response(np.array([tref]), t50, slope)[0])
            if r_ref <= 0.0:
                raise ValueError(f"thermal_cap r_ref for sp{sp} is 0 (check tref/t50/slope).")
        else:
            r_ref = 0.0
        factor[:, sp] = normalize_factor(r, mode, r_ref, window_idx, floor)

    return factor.astype(np.float64), enabled, offset


def _load_depensation_gate(
    cfg: dict[str, str], n_species: int
) -> tuple[NDArray[np.bool_] | None, NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """Load the recruitment depensation / Allee gate (spec 2026-07-16).

    Returns (enabled, s50, theta), each None when the gate is off. Raises a
    clear error on any invalid configuration (global-on with no species
    enabled, theta < 1, or s50 <= 0).
    """
    if not _enabled(cfg, "reproduction.depensation.gate.enabled"):
        return None, None, None

    enabled = np.zeros(n_species, dtype=np.bool_)
    s50 = np.zeros(n_species, dtype=np.float64)
    theta = np.ones(n_species, dtype=np.float64)
    for sp in range(n_species):
        if _enabled(cfg, f"reproduction.depensation.gate.species.enabled.sp{sp}"):
            enabled[sp] = True
            s50[sp] = float(cfg.get(f"reproduction.depensation.gate.s50.sp{sp}", "0"))
            theta[sp] = float(cfg.get(f"reproduction.depensation.gate.theta.sp{sp}", "1"))
            if theta[sp] < 1.0:
                raise ValueError(
                    f"reproduction.depensation.gate.theta.sp{sp}={theta[sp]} must be >= 1.0"
                )
            if s50[sp] <= 0.0:
                raise ValueError(f"reproduction.depensation.gate.s50.sp{sp}={s50[sp]} must be > 0")
    if not enabled.any():
        raise ValueError(
            "Depensation gate enabled but no species enabled "
            "(reproduction.depensation.gate.species.enabled.sp{idx})."
        )
    return enabled, s50, theta


def _load_salinity_gate(
    cfg: dict[str, str], n_species: int
) -> tuple[bool, NDArray[np.bool_] | None, float, float, PhysicalData | None]:
    """Load the salinity-gated occupancy config (spec 2026-07-04).

    Returns (enabled, species_mask, s_low, s_high, salinity_field). Off →
    (False, None, 3.0, 6.0, None). Fail-fast (ValueError / FileNotFoundError)
    on bad config: s_high <= s_low, no gated species, or no resolvable field.
    """
    s_low = float(cfg.get("movement.salinity.gate.s.low", "3.0"))
    s_high = float(cfg.get("movement.salinity.gate.s.high", "6.0"))
    if cfg.get("movement.salinity.gate.enabled", "false").lower() != "true":
        return False, None, s_low, s_high, None

    if s_high <= s_low:
        raise ValueError(f"movement.salinity.gate.s.high ({s_high}) must be > s.low ({s_low})")

    mask = np.zeros(n_species, dtype=np.bool_)
    for sp in range(n_species):
        if cfg.get(f"movement.salinity.gate.species.enabled.sp{sp}", "false").lower() == "true":
            mask[sp] = True
    if not mask.any():
        raise ValueError(
            "salinity gate enabled but no species enabled "
            "(movement.salinity.gate.species.enabled.sp{idx})."
        )

    const_str = cfg.get("movement.salinity.field.constant", "")
    file_str = cfg.get("movement.salinity.field.file", "")
    if const_str:
        field = PhysicalData.from_constant(float(const_str))
    elif file_str:
        path = _require_file(file_str, _cfg_dir(cfg), "movement.salinity.field.file")
        varname = cfg.get("movement.salinity.field.varname", "so")
        field = PhysicalData.from_netcdf(path, varname=varname)
    else:
        raise ValueError(
            "salinity gate enabled but no salinity field "
            "(set movement.salinity.field.constant or .file)."
        )
    return True, mask, s_low, s_high, field


def _load_recruitment_ceiling(
    cfg: dict[str, str],
    n_species: int,
    n_dt: int,
    spawning_season: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64] | None, NDArray[np.bool_] | None]:
    """Load the unfished-level recruitment ceiling (spec 2026-07-03).

    Returns (ceiling_by_season, enabled_mask). ceiling_by_season has shape
    (n_cols, n_species) where n_cols is the model's within-year season count;
    enabled_mask has shape (n_species,). Both are None when the master switch is
    off. Fail-fast (ValueError / FileNotFoundError) on invalid configuration.
    """
    if cfg.get("reproduction.recruitment.ceiling.enabled", "false").lower() != "true":
        return None, None

    n_cols = spawning_season.shape[1] if spawning_season is not None else n_dt

    file_key = cfg.get("reproduction.recruitment.ceiling.series.file", "")
    if not file_key:
        raise ValueError(
            "Recruitment ceiling enabled but reproduction.recruitment.ceiling.series.file is empty."
        )
    path = _require_file(file_key, _cfg_dir(cfg), "reproduction.recruitment.ceiling.series.file")
    df = pd.read_csv(path)
    if "season_idx" not in df.columns:
        raise ValueError(f"Recruitment ceiling {path} missing 'season_idx' column.")
    seasons = df["season_idx"].to_numpy()
    if not np.array_equal(seasons, np.arange(n_cols)):
        raise ValueError(
            f"Recruitment ceiling {path} season_idx must be 0..{n_cols - 1} "
            f"contiguous (model has {n_cols} season columns), got {seasons.tolist()}."
        )

    ceiling = np.full((n_cols, n_species), np.inf, dtype=np.float64)
    for sp in range(n_species):
        col = f"ceiling_sp{sp}"
        if col in df.columns:
            ceiling[:, sp] = df[col].to_numpy(dtype=np.float64)
    # Disabled species keep the inf sentinel (harmless); only real values are
    # checked. NaN is caught here; the finite-column check for ENABLED species
    # is the last loop below.
    finite = np.isfinite(ceiling)
    if np.any(ceiling[finite] < 0):
        raise ValueError(f"Recruitment ceiling {path} has negative values.")
    if np.any(np.isnan(ceiling)):
        raise ValueError(f"Recruitment ceiling {path} has NaN values.")

    enabled = np.zeros(n_species, dtype=np.bool_)
    for sp in range(n_species):
        key = f"reproduction.recruitment.ceiling.species.enabled.sp{sp}"
        if cfg.get(key, "false").lower() == "true":
            enabled[sp] = True
    if not enabled.any():
        raise ValueError(
            "Recruitment ceiling enabled but no species enabled "
            "(reproduction.recruitment.ceiling.species.enabled.sp{idx})."
        )
    # An enabled species must have a finite ceiling column.
    for sp in np.where(enabled)[0]:
        if not np.all(np.isfinite(ceiling[:, sp])):
            raise ValueError(
                f"Recruitment ceiling enabled for sp{sp} but no ceiling_sp{sp} column in {path}."
            )
    return ceiling, enabled


def _load_additional_mortality_by_dt(
    cfg: dict[str, str], n_species: int
) -> list[NDArray[np.float64] | None] | None:
    """Load time-varying additional mortality CSV (BY_DT scenario)."""
    return _load_per_species_timeseries(
        cfg, n_species, "mortality.additional.rate.bytdt.file.sp{i}", "additional_mortality_by_dt"
    )


def _load_additional_mortality_spatial(
    cfg: dict[str, str], n_species: int
) -> list[NDArray[np.float64] | None] | None:
    """Load spatial additional mortality distribution maps.

    Returns a list of 2D arrays (one per species), or None if no files found.
    """
    result: list[NDArray[np.float64] | None] = [None] * n_species
    found_any = False

    for i in range(n_species):
        file_key = cfg.get(f"mortality.additional.spatial.distrib.file.sp{i}", "")
        if not file_key:
            continue
        # Non-empty key: file must exist. (v3 C-6)
        path = _require_file(
            file_key, _cfg_dir(cfg), f"mortality.additional.spatial.distrib.file.sp{i}"
        )
        result[i] = _load_spatial_csv(path)
        found_any = True

    return result if found_any else None


def _load_additional_mortality_by_dt_by_class(
    cfg: dict[str, str], n_species: int, n_dt_per_year: int, n_dt_simu: int
) -> list | None:
    """Load by-dt-by-class additional mortality from CSV.

    For byAge: Java converts class thresholds from years to time steps
    (threshold * nStepYear). For bySize: thresholds are in cm, used as-is.
    """
    from osmose.engine.timeseries import ByClassTimeSeries

    result: list = [None] * n_species
    found = False
    for i in range(n_species):
        for variant in ["byDt.byAge", "byDt.bySize"]:
            key = f"mortality.additional.rate.{variant}.file.sp{i}"
            if key in cfg:
                path = _require_file(cfg[key], _cfg_dir(cfg), key)
                ts = ByClassTimeSeries.from_csv(path, n_dt_per_year, n_dt_simu)
                # Java converts age thresholds from years to time steps
                if "byAge" in variant:
                    ts.classes = np.round(ts.classes * n_dt_per_year).astype(np.float64)
                result[i] = ts
                found = True
                break
    return result if found else None


def _detect_larva_by_dt_key(cfg: dict[str, str], species_idx: int) -> str | None:
    """Detect larval by-dt file, supporting both bytDt (Java typo) and byDt."""
    for variant in ["bytDt", "byDt"]:
        key = f"mortality.additional.larva.rate.{variant}.file.sp{species_idx}"
        if key in cfg:
            return cfg[key]
    return None


def _load_larva_mortality_by_dt(
    cfg: dict[str, str], n_species: int, n_dt_per_year: int, n_dt_simu: int
) -> list | None:
    """Load time-varying larval mortality CSV (ByDtLarvaMortality)."""
    from osmose.engine.timeseries import SingleTimeSeries

    result: list = [None] * n_species
    found = False
    for i in range(n_species):
        path_str = _detect_larva_by_dt_key(cfg, i)
        if path_str:
            path = _require_file(path_str, _cfg_dir(cfg), f"larva.rate.byDt.sp{i}")
            result[i] = SingleTimeSeries.from_csv(path, n_dt_per_year, n_dt_simu)
            found = True
    return result if found else None


# ---------------------------------------------------------------------------
# SP-1: Fishing scenario detection (matches Java FishingMortality.Scenario enum)
# ---------------------------------------------------------------------------

_FISHING_SCENARIOS = [
    ("rate_annual", "mortality.fishing.rate.sp"),
    ("rate_by_year", "mortality.fishing.rate.byYear.file.sp"),
    ("rate_by_dt_by_class", "mortality.fishing.rate.byDt.byAge.file.sp"),
    ("rate_by_dt_by_class", "mortality.fishing.rate.byDt.bySize.file.sp"),
    ("catches_annual", "mortality.fishing.catches.sp"),
    ("catches_by_year", "mortality.fishing.catches.byYear.file.sp"),
    ("catches_by_dt_by_class", "mortality.fishing.catches.byDt.byAge.file.sp"),
    ("catches_by_dt_by_class", "mortality.fishing.catches.byDt.bySize.file.sp"),
]


def detect_fishing_scenario(config: dict[str, str], species_idx: int) -> str | None:
    """Detect fishing scenario for a species from config keys.

    Matches Java FishingMortality.findScenario(). Returns scenario name or None.
    """
    for scenario_name, key_prefix in _FISHING_SCENARIOS:
        if f"{key_prefix}{species_idx}" in config:
            return scenario_name
    return None


def _validate_trait_declarations(cfg: dict[str, str], n_sp: int) -> None:
    """Each declared `evolution.trait.<name>.target` must have a per-species
    mean for every species where variance is nonzero."""
    import re

    trait_names: set[str] = set()
    for key in cfg:
        m = re.match(r"evolution\.trait\.(\w+)\.target", key)
        if m:
            trait_names.add(m.group(1))
    for name in trait_names:
        for i in range(n_sp):
            var_key = f"evolution.trait.{name}.var.sp{i}"
            mean_key = f"evolution.trait.{name}.mean.sp{i}"
            var_str = cfg.get(var_key, "0.0")
            try:
                var = float(var_str)
            except ValueError:
                raise ValueError(f"{var_key}: not a number ({var_str!r})")
            if var > 0.0 and mean_key not in cfg:
                raise ValueError(
                    f"{mean_key} missing: trait '{name}' declares nonzero variance "
                    f"on species {i} but no mean is specified"
                )


@dataclass
class EngineConfig:
    """Typed engine configuration extracted from a flat OSMOSE config dict."""

    n_species: int
    n_dt_per_year: int
    n_year: int
    n_steps: int
    n_schools: NDArray[np.int32]
    species_names: list[str]

    # Background species
    n_background: int
    background_file_indices: list[int]
    all_species_names: list[str]

    linf: NDArray[np.float64]
    k: NDArray[np.float64]
    t0: NDArray[np.float64]
    egg_size: NDArray[np.float64]
    condition_factor: NDArray[np.float64]
    allometric_power: NDArray[np.float64]
    vb_threshold_age: NDArray[np.float64]
    lifespan_dt: NDArray[np.int32]

    mortality_subdt: int
    ingestion_rate: NDArray[np.float64]
    critical_success_rate: NDArray[np.float64]

    # Growth
    delta_lmax_factor: NDArray[np.float64]  # max growth scaling factor (default 2.0)

    # Natural mortality
    additional_mortality_rate: NDArray[np.float64]  # annual additional mortality rate per species
    additional_mortality_by_dt: list[NDArray[np.float64] | None] | None  # BY_DT per-step rates
    additional_mortality_by_dt_by_class: list | None  # BY_DT by-class rates (ByClassTimeSeries)
    additional_mortality_spatial: list[NDArray[np.float64] | None] | None  # spatial multiplier maps

    # Reproduction
    sex_ratio: NDArray[np.float64]  # fraction female per species
    relative_fecundity: NDArray[np.float64]  # eggs per gram of mature female
    maturity_size: NDArray[np.float64]  # length at maturity (cm)
    seeding_biomass: NDArray[np.float64]  # initial biomass for seeding (tonnes)
    seeding_max_step: NDArray[np.int32]  # max step for seeding (default: lifespan_dt)
    larva_mortality_rate: NDArray[np.float64]  # additional mortality for eggs/larvae
    larva_mortality_by_dt: list | None  # time-varying larval mortality (SingleTimeSeries per sp)
    # Stock-recruitment (post-parity divergence; Java has no equivalent)
    recruitment_type: list[
        str
    ]  # one of {"none","beverton_holt","ricker","hockey_stick","shepherd"} per species
    recruitment_ssb_half: NDArray[np.float64]  # tonnes; ignored when type=="none"
    shepherd_beta: NDArray[np.float64]  # per-species Shepherd exponent; 1.0 ≡ B-H
    # Predator functional response (post-parity; opt-in, default code 1 ≡ existing)
    fr_shape: NDArray[np.int32]  # per-species Holling form code: 1=type-I, 2=type-II, 3=type-III
    fr_halfsat: NDArray[np.float64]  # per-species ration-relative half-saturation K (type2/3 only)

    # Predation — 2D arrays of shape (n_total, max_stages)
    size_ratio_min: NDArray[np.float64]  # min pred/prey ratio per species per stage
    size_ratio_max: NDArray[np.float64]  # max pred/prey ratio per species per stage

    # Feeding stages
    feeding_stage_thresholds: list  # per-species list of threshold floats
    feeding_stage_metric: list[str]  # per-species metric name ("size"/"age"/"weight"/"tl")
    n_feeding_stages: NDArray[np.int32]  # number of feeding stages per species

    # Starvation
    starvation_rate_max: NDArray[np.float64]  # max starvation mortality rate

    # Fishing
    fishing_enabled: bool  # global fishing toggle
    fishing_rate: NDArray[np.float64]  # annual fishing mortality rate per species
    fishing_selectivity_l50: NDArray[np.float64]  # length at 50% selectivity

    # Fishing — fisheries-based selectivity
    fishing_selectivity_a50: NDArray[np.float64]  # age at 50% selectivity (years), NaN = unused
    fishing_selectivity_type: NDArray[np.int32]  # 0=age, 1=sigmoidal size, -1=none
    fishing_selectivity_slope: NDArray[np.float64]  # sigmoid slope (type 1 only)

    # Fishing seasonality: (n_species, n_dt_per_year) normalized weights, or None
    fishing_seasonality: NDArray[np.float64] | None

    # Fishing rate by year: per-species array of annual rates, or None
    fishing_rate_by_year: list[NDArray[np.float64] | None] | None

    # Fishing rate by dt and age/size class: per-species ByClassTimeSeries, or None
    fishing_rate_by_dt_by_class: list | None  # list[ByClassTimeSeries | None] | None

    # Catch-based fishing: annual target catches (tonnes) per species, or None
    fishing_catches: NDArray[np.float64] | None
    # Catch by year: per-species list of annual catch values, or None
    fishing_catches_by_year: list | None  # list[NDArray[np.float64] | None] | None
    # Catch seasonality: same as fishing_seasonality
    fishing_catches_season: NDArray[np.float64] | None

    # L75: length at 75% selectivity (for sigmoid/Gaussian/log-normal)
    fishing_selectivity_l75: NDArray[np.float64]

    # Marine Protected Areas
    mpa_zones: list[MPAZone] | None

    # Fishery discards: per-species discard fraction (0-1), or None
    fishing_discard_rate: NDArray[np.float64] | None

    # Predation accessibility
    accessibility_matrix: NDArray[np.float64] | None  # (n_pred, n_prey) or None
    stage_accessibility: AccessibilityMatrix | None  # stage-indexed accessibility, or None

    # Dynamic accessibility — density-dependent scaling
    dynamic_accessibility_enabled: bool
    dynamic_accessibility_exponent: float
    dynamic_accessibility_floor: float

    # Reproduction
    spawning_season: NDArray[np.float64] | None  # (n_species, n_dt_per_year) or None

    # Reproductive-volume recruitment gate (all None when disabled)
    rv_gate_factor_by_index: NDArray[np.float64] | None  # (n_years,), mode already applied
    rv_gate_enabled: NDArray[np.bool_] | None  # (n_species,) per-species enable mask
    rv_gate_offset: int  # start_year - first_year (see _load_rv_gate)

    # Salinity-gated occupancy (prototype spike; feature inert when disabled)
    salinity_gate_enabled: bool
    salinity_gate_species: NDArray[np.bool_] | None
    salinity_gate_s_low: float
    salinity_gate_s_high: float
    salinity_field: PhysicalData | None

    # Unfished-level recruitment ceiling (both None when disabled)
    recruitment_ceiling_by_season: NDArray[np.float64] | None  # (n_cols, n_species)
    recruitment_ceiling_enabled: NDArray[np.bool_] | None  # (n_species,) enable mask

    # Percid thermal recruitment gate (all None/0 when disabled)
    thermal_gate_factor_by_index: NDArray[np.float64] | None  # (n_years, n_species)
    thermal_gate_enabled: NDArray[np.bool_] | None  # (n_species,) per-species enable mask
    thermal_gate_offset: int  # start_year - first_year

    # Recruitment depensation / Allee gate (all None when disabled)
    depensation_gate_enabled: NDArray[np.bool_] | None  # (n_species,) per-species enable mask
    depensation_s50: NDArray[np.float64] | None  # (n_species,) SSB at which A=0.5, tonnes
    depensation_theta: NDArray[np.float64] | None  # (n_species,) Allee steepness, >= 1

    # Movement
    movement_method: list[str]
    random_walk_range: NDArray[np.int32]
    out_mortality_rate: NDArray[np.float64]

    # Maturity age in timesteps (0 = no age threshold, only size-based)
    maturity_age_dt: NDArray[np.int32]

    # Maximum length cap (may differ from linf)
    lmax: NDArray[np.float64]

    # Spatial fishing distribution maps: one 2D grid per species, or None
    fishing_spatial_maps: list  # list[NDArray[np.float64] | None]

    # Egg weight override: if species.egg.weight.sp{i} is set, use it instead of allometry
    egg_weight_override: NDArray[np.float64] | None  # shape (n_species,) or None

    # Output cutoff age: exclude schools younger than this from biomass/abundance output
    output_cutoff_age: NDArray[np.float64] | None  # shape (n_total,) or None

    # Output recording frequency (steps between output records)
    output_record_frequency: int

    # Diet composition output
    diet_output_enabled: bool

    # Initial state output (step -1)
    output_step0_include: bool

    # RNG seeding flags
    movement_seed_fixed: bool  # per-species independent RNG for movement
    mortality_seed_fixed: bool  # per-species independent RNG for mortality
    java_compat_rng: bool  # use Java's XorshiftRNG for bit-exact parity

    # Random distribution patch constraint: per-species ncell values, or None
    random_distribution_ncell: NDArray[np.int32] | None

    # Growth class per species: "VB" or "GOMPERTZ"
    growth_class: list[str]

    # Raw config dict for subsystems that need unparsed access (e.g. ResourceState)
    raw_config: dict[str, str]

    # Pre-computed set of species IDs with VB growth (for fast mask creation)
    vb_species_ids: frozenset[int] = frozenset()

    # Movement map coverage enforcement
    movement_strict_coverage: bool = False

    # Gompertz growth parameters (None when no GOMPERTZ species)
    gompertz_ke: NDArray[np.float64] | None = None
    gompertz_lstart: NDArray[np.float64] | None = None
    gompertz_kg: NDArray[np.float64] | None = None
    gompertz_tg: NDArray[np.float64] | None = None
    gompertz_linf: NDArray[np.float64] | None = None
    gompertz_thr_age_exp_dt: NDArray[np.int32] | None = None
    gompertz_thr_age_gom_dt: NDArray[np.int32] | None = None

    # Bioenergetic model toggle
    bioen_enabled: bool = False

    # Bioenergetic global flags
    bioen_phit_enabled: bool = True
    bioen_fo2_enabled: bool = True

    # Bioenergetic per-species parameters (None when bioen disabled).
    # Coupling invariant: when bioen_enabled=True, from_dict() populates ALL 18
    # bioen_* fields with per-species arrays using hard-coded defaults where config
    # keys are absent.  No bioen_* field can be None after from_dict() returns with
    # bioen_enabled=True.  __post_init__ does not need a redundant None-check because
    # from_dict() is the only supported construction path (direct dataclass
    # instantiation with None bioen arrays + bioen_enabled=True is unsupported).
    bioen_beta: NDArray[np.float64] | None = None  # allometric exponent
    bioen_zlayer: NDArray[np.int32] | None = None  # depth layer index
    bioen_assimilation: NDArray[np.float64] | None = None  # assimilation efficiency
    bioen_c_m: NDArray[np.float64] | None = None  # maintenance coefficient
    bioen_eta: NDArray[np.float64] | None = None  # energy density ratio
    bioen_r: NDArray[np.float64] | None = None  # reproductive allocation
    bioen_m0: NDArray[np.float64] | None = None  # LMRN intercept
    bioen_m1: NDArray[np.float64] | None = None  # LMRN slope
    bioen_e_mobi: NDArray[np.float64] | None = None  # Johnson e_M (eV)
    bioen_e_d: NDArray[np.float64] | None = None  # Johnson e_D (eV)
    bioen_tp: NDArray[np.float64] | None = None  # peak temperature (°C)
    bioen_e_maint: NDArray[np.float64] | None = None  # Arrhenius maintenance energy (eV)
    bioen_o2_c1: NDArray[np.float64] | None = None  # O2 dose-response asymptote
    bioen_o2_c2: NDArray[np.float64] | None = None  # O2 half-saturation
    bioen_i_max: NDArray[np.float64] | None = None  # max ingestion rate (bioen)
    bioen_theta: NDArray[np.float64] | None = None  # larvae ingestion multiplier
    bioen_c_rate: NDArray[np.float64] | None = None  # larvae correction coefficient
    bioen_k_for: NDArray[np.float64] | None = None  # foraging mortality

    # Foraging mortality (bioen only)
    foraging_k1_for: NDArray[np.float64] | None = None  # genetic mode base
    foraging_k2_for: NDArray[np.float64] | None = None  # genetic mode exponent
    foraging_I_max: NDArray[np.float64] | None = None  # reference I_max

    # Ev-OSMOSE genetics toggle
    genetics_enabled: bool = False
    genetics_transmission_year: int = 0  # first year seeding is active (0 = always normal)
    genetics_n_neutral: int = 0  # number of neutral loci (0 = disabled)
    genetics_n_neutral_val: int = 50  # number of allele values per neutral locus

    # DSVM fleet dynamics toggle
    economics_enabled: bool = False

    # Distribution output flags
    output_biomass_byage: bool = False
    output_biomass_bysize: bool = False
    output_abundance_byage: bool = False
    output_abundance_bysize: bool = False
    output_meantl: bool = False
    output_yield_abundance: bool = False
    output_mean_size: bool = False
    output_yield_abundance_netcdf: bool = False
    output_mean_size_netcdf: bool = False
    output_ssb: bool = False
    output_ssb_netcdf: bool = False
    output_size_min: float = 0.0
    output_size_max: float = 205.0
    output_size_incr: float = 10.0

    # Bioenergetic output flags (default False; meanEnet is always written when bioen enabled)
    output_bioen_ingest: bool = False
    output_bioen_maint: bool = False
    output_bioen_rho: bool = False
    output_bioen_sizeinf: bool = False

    # NetCDF output flags (gated per-variable; no master switch)
    output_biomass_netcdf: bool = False
    output_abundance_netcdf: bool = False
    output_yield_biomass_netcdf: bool = False
    output_biomass_byage_netcdf: bool = False
    output_abundance_byage_netcdf: bool = False
    output_biomass_bysize_netcdf: bool = False
    output_abundance_bysize_netcdf: bool = False
    output_mortality_netcdf: bool = False

    # Spatial output flags (master gate + per-variant)
    output_spatial_enabled: bool = False
    output_spatial_biomass: bool = False
    output_spatial_abundance: bool = False
    output_spatial_yield_biomass: bool = False

    # Spatial reproductive-volume egg-survival field (Task 2/4; consumed in larva_mortality)
    rv_spatial_field: PhysicalData | None = None
    rv_spatial_enabled: NDArray[np.bool_] | None = None

    @cached_property
    def movement_is_random(self) -> NDArray[np.bool_]:
        """Per-species mask: True where the movement method is ``"random"``.

        Length ``len(movement_method)`` (focal + background; background entries are
        ``"none"`` and never indexed during ``movement()``). Precomputed once so the
        per-timestep movement hot path can fancy-index by the school ``species_id``
        array instead of a Python comprehension over every school (A1/A2 template).
        """
        return np.array([m == "random" for m in self.movement_method], dtype=np.bool_)

    @cached_property
    def movement_is_maps(self) -> NDArray[np.bool_]:
        """Per-species mask: True where the movement method is ``"maps"``. See
        :attr:`movement_is_random`."""
        return np.array([m == "maps" for m in self.movement_method], dtype=np.bool_)

    def __post_init__(self) -> None:
        """Validate invariants after construction.

        Bioen coupling note (I-2): when bioen_enabled=True all 18 bioen_* per-species
        arrays are guaranteed non-None by from_dict(), which assigns defaults for every
        missing key.  A runtime None-check in __post_init__ is therefore a no-op and is
        intentionally omitted; the coupling is implicitly enforced by the parser.
        """
        n_total = self.n_species + self.n_background

        # Check n_steps consistency
        expected_steps = self.n_dt_per_year * self.n_year
        if self.n_steps != expected_steps:
            raise ValueError(
                f"n_steps ({self.n_steps}) != n_dt_per_year ({self.n_dt_per_year}) "
                f"* n_year ({self.n_year}) = {expected_steps}"
            )

        # Check per-species array lengths
        per_species_arrays = {
            "linf": self.linf,
            "k": self.k,
            "t0": self.t0,
            "egg_size": self.egg_size,
            "condition_factor": self.condition_factor,
            "allometric_power": self.allometric_power,
            "lifespan_dt": self.lifespan_dt,
            "ingestion_rate": self.ingestion_rate,
            "critical_success_rate": self.critical_success_rate,
            "additional_mortality_rate": self.additional_mortality_rate,
            "starvation_rate_max": self.starvation_rate_max,
            "shepherd_beta": self.shepherd_beta,
            "fr_shape": self.fr_shape,
            "fr_halfsat": self.fr_halfsat,
        }
        for name, arr in per_species_arrays.items():
            if hasattr(arr, "__len__") and len(arr) != n_total:
                raise ValueError(
                    f"{name} has length {len(arr)}, expected {n_total} "
                    f"(n_species={self.n_species} + n_background={self.n_background})"
                )

        # Check biological positivity constraints (focal species only)
        for name, arr in [("linf", self.linf), ("k", self.k)]:
            if hasattr(arr, "__len__"):
                for i in range(self.n_species):
                    if arr[i] <= 0:
                        raise ValueError(
                            f"{name}[{i}] = {arr[i]}, must be positive for "
                            f"focal species '{self.species_names[i]}'"
                        )

        self._warn_unsupported_mortality_features()

    def _warn_unsupported_mortality_features(self) -> None:
        """Warn loudly about mortality features that are PARSED but NOT applied.

        The interleaved per-cell mortality loop applies fishing as a flat F-rate and
        additional mortality as flat/by-step rates only. Catch-based fishing, by-age/size
        fishing rates, and by-age/size additional mortality are parsed into the config but
        never reach the loop, so a config relying on them would silently get wrong results.
        Surface that as a clear warning rather than a silent divergence. (Wiring these is
        deferred until a config needs them — so the work can be parity-validated as part of
        delivering it; see docs/superpowers/plans/2026-06-20-deep-review-remediation.md PR-1.)
        """

        def _sp(i: int) -> str:
            return self.species_names[i] if i < len(self.species_names) else f"sp{i}"

        def _warn_once(msg: str) -> None:
            # Throttle: EngineConfig is rebuilt per candidate during calibration.
            if msg not in _WARNED_UNSUPPORTED_MORTALITY:
                _WARNED_UNSUPPORTED_MORTALITY.add(msg)
                _log.warning("%s", msg)

        fc = self.fishing_catches
        if fc is not None:
            affected = [_sp(i) for i in range(len(fc)) if np.isfinite(fc[i]) and fc[i] > 0]
            if affected:
                _warn_once(
                    f"Catch-based fishing (mortality.fishing.catches.*) is configured for "
                    f"{', '.join(affected)} but the Python engine applies fishing as an F-rate "
                    "only — catch-based fishing is NOT applied. Use mortality.fishing.rate.* "
                    "instead, or run the Java engine."
                )

        abc = self.additional_mortality_by_dt_by_class
        if abc is not None:
            affected = [_sp(i) for i, ts in enumerate(abc) if ts is not None]
            if affected:
                _warn_once(
                    "Per-age/size additional mortality (mortality.additional.rate.byDt."
                    f"byAge/bySize.*) is configured for {', '.join(affected)} but the Python "
                    "engine applies only flat and by-step additional mortality — the by-class "
                    "rates are NOT applied. Use mortality.additional.rate(.byDt).* instead, or "
                    "run the Java engine."
                )

        frc = self.fishing_rate_by_dt_by_class
        if frc is not None:
            affected = [_sp(i) for i, ts in enumerate(frc) if ts is not None]
            if affected:
                _warn_once(
                    "Per-age/size fishing rate (mortality.fishing.rate.byDt.byAge/bySize.*) is "
                    f"configured for {', '.join(affected)} but the Python engine applies a flat "
                    "F-rate only — the by-class fishing rates are NOT applied. Use "
                    "mortality.fishing.rate.* instead, or run the Java engine."
                )

        sel = self.fishing_selectivity_type
        affected = [_sp(i) for i in range(self.n_species) if sel[i] in (2, 3)]
        if affected:
            _warn_once(
                "Fishing selectivity type 2 (Gaussian) / 3 (log-normal) is configured for "
                f"{', '.join(affected)} but the Python engine's interleaved mortality loop "
                "applies only knife-edge (type 0) and logistic (type 1) selectivity — types "
                "2/3 are silently treated as length knife-edge. Use selectivity type 0 or 1, "
                "or run the Java engine."
            )

    @classmethod
    def from_dict(cls, cfg: dict[str, str]) -> EngineConfig:
        from osmose.engine.config_validation import validate as _validate_cfg

        _raw_mode = cfg.get("validation.strict.enabled", "off")
        _mode = _raw_mode if isinstance(_raw_mode, str) else "off"
        _validate_cfg(cfg, _mode)

        from osmose.config.aliases import canonicalize_config

        cfg, _deprecated = canonicalize_config(cfg)

        # config_dir is extracted from cfg by _cfg_dir() at each _resolve_file call
        n_sp = int(_get(cfg, "simulation.nspecies"))
        n_dt = int(_get(cfg, "simulation.time.ndtperyear"))
        n_yr = int(_get(cfg, "simulation.time.nyear"))
        lifespan_years = _species_float(cfg, "species.lifespan.sp{i}", n_sp)

        # Fishing rate: try fisheries-based (v4), then per-species patterns
        fisheries_enabled = (
            cfg.get("module.multispecies.fisheries.enabled", "false").lower() == "true"
        )
        n_fisheries = int(cfg.get("simulation.nfisheries", "0"))

        # Build species names early for fisheries parsing
        _focal_names = _species_str(cfg, "species.name.sp{i}", n_sp)

        if fisheries_enabled and n_fisheries > 0:
            (
                fishing,
                focal_fishing_a50,
                focal_fishing_sel_type,
                focal_fishing_l50_fsh,
                focal_fishing_slope,
            ) = _parse_fisheries(cfg, _focal_names, n_sp)
        else:
            # Legacy per-species rate
            fishing = _species_float_optional(
                cfg, "mortality.fishing.rate.sp{i}", n_sp, default=0.0
            )
            if fishing.sum() == 0:
                fishing = _species_float_optional(cfg, "fishing.rate.sp{i}", n_sp, default=0.0)
            focal_fishing_a50 = np.full(n_sp, np.nan, dtype=np.float64)
            focal_fishing_sel_type = np.full(n_sp, -1, dtype=np.int32)
            focal_fishing_l50_fsh = np.zeros(n_sp, dtype=np.float64)
            focal_fishing_slope = np.zeros(n_sp, dtype=np.float64)

        # Parse background species
        background_list: list[BackgroundSpeciesInfo] = parse_background_species(
            cfg, n_focal=n_sp, n_dt_per_year=n_dt
        )
        n_bkg = len(background_list)

        # Build focal-only arrays first
        focal_species_names = _focal_names
        _growth = _parse_growth_params(cfg, n_sp, n_dt, lifespan_years)
        focal_linf = _growth["focal_linf"]
        focal_k = _growth["focal_k"]
        focal_t0 = _growth["focal_t0"]
        focal_egg_size = _growth["focal_egg_size"]
        focal_condition_factor = _growth["focal_condition_factor"]
        focal_allometric_power = _growth["focal_allometric_power"]
        focal_vb_threshold_age = _growth["focal_vb_threshold_age"]
        focal_lifespan_dt = _growth["focal_lifespan_dt"]
        focal_delta_lmax_factor = _growth["focal_delta_lmax_factor"]
        focal_additional_mortality_rate = _growth["focal_additional_mortality_rate"]
        focal_lmax = _growth["focal_lmax"]
        _repro = _parse_reproduction_params(cfg, n_sp, n_dt, lifespan_years)
        focal_sex_ratio = _repro["focal_sex_ratio"]
        focal_relative_fecundity = _repro["focal_relative_fecundity"]
        focal_maturity_size = _repro["focal_maturity_size"]
        focal_seeding_biomass = _repro["focal_seeding_biomass"]
        focal_seeding_max_step = _repro["focal_seeding_max_step"]
        focal_larva_mortality_rate = _repro["focal_larva_mortality_rate"]
        focal_maturity_age_dt = _repro["focal_maturity_age_dt"]
        focal_recruitment_type = _repro["focal_recruitment_type"]
        focal_recruitment_ssb_half = _repro["focal_recruitment_ssb_half"]
        focal_recruitment_shepherd_beta = _repro["focal_recruitment_shepherd_beta"]
        focal_fr_shape = _repro["focal_fr_shape"]
        focal_fr_halfsat = _repro["focal_fr_halfsat"]
        # Fishing spatial distribution maps
        focal_fishing_spatial_maps: list[np.ndarray | None] = []
        # Try shared fisheries map first (v4)
        shared_fishing_map_file = cfg.get("fisheries.movement.file.map0", "")
        shared_fishing_map: np.ndarray | None = None
        if shared_fishing_map_file:
            shared_path = _require_file(
                shared_fishing_map_file, _cfg_dir(cfg), "fisheries.movement.file.map0"
            )
            shared_fishing_map = _load_spatial_csv(shared_path)
        for i in range(n_sp):
            sp_map_file = cfg.get(f"mortality.fishing.spatial.distrib.file.sp{i}", "")
            if sp_map_file:
                # User explicitly set a per-species map — missing file is a
                # config error, not a silent fallback to the shared map. (v3 C-4)
                sp_path = _require_file(
                    sp_map_file, _cfg_dir(cfg), f"mortality.fishing.spatial.distrib.file.sp{i}"
                )
                focal_fishing_spatial_maps.append(_load_spatial_csv(sp_path))
            else:
                focal_fishing_spatial_maps.append(shared_fishing_map)

        _pred = _parse_predation_params(cfg, n_sp, n_dt, background_list, focal_fishing_l50_fsh)
        focal_ingestion_rate = _pred["focal_ingestion_rate"]
        focal_critical_success_rate = _pred["focal_critical_success_rate"]
        focal_starvation_rate_max = _pred["focal_starvation_rate_max"]
        focal_fishing_selectivity_l50 = _pred["focal_fishing_selectivity_l50"]
        focal_movement_method = _pred["focal_movement_method"]
        focal_random_walk_range = _pred["focal_random_walk_range"]
        focal_out_mortality_rate = _pred["focal_out_mortality_rate"]
        focal_n_schools = _pred["focal_n_schools"]
        all_thresholds = _pred["all_thresholds"]
        all_metrics = _pred["all_metrics"]
        n_feeding_stages = _pred["n_feeding_stages"]
        size_ratio_min_2d = _pred["size_ratio_min_2d"]
        size_ratio_max_2d = _pred["size_ratio_max_2d"]

        # Merge focal arrays with background species defaults
        _focal = {
            "focal_linf": focal_linf,
            "focal_k": focal_k,
            "focal_t0": focal_t0,
            "focal_egg_size": focal_egg_size,
            "focal_condition_factor": focal_condition_factor,
            "focal_allometric_power": focal_allometric_power,
            "focal_vb_threshold_age": focal_vb_threshold_age,
            "focal_lifespan_dt": focal_lifespan_dt,
            "focal_delta_lmax_factor": focal_delta_lmax_factor,
            "focal_additional_mortality_rate": focal_additional_mortality_rate,
            "focal_lmax": focal_lmax,
            "focal_ingestion_rate": focal_ingestion_rate,
            "focal_critical_success_rate": focal_critical_success_rate,
            "focal_sex_ratio": focal_sex_ratio,
            "focal_relative_fecundity": focal_relative_fecundity,
            "focal_maturity_size": focal_maturity_size,
            "focal_seeding_biomass": focal_seeding_biomass,
            "focal_seeding_max_step": focal_seeding_max_step,
            "focal_larva_mortality_rate": focal_larva_mortality_rate,
            "focal_maturity_age_dt": focal_maturity_age_dt,
            "focal_recruitment_type": focal_recruitment_type,
            "focal_recruitment_ssb_half": focal_recruitment_ssb_half,
            "focal_recruitment_shepherd_beta": focal_recruitment_shepherd_beta,
            "focal_fr_shape": focal_fr_shape,
            "focal_fr_halfsat": focal_fr_halfsat,
            "focal_starvation_rate_max": focal_starvation_rate_max,
            "focal_fishing_selectivity_l50": focal_fishing_selectivity_l50,
            "focal_fishing_a50": focal_fishing_a50,
            "focal_fishing_sel_type": focal_fishing_sel_type,
            "focal_fishing_slope": focal_fishing_slope,
            "focal_random_walk_range": focal_random_walk_range,
            "focal_out_mortality_rate": focal_out_mortality_rate,
            "focal_n_schools": focal_n_schools,
            "fishing": fishing,
        }
        _merged = _merge_focal_background(
            _focal,
            background_list,
            focal_species_names,
            focal_fishing_spatial_maps,
            focal_movement_method,
        )
        all_species_names = _merged["all_species_names"]
        linf = _merged["linf"]
        k = _merged["k"]
        t0 = _merged["t0"]
        egg_size = _merged["egg_size"]
        condition_factor = _merged["condition_factor"]
        allometric_power = _merged["allometric_power"]
        vb_threshold_age = _merged["vb_threshold_age"]
        lifespan_dt = _merged["lifespan_dt"]
        ingestion_rate = _merged["ingestion_rate"]
        critical_success_rate = _merged["critical_success_rate"]
        delta_lmax_factor = _merged["delta_lmax_factor"]
        additional_mortality_rate = _merged["additional_mortality_rate"]
        sex_ratio = _merged["sex_ratio"]
        relative_fecundity = _merged["relative_fecundity"]
        maturity_size = _merged["maturity_size"]
        seeding_biomass = _merged["seeding_biomass"]
        seeding_max_step = _merged["seeding_max_step"]
        larva_mortality_rate = _merged["larva_mortality_rate"]
        maturity_age_dt = _merged["maturity_age_dt"]
        recruitment_type = _merged["recruitment_type"]
        recruitment_ssb_half = _merged["recruitment_ssb_half"]
        recruitment_shepherd_beta = _merged["recruitment_shepherd_beta"]
        fr_shape = _merged["fr_shape"]
        fr_halfsat = _merged["fr_halfsat"]
        lmax = _merged["lmax"]
        starvation_rate_max = _merged["starvation_rate_max"]
        fishing_rate = _merged["fishing_rate"]
        fishing_selectivity_l50 = _merged["fishing_selectivity_l50"]
        fishing_selectivity_a50 = _merged["fishing_selectivity_a50"]
        fishing_selectivity_type = _merged["fishing_selectivity_type"]
        fishing_selectivity_slope = _merged["fishing_selectivity_slope"]
        movement_method = _merged["movement_method"]
        random_walk_range = _merged["random_walk_range"]
        out_mortality_rate = _merged["out_mortality_rate"]
        n_schools = _merged["n_schools"]
        fishing_spatial_maps = _merged["fishing_spatial_maps"]

        # Egg weight override: use species.egg.weight.sp{i} if provided
        # Java convention: config value is in GRAMS, convert to tonnes (* 1e-6)
        egg_weight_vals = [cfg.get(f"species.egg.weight.sp{i}", "") for i in range(n_sp)]
        if any(v for v in egg_weight_vals):
            egg_weight_override = np.array(
                [float(v) * 1e-6 if v else float("nan") for v in egg_weight_vals],
                dtype=np.float64,
            )
        else:
            egg_weight_override = None

        _output = _parse_output_flags(cfg, n_sp, n_bkg)

        # Phase 2 fishing features
        fishing_seasonality = _load_fishing_seasonality(cfg, n_sp, n_dt, focal_species_names)
        fishing_rate_by_year = _load_fishing_rate_by_year(cfg, n_sp)
        mpa_zones = _parse_mpa_zones(cfg)
        fishing_discard_rate = _load_discard_rates(cfg, focal_species_names, n_sp)

        # SP-1: Rate by dt by class
        from osmose.engine.timeseries import ByClassTimeSeries

        fishing_rate_by_dt_by_class: list | None = None
        _have_dt_class = False
        _dt_class_list: list = [None] * n_sp
        for i in range(n_sp):
            for variant in ["byDt.byAge", "byDt.bySize"]:
                key = f"mortality.fishing.rate.{variant}.file.sp{i}"
                if key in cfg and cfg[key]:
                    ts_path = _resolve_file(cfg[key], _cfg_dir(cfg))
                    if ts_path is not None:
                        n_years = int(cfg.get("simulation.time.nyear", "1"))
                        ts = ByClassTimeSeries.from_csv(ts_path, n_dt, n_dt * n_years)
                        _dt_class_list[i] = ts
                        _have_dt_class = True
                    break
        if _have_dt_class:
            fishing_rate_by_dt_by_class = _dt_class_list

        # SP-1: Catch-based fishing
        fishing_catches: np.ndarray | None = None
        fishing_catches_by_year: list | None = None
        fishing_catches_season: np.ndarray | None = None
        for i in range(n_sp):
            key = f"mortality.fishing.catches.sp{i}"
            if key in cfg:
                if fishing_catches is None:
                    fishing_catches = np.zeros(n_sp, dtype=np.float64)
                fishing_catches[i] = float(cfg[key])
            year_key = f"mortality.fishing.catches.byYear.file.sp{i}"
            if year_key in cfg and cfg[year_key]:
                if fishing_catches_by_year is None:
                    fishing_catches_by_year = [None] * n_sp
                yr_path = _resolve_file(cfg[year_key], _cfg_dir(cfg))
                if yr_path is not None:
                    n_years = int(cfg.get("simulation.time.nyear", "1"))
                    arr = np.loadtxt(yr_path, delimiter=";")
                    if arr.ndim == 0:
                        arr = np.array([float(arr)])
                    fishing_catches_by_year[i] = arr[:n_years]

        # SP-1: L75 selectivity parameter
        fishing_selectivity_l75 = _species_float_optional(
            cfg, "fisheries.selectivity.l75.fsh{i}", n_sp, 0.0
        )
        # Also try per-species key format
        for i in range(n_sp):
            sp_key = f"fishing.selectivity.l75.sp{i}"
            if sp_key in cfg:
                fishing_selectivity_l75[i] = float(cfg[sp_key])

        # Pad fishing_seasonality and fishing_discard_rate for background species.
        # These loaders only know about focal species (n_sp), but work_state.species_id
        # in mortality indexing includes background IDs in [n_sp, n_total). Without
        # padding, fishing_seasonality[sp, step] raises IndexError when background
        # species are present and fishing features are enabled. The zero values mean
        # background species effectively have no fishing, which is also what
        # fishing_rate (concatenated with bkg_zeros_f above) already enforces — so
        # this preserves the "background species are unfished" invariant used
        # throughout the rest of the engine.
        if n_bkg > 0:
            if fishing_seasonality is not None:
                fishing_seasonality = np.concatenate(
                    [fishing_seasonality, np.zeros((n_bkg, n_dt), dtype=np.float64)],
                    axis=0,
                )
            if fishing_discard_rate is not None:
                fishing_discard_rate = np.concatenate(
                    [fishing_discard_rate, np.zeros(n_bkg, dtype=np.float64)]
                )

        # Phase 4: Random distribution patch constraint
        ncell_vals = []
        ncell_found = False
        for i in range(n_sp):
            val = cfg.get(f"movement.distribution.ncell.sp{i}", "")
            if val:
                ncell_vals.append(int(val))
                ncell_found = True
            else:
                ncell_vals.append(0)
        random_distribution_ncell = np.array(ncell_vals, dtype=np.int32) if ncell_found else None

        # Phase 4: Random seed flags
        movement_seed_fixed = cfg.get("movement.randomseed.fixed", "false").lower() == "true"
        mortality_seed_fixed = (
            cfg.get("stochastic.mortality.randomseed.fixed", "false").lower() == "true"
        )
        java_compat_rng = cfg.get("engine.rng.java_compat", "false").lower() == "true"
        movement_strict = cfg.get("movement.map.strict.coverage", "false").lower() == "true"

        # Growth class dispatch: parse classname for each focal species
        growth_class = [
            _GROWTH_MAP.get(
                cfg.get(f"growth.java.classname.sp{i}", "").strip(),
                "VB",
            )
            for i in range(n_sp)
        ]

        # Gompertz parameters: only parsed when at least one species uses GOMPERTZ
        gompertz_ke = gompertz_lstart = gompertz_kg = gompertz_tg = gompertz_linf = None
        gompertz_thr_age_exp_dt = gompertz_thr_age_gom_dt = None
        if "GOMPERTZ" in growth_class:
            gompertz_ke = _species_float_optional(cfg, "growth.exponential.ke.sp{i}", n_sp, 0.0)
            gompertz_lstart = _species_float_optional(
                cfg, "growth.exponential.lstart.sp{i}", n_sp, 0.1
            )
            gompertz_kg = _species_float_optional(cfg, "growth.gompertz.kg.sp{i}", n_sp, 0.0)
            gompertz_tg = _species_float_optional(cfg, "growth.gompertz.tg.sp{i}", n_sp, 0.0)
            gompertz_linf = _species_float_optional(cfg, "growth.gompertz.linf.sp{i}", n_sp, 0.0)
            # M4: Gompertz growth curve is undefined for non-positive linf or kg.
            # Default-to-zero would produce L(t) = 0 * exp(...) = 0 for all ages
            # — silently no growth. Surface as a hard error so the config author
            # fixes the input.
            for i in range(n_sp):
                if growth_class[i] != "GOMPERTZ":
                    continue
                if gompertz_linf[i] <= 0.0:
                    raise ValueError(
                        f"growth.gompertz.linf.sp{i} must be > 0 for GOMPERTZ growth, "
                        f"got {float(gompertz_linf[i])}"
                    )
                if gompertz_kg[i] <= 0.0:
                    raise ValueError(
                        f"growth.gompertz.kg.sp{i} must be > 0 for GOMPERTZ growth, "
                        f"got {float(gompertz_kg[i])}"
                    )
            exp_yrs = _species_float_optional(cfg, "growth.exponential.thr.age.sp{i}", n_sp, 0.0)
            gom_yrs = _species_float_optional(cfg, "growth.gompertz.thr.age.sp{i}", n_sp, 0.0)
            gompertz_thr_age_exp_dt = (exp_yrs * n_dt).astype(np.int32)
            gompertz_thr_age_gom_dt = (gom_yrs * n_dt).astype(np.int32)

        # Bioenergetic parameters: only parsed when module.bioenergetics.enabled=true
        _bioen_enabled = cfg.get("module.bioenergetics.enabled", "false").lower() == "true"
        _bioen_phit_enabled = cfg.get("simulation.bioen.phit.enabled", "true").lower() == "true"
        _bioen_fo2_enabled = cfg.get("simulation.bioen.fo2.enabled", "true").lower() == "true"
        bioen_beta = bioen_zlayer = bioen_assimilation = bioen_c_m = None
        bioen_eta = bioen_r = bioen_m0 = bioen_m1 = None
        bioen_e_mobi = bioen_e_d = bioen_tp = bioen_e_maint = None
        bioen_o2_c1 = bioen_o2_c2 = bioen_i_max = bioen_theta = bioen_c_rate = bioen_k_for = None
        if _bioen_enabled:
            bioen_beta = _species_float_optional(cfg, "species.beta.sp{i}", n_sp, 0.8)
            bioen_zlayer = _species_int_optional(cfg, "species.zlayer.sp{i}", n_sp, 0)
            bioen_assimilation = _species_float_optional(
                cfg, "species.bioen.assimilation.sp{i}", n_sp, 0.7
            )
            bioen_c_m = _species_float_optional(
                cfg, "species.bioen.maint.energy.c_m.sp{i}", n_sp, 0.0
            )
            bioen_eta = _species_float_optional(cfg, "species.maturity.eta.sp{i}", n_sp, 1.0)
            bioen_r = _species_float_optional(cfg, "species.maturity.r.sp{i}", n_sp, 0.0)
            bioen_m0 = _species_float_optional(cfg, "species.maturity.m0.sp{i}", n_sp, 0.0)
            bioen_m1 = _species_float_optional(cfg, "species.maturity.m1.sp{i}", n_sp, 0.0)
            bioen_e_mobi = _species_float_optional(
                cfg, "species.bioen.mobilized.e.mobi.sp{i}", n_sp, 0.65
            )
            bioen_e_d = _species_float_optional(cfg, "species.bioen.mobilized.e.D.sp{i}", n_sp, 1.5)
            bioen_tp = _species_float_optional(cfg, "species.bioen.mobilized.Tp.sp{i}", n_sp, 20.0)
            bioen_e_maint = _species_float_optional(
                cfg, "species.bioen.maint.e.maint.sp{i}", n_sp, 0.65
            )
            bioen_o2_c1 = _species_float_optional(cfg, "species.oxygen.c1.sp{i}", n_sp, 1.0)
            bioen_o2_c2 = _species_float_optional(cfg, "species.oxygen.c2.sp{i}", n_sp, 1.0)
            bioen_i_max = _species_float_optional(
                cfg, "predation.ingestion.rate.max.sp{i}", n_sp, 0.0
            )
            bioen_theta = _species_float_optional(
                cfg, "predation.larval.ingestion.rate.increase.ratio.sp{i}", n_sp, 1.0
            )
            bioen_c_rate = _species_float_optional(cfg, "predation.c.bioen.sp{i}", n_sp, 0.0)
            bioen_k_for = _species_float_optional(
                cfg, "species.bioen.forage.k_for.sp{i}", n_sp, 0.0
            )

        # Foraging mortality parameters (bioen only)
        foraging_k1_for = None
        foraging_k2_for = None
        foraging_I_max = None
        if _bioen_enabled:
            foraging_k1_for = _species_float_optional(
                cfg, "species.bioen.forage.k1_for.sp{i}", n_sp, 0.0
            )
            foraging_k2_for = _species_float_optional(
                cfg, "species.bioen.forage.k2_for.sp{i}", n_sp, 0.0
            )
            foraging_I_max = _species_float_optional(
                cfg, "predation.ingestion.rate.max.sp{i}", n_sp, 0.0
            )

        # Ev-OSMOSE genetics
        genetics_enabled = _enabled(cfg, "module.genetics.enabled")
        genetics_transmission_year = int(cfg.get("evolution.seeding.year", "0"))
        genetics_n_neutral = int(cfg.get("evolution.neutral.nlocus", "0"))
        genetics_n_neutral_val = int(cfg.get("evolution.neutral.nval", "50"))

        if genetics_enabled:
            _validate_trait_declarations(cfg, n_sp)

        # DSVM fleet economics
        # #121: read the real upstream 4.4.0 name (module.bioeconomics.enabled) first; the
        # osmopy-invented simulation.economic.enabled remains a back-compat fallback.
        economics_enabled = _enabled(cfg, "module.bioeconomics.enabled") or _enabled(
            cfg, "simulation.economic.enabled"
        )

        # ── Post-validation: reject non-finite / out-of-range parameters ──
        # Note: fishing_selectivity_a50 uses NaN as "unused" sentinel — skip it.
        for _arr, _label in (
            (fishing_selectivity_slope, "fishing_selectivity_slope"),
            (fishing_selectivity_l50, "fishing_selectivity_l50"),
        ):
            bad = np.flatnonzero(~np.isfinite(_arr))
            if len(bad) > 0:
                raise ValueError(
                    f"Non-finite values in {_label} at species indices "
                    f"{bad.tolist()}: {_arr[bad].tolist()}"
                )

        bad_allo = np.flatnonzero(allometric_power <= 0)
        if len(bad_allo) > 0:
            raise ValueError(
                f"allometric_power must be > 0 for all species; "
                f"invalid at indices {bad_allo.tolist()}: {allometric_power[bad_allo].tolist()}"
            )

        rv_gate_factor_by_index, rv_gate_enabled, rv_gate_offset = _load_rv_gate(
            cfg, n_sp, n_dt, n_yr
        )
        rv_spatial_field, rv_spatial_enabled = _load_rv_spatial(cfg, n_sp)
        (
            salinity_gate_enabled,
            salinity_gate_species,
            salinity_gate_s_low,
            salinity_gate_s_high,
            salinity_field,
        ) = _load_salinity_gate(cfg, n_sp)
        _spawning_season = _load_spawning_seasons(cfg, n_sp, n_dt)
        recruitment_ceiling_by_season, recruitment_ceiling_enabled = _load_recruitment_ceiling(
            cfg, n_sp, n_dt, _spawning_season
        )
        thermal_gate_factor_by_index, thermal_gate_enabled, thermal_gate_offset = (
            _load_thermal_gate(cfg, n_sp, n_dt, n_yr)
        )
        depensation_gate_enabled, depensation_s50, depensation_theta = _load_depensation_gate(
            cfg, n_sp
        )

        return cls(
            n_species=n_sp,
            n_dt_per_year=n_dt,
            n_year=n_yr,
            n_steps=n_dt * n_yr,
            n_schools=n_schools,
            species_names=focal_species_names,
            n_background=n_bkg,
            background_file_indices=[b.file_index for b in background_list],
            all_species_names=all_species_names,
            linf=linf,
            k=k,
            t0=t0,
            egg_size=egg_size,
            condition_factor=condition_factor,
            allometric_power=allometric_power,
            vb_threshold_age=vb_threshold_age,
            lifespan_dt=lifespan_dt,
            mortality_subdt=max(1, int(cfg.get("mortality.subdt", "10"))),
            ingestion_rate=ingestion_rate,
            critical_success_rate=critical_success_rate,
            delta_lmax_factor=delta_lmax_factor,
            additional_mortality_rate=additional_mortality_rate,
            additional_mortality_by_dt=_load_additional_mortality_by_dt(cfg, n_sp),
            additional_mortality_by_dt_by_class=_load_additional_mortality_by_dt_by_class(
                cfg, n_sp, n_dt, n_dt * n_yr
            ),
            additional_mortality_spatial=_load_additional_mortality_spatial(cfg, n_sp),
            sex_ratio=sex_ratio,
            relative_fecundity=relative_fecundity,
            maturity_size=maturity_size,
            maturity_age_dt=maturity_age_dt,
            lmax=lmax,
            fishing_spatial_maps=fishing_spatial_maps,
            seeding_biomass=seeding_biomass,
            seeding_max_step=seeding_max_step,
            larva_mortality_rate=larva_mortality_rate,
            larva_mortality_by_dt=_load_larva_mortality_by_dt(cfg, n_sp, n_dt, n_dt * n_yr),
            recruitment_type=recruitment_type,
            recruitment_ssb_half=recruitment_ssb_half,
            shepherd_beta=recruitment_shepherd_beta,
            fr_shape=fr_shape,
            fr_halfsat=fr_halfsat,
            size_ratio_min=size_ratio_min_2d,
            size_ratio_max=size_ratio_max_2d,
            feeding_stage_thresholds=all_thresholds,
            feeding_stage_metric=all_metrics,
            n_feeding_stages=n_feeding_stages,
            starvation_rate_max=starvation_rate_max,
            accessibility_matrix=_load_accessibility(cfg, n_sp),
            stage_accessibility=_load_stage_accessibility(cfg, all_species_names),
            dynamic_accessibility_enabled=(
                cfg.get("predation.accessibility.dynamic.enabled", "false").lower() == "true"
            ),
            dynamic_accessibility_exponent=float(
                cfg.get("predation.accessibility.dynamic.exponent", "1.0")
            ),
            dynamic_accessibility_floor=float(
                cfg.get("predation.accessibility.dynamic.floor", "0.05")
            ),
            spawning_season=_spawning_season,
            rv_gate_factor_by_index=rv_gate_factor_by_index,
            rv_gate_enabled=rv_gate_enabled,
            rv_gate_offset=rv_gate_offset,
            rv_spatial_field=rv_spatial_field,
            rv_spatial_enabled=rv_spatial_enabled,
            salinity_gate_enabled=salinity_gate_enabled,
            salinity_gate_species=salinity_gate_species,
            salinity_gate_s_low=salinity_gate_s_low,
            salinity_gate_s_high=salinity_gate_s_high,
            salinity_field=salinity_field,
            recruitment_ceiling_by_season=recruitment_ceiling_by_season,
            recruitment_ceiling_enabled=recruitment_ceiling_enabled,
            thermal_gate_factor_by_index=thermal_gate_factor_by_index,
            thermal_gate_enabled=thermal_gate_enabled,
            thermal_gate_offset=thermal_gate_offset,
            depensation_gate_enabled=depensation_gate_enabled,
            depensation_s50=depensation_s50,
            depensation_theta=depensation_theta,
            fishing_enabled=(
                cfg.get("simulation.fishing.mortality.enabled", "true").lower() == "true"
                or fisheries_enabled
            ),
            fishing_rate=fishing_rate,
            fishing_selectivity_l50=fishing_selectivity_l50,
            fishing_selectivity_a50=fishing_selectivity_a50,
            fishing_selectivity_type=fishing_selectivity_type,
            fishing_selectivity_slope=fishing_selectivity_slope,
            fishing_seasonality=fishing_seasonality,
            fishing_rate_by_year=fishing_rate_by_year,
            fishing_rate_by_dt_by_class=fishing_rate_by_dt_by_class,
            fishing_catches=fishing_catches,
            fishing_catches_by_year=fishing_catches_by_year,
            fishing_catches_season=fishing_catches_season,
            fishing_selectivity_l75=np.concatenate(
                [fishing_selectivity_l75, np.zeros(n_bkg, dtype=np.float64)]
            )
            if n_bkg > 0
            else fishing_selectivity_l75,
            mpa_zones=mpa_zones,
            fishing_discard_rate=fishing_discard_rate,
            movement_method=movement_method,
            random_walk_range=random_walk_range,
            out_mortality_rate=out_mortality_rate,
            egg_weight_override=egg_weight_override,
            output_cutoff_age=_output["output_cutoff_age"],
            output_record_frequency=_output["output_record_frequency"],
            diet_output_enabled=_output["diet_output_enabled"],
            output_step0_include=_output["output_step0_include"],
            movement_seed_fixed=movement_seed_fixed,
            mortality_seed_fixed=mortality_seed_fixed,
            java_compat_rng=java_compat_rng,
            movement_strict_coverage=movement_strict,
            random_distribution_ncell=random_distribution_ncell,
            growth_class=growth_class,
            vb_species_ids=frozenset(i for i, gc in enumerate(growth_class) if gc == "VB"),
            gompertz_ke=gompertz_ke,
            gompertz_lstart=gompertz_lstart,
            gompertz_kg=gompertz_kg,
            gompertz_tg=gompertz_tg,
            gompertz_linf=gompertz_linf,
            gompertz_thr_age_exp_dt=gompertz_thr_age_exp_dt,
            gompertz_thr_age_gom_dt=gompertz_thr_age_gom_dt,
            raw_config=cfg,
            bioen_enabled=_bioen_enabled,
            bioen_phit_enabled=_bioen_phit_enabled,
            bioen_fo2_enabled=_bioen_fo2_enabled,
            bioen_beta=bioen_beta,
            bioen_zlayer=bioen_zlayer,
            bioen_assimilation=bioen_assimilation,
            bioen_c_m=bioen_c_m,
            bioen_eta=bioen_eta,
            bioen_r=bioen_r,
            bioen_m0=bioen_m0,
            bioen_m1=bioen_m1,
            bioen_e_mobi=bioen_e_mobi,
            bioen_e_d=bioen_e_d,
            bioen_tp=bioen_tp,
            bioen_e_maint=bioen_e_maint,
            bioen_o2_c1=bioen_o2_c1,
            bioen_o2_c2=bioen_o2_c2,
            bioen_i_max=bioen_i_max,
            bioen_theta=bioen_theta,
            bioen_c_rate=bioen_c_rate,
            bioen_k_for=bioen_k_for,
            foraging_k1_for=foraging_k1_for,
            foraging_k2_for=foraging_k2_for,
            foraging_I_max=foraging_I_max,
            genetics_enabled=genetics_enabled,
            genetics_transmission_year=genetics_transmission_year,
            genetics_n_neutral=genetics_n_neutral,
            genetics_n_neutral_val=genetics_n_neutral_val,
            economics_enabled=economics_enabled,
            output_biomass_byage=_output["output_biomass_byage"],
            output_biomass_bysize=_output["output_biomass_bysize"],
            output_abundance_byage=_output["output_abundance_byage"],
            output_abundance_bysize=_output["output_abundance_bysize"],
            output_meantl=_output["output_meantl"],
            output_yield_abundance=_output["output_yield_abundance"],
            output_mean_size=_output["output_mean_size"],
            output_yield_abundance_netcdf=_output["output_yield_abundance_netcdf"],
            output_mean_size_netcdf=_output["output_mean_size_netcdf"],
            output_ssb=_output["output_ssb"],
            output_ssb_netcdf=_output["output_ssb_netcdf"],
            output_size_min=_output["output_size_min"],
            output_size_max=_output["output_size_max"],
            output_size_incr=_output["output_size_incr"],
            output_bioen_ingest=_output["output_bioen_ingest"],
            output_bioen_maint=_output["output_bioen_maint"],
            output_bioen_rho=_output["output_bioen_rho"],
            output_bioen_sizeinf=_output["output_bioen_sizeinf"],
            output_biomass_netcdf=_output["output_biomass_netcdf"],
            output_abundance_netcdf=_output["output_abundance_netcdf"],
            output_yield_biomass_netcdf=_output["output_yield_biomass_netcdf"],
            output_biomass_byage_netcdf=_output["output_biomass_byage_netcdf"],
            output_abundance_byage_netcdf=_output["output_abundance_byage_netcdf"],
            output_biomass_bysize_netcdf=_output["output_biomass_bysize_netcdf"],
            output_abundance_bysize_netcdf=_output["output_abundance_bysize_netcdf"],
            output_mortality_netcdf=_output["output_mortality_netcdf"],
            output_spatial_enabled=_output["output_spatial_enabled"],
            output_spatial_biomass=_output["output_spatial_biomass"],
            output_spatial_abundance=_output["output_spatial_abundance"],
            output_spatial_yield_biomass=_output["output_spatial_yield_biomass"],
        )
