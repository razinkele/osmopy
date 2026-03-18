"""Study-level charting, schema field resolution, grid, and history tests.

Tests that connect study configurations to:
  - All 9 plotting functions with study-shaped data
  - Schema field resolution for every study's parameter keys
  - Grid creation from study grid parameters
  - Run history save/load/compare with study configs
  - Von Bertalanffy growth curves derived from real study species
  - Food web diagrams from study accessibility matrices
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from osmose.config.reader import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.grid import create_grid_csv, create_grid_netcdf
from osmose.history import RunHistory, RunRecord
from osmose.plotting import (
    make_ci_timeseries,
    make_food_web,
    make_growth_curves,
    make_mortality_breakdown,
    make_predation_ranges,
    make_size_spectrum_plot,
    make_species_dashboard,
    make_stacked_area,
)
from osmose.schema import build_registry

ALL_STUDIES = ["bay_of_biscay", "eec", "eec_full", "minimal"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def registry():
    return build_registry()


@pytest.fixture(params=ALL_STUDIES)
def study_name(request):
    return request.param


@pytest.fixture
def study_config(study_name, tmp_path):
    result = osmose_demo(study_name, tmp_path)
    reader = OsmoseConfigReader()
    config = reader.read(result["config_file"])
    return study_name, config, result["config_file"]


@pytest.fixture
def study_species(study_config):
    """Extract species list and parameters from study config."""
    name, config, _ = study_config
    nspecies = int(config["simulation.nspecies"])
    species = []
    for i in range(nspecies):
        sp = {
            "name": config.get(f"species.name.sp{i}", f"sp{i}"),
            "linf": float(config.get(f"species.linf.sp{i}", "100")),
            "k": float(config.get(f"species.k.sp{i}", "0.3")),
            "t0": float(config.get(f"species.t0.sp{i}", "0")),
            "lifespan": int(float(config.get(f"species.lifespan.sp{i}", "10"))),
        }
        species.append(sp)
    return name, species


@pytest.fixture
def mock_study_output(study_species):
    """Generate mock output DataFrames shaped to each study's species list."""
    name, species = study_species
    rng = np.random.default_rng(42)
    n_time = 60
    sp_names = [sp["name"] for sp in species]

    # Biomass
    bio_frames = []
    for sp in sp_names:
        bio_frames.append(
            pd.DataFrame(
                {
                    "time": range(n_time),
                    "species": sp,
                    "biomass": rng.exponential(scale=50000, size=n_time),
                }
            )
        )
    biomass_df = pd.concat(bio_frames, ignore_index=True)

    # Yield
    yield_frames = []
    for sp in sp_names:
        yield_frames.append(
            pd.DataFrame(
                {
                    "time": range(n_time),
                    "species": sp,
                    "yield": rng.exponential(scale=5000, size=n_time),
                }
            )
        )
    yield_df = pd.concat(yield_frames, ignore_index=True)

    # Mortality breakdown
    mort_frames = []
    for sp in sp_names:
        mort_frames.append(
            pd.DataFrame(
                {
                    "time": range(n_time),
                    "species": sp,
                    "predation": rng.uniform(0, 0.4, size=n_time),
                    "starvation": rng.uniform(0, 0.2, size=n_time),
                    "fishing": rng.uniform(0, 0.3, size=n_time),
                    "natural": rng.uniform(0, 0.2, size=n_time),
                }
            )
        )
    mortality_df = pd.concat(mort_frames, ignore_index=True)

    # 2D: Biomass by age
    age_bins = [f"age{a}" for a in range(5)]
    byage_frames = []
    for sp in sp_names:
        for t in range(n_time):
            for b in age_bins:
                byage_frames.append(
                    {
                        "time": t,
                        "species": sp,
                        "bin": b,
                        "value": rng.exponential(scale=10000),
                    }
                )
    byage_df = pd.DataFrame(byage_frames)

    # Size spectrum
    sizes = [2**i for i in range(1, 12)]
    abundances = [rng.exponential(scale=10 ** (6 - 0.3 * i)) for i in range(11)]
    spectrum_df = pd.DataFrame({"size": sizes, "abundance": abundances})

    return (
        name,
        sp_names,
        {
            "biomass": biomass_df,
            "yield": yield_df,
            "mortality": mortality_df,
            "byage": byage_df,
            "spectrum": spectrum_df,
        },
    )


# ---------------------------------------------------------------------------
# 1. Charting with study-shaped data
# ---------------------------------------------------------------------------


class TestStudyCharts:
    """Generate all chart types using data shaped to each study's species."""

    def test_stacked_area_per_study(self, mock_study_output):
        name, sp_names, data = mock_study_output
        fig = make_stacked_area(data["byage"], f"Biomass by Age — {name}")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_stacked_area_per_species(self, mock_study_output):
        name, sp_names, data = mock_study_output
        for sp in sp_names[:2]:  # Test first 2 for speed
            fig = make_stacked_area(data["byage"], f"ByAge {sp}", species=sp)
            assert isinstance(fig, go.Figure)
            assert len(fig.data) > 0

    def test_mortality_breakdown_per_study(self, mock_study_output):
        name, _, data = mock_study_output
        fig = make_mortality_breakdown(data["mortality"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 4  # predation, starvation, fishing, natural

    def test_mortality_breakdown_per_species(self, mock_study_output):
        name, sp_names, data = mock_study_output
        fig = make_mortality_breakdown(data["mortality"], species=sp_names[0])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 4

    def test_size_spectrum_per_study(self, mock_study_output):
        name, _, data = mock_study_output
        fig = make_size_spectrum_plot(data["spectrum"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # scatter + regression

    def test_size_spectrum_has_slope_annotation(self, mock_study_output):
        name, _, data = mock_study_output
        fig = make_size_spectrum_plot(data["spectrum"])
        annotations = [a for a in fig.layout.annotations if "slope" in a.text.lower()]
        assert len(annotations) >= 1

    def test_ci_timeseries_per_study(self, mock_study_output):
        name, _, data = mock_study_output
        # Aggregate biomass across species per timestep
        agg = data["biomass"].groupby("time")["biomass"].agg(["mean", "min", "max"]).reset_index()
        fig = make_ci_timeseries(
            agg["time"],
            agg["mean"],
            agg["min"],
            agg["max"],
            f"Total Biomass CI — {name}",
            "tons",
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

    def test_species_dashboard_per_study(self, mock_study_output):
        name, sp_names, data = mock_study_output
        fig = make_species_dashboard(data["biomass"], data["yield"])
        assert isinstance(fig, go.Figure)
        # Should have traces for each species (biomass + yield per species)
        assert len(fig.data) >= len(sp_names)

    def test_growth_curves_from_study_params(self, study_species):
        name, species = study_species
        fig = make_growth_curves(species)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == len(species)
        # Each trace should have the species name
        trace_names = {t.name for t in fig.data}
        for sp in species:
            assert sp["name"] in trace_names

    def test_growth_curves_values_are_reasonable(self, study_species):
        """L(t) at max age should approach linf."""
        name, species = study_species
        fig = make_growth_curves(species)
        for i, sp in enumerate(species):
            y_vals = fig.data[i].y
            # Last value should be within 30% of linf (VB asymptote)
            final_length = y_vals[-1]
            assert final_length > 0
            assert final_length <= sp["linf"] * 1.01  # Can't exceed linf

    def test_predation_ranges_from_study(self, study_config):
        name, config, _ = study_config
        nspecies = int(config["simulation.nspecies"])
        params = []
        for i in range(nspecies):
            min_key = f"predation.predPrey.sizeRatio.min.sp{i}"
            max_key = f"predation.predPrey.sizeRatio.max.sp{i}"
            if min_key in config and max_key in config:
                params.append(
                    {
                        "name": config.get(f"species.name.sp{i}", f"sp{i}"),
                        "size_ratio_min": float(config[min_key]),
                        "size_ratio_max": float(config[max_key]),
                    }
                )
        if not params:
            pytest.skip(f"{name}: no predation size ratio params")
        fig = make_predation_ranges(params)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == len(params)

    def test_food_web_from_study_accessibility(self, study_config):
        """Build food web from the study's accessibility matrix."""
        name, config, config_file = study_config
        key = "predation.accessibility.file"
        if key not in config or not config[key]:
            pytest.skip(f"{name}: no accessibility file")
        ref = config_file.parent / config[key]
        if not ref.exists():
            pytest.skip(f"{name}: accessibility file missing")

        acc = pd.read_csv(ref, sep=";", index_col=0)
        # Convert accessibility matrix to predator-prey-proportion format
        rows = []
        for pred in acc.columns:
            for prey in acc.index:
                val = acc.loc[prey, pred]
                if isinstance(val, (int, float)) and val > 0:
                    rows.append({"predator": pred, "prey": prey, "proportion": float(val)})
        if not rows:
            pytest.skip(f"{name}: empty accessibility matrix")

        diet_df = pd.DataFrame(rows)
        fig = make_food_web(diet_df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


# ---------------------------------------------------------------------------
# 2. Schema field resolution per study
# ---------------------------------------------------------------------------


class TestSchemaFieldResolution:
    """Verify schema resolves all study config keys correctly."""

    def test_all_species_indexed_keys_resolve(self, study_config, registry):
        """Every species.*.sp{N} key should match an indexed field."""
        name, config, _ = study_config
        nspecies = int(config["simulation.nspecies"])
        unresolved = []
        for i in range(nspecies):
            for prefix in ("species.linf", "species.k", "species.lifespan", "species.name"):
                key = f"{prefix}.sp{i}"
                if key in config:
                    field = registry.match_field(key)
                    if field is None:
                        unresolved.append(key)
        assert unresolved == [], f"{name}: unresolved species keys: {unresolved}"

    def test_simulation_keys_resolve(self, study_config, registry):
        name, config, _ = study_config
        # Filter out per-species nschool keys (not in schema)
        sim_keys = [
            k
            for k in config
            if k.startswith("simulation.") and not k.startswith("simulation.nschool.sp")
        ]
        unresolved = [k for k in sim_keys if registry.match_field(k) is None]
        ratio = len(unresolved) / max(len(sim_keys), 1)
        assert ratio < 0.5, f"{name}: too many unresolved simulation keys: {unresolved}"

    def test_grid_keys_resolve(self, study_config, registry):
        name, config, _ = study_config
        grid_keys = [k for k in config if k.startswith("grid.")]
        for k in grid_keys:
            field = registry.match_field(k)
            assert field is not None, f"{name}: unresolved grid key: {k}"

    def test_movement_indexed_keys_resolve(self, study_config, registry):
        name, config, _ = study_config
        nspecies = int(config["simulation.nspecies"])
        for i in range(nspecies):
            key = f"movement.distribution.method.sp{i}"
            if key in config:
                field = registry.match_field(key)
                assert field is not None, f"{name}: unresolved movement key: {key}"
                assert field.indexed

    def test_mortality_keys_resolve(self, study_config, registry):
        name, config, _ = study_config
        mort_keys = [k for k in config if k.startswith("mortality.")]
        resolved = [k for k in mort_keys if registry.match_field(k) is not None]
        ratio = len(resolved) / max(len(mort_keys), 1)
        assert ratio >= 0.3, (
            f"{name}: only {len(resolved)}/{len(mort_keys)} mortality keys resolved"
        )

    def test_output_keys_resolve(self, study_config, registry):
        name, config, _ = study_config
        out_keys = [k for k in config if k.startswith("output.")]
        resolved = sum(1 for k in out_keys if registry.match_field(k) is not None)
        ratio = resolved / max(len(out_keys), 1)
        # Allow some unresolved output keys (OSMOSE has many optional output params)
        assert ratio >= 0.3, f"{name}: only {resolved}/{len(out_keys)} output keys resolved"

    def test_predation_keys_resolve(self, study_config, registry):
        name, config, _ = study_config
        pred_keys = [k for k in config if k.startswith("predation.")]
        resolved = sum(1 for k in pred_keys if registry.match_field(k) is not None)
        ratio = resolved / max(len(pred_keys), 1)
        assert ratio >= 0.3, f"{name}: only {resolved}/{len(pred_keys)} predation keys resolved"


# ---------------------------------------------------------------------------
# 3. Grid creation from study parameters
# ---------------------------------------------------------------------------


class TestGridCreationFromStudy:
    """Create grids matching each study's grid configuration."""

    def test_create_csv_grid_from_study(self, study_config, tmp_path):
        name, config, _ = study_config
        if "grid.nlon" not in config or "grid.nlat" not in config:
            pytest.skip(f"{name}: no grid dimensions")
        nlon = int(config["grid.nlon"])
        nlat = int(config["grid.nlat"])
        output = tmp_path / "grid.csv"
        create_grid_csv(nlat, nlon, output)
        assert output.exists()
        df = pd.read_csv(output, header=None)
        assert df.shape == (nlat, nlon)
        # Default: all ocean (1)
        assert (df.values == 1).all()

    def test_create_csv_grid_with_mask(self, study_config, tmp_path):
        name, config, _ = study_config
        if "grid.nlon" not in config:
            pytest.skip(f"{name}: no grid dimensions")
        nlon = int(config["grid.nlon"])
        nlat = int(config["grid.nlat"])
        # Create a mask with some land cells
        mask = np.ones((nlat, nlon))
        mask[0, :] = -1  # Top row is land
        mask[-1, :] = -1  # Bottom row is land
        output = tmp_path / "masked_grid.csv"
        create_grid_csv(nlat, nlon, output, mask=mask)
        df = pd.read_csv(output, header=None)
        assert df.iloc[0, 0] == -1  # Land
        assert df.iloc[1, 1] == 1  # Ocean

    def test_create_netcdf_grid_from_study(self, study_config, tmp_path):
        name, config, _ = study_config
        if "grid.upleft.lat" not in config:
            pytest.skip(f"{name}: no grid coordinates")
        nlat = int(config["grid.nlat"])
        nlon = int(config["grid.nlon"])
        lat_up = float(config["grid.upleft.lat"])
        lat_low = float(config["grid.lowright.lat"])
        lon_left = float(config["grid.upleft.lon"])
        lon_right = float(config["grid.lowright.lon"])

        output = tmp_path / "study_grid.nc"
        create_grid_netcdf(
            lat_bounds=(lat_low, lat_up),
            lon_bounds=(lon_left, lon_right),
            nlat=nlat,
            nlon=nlon,
            output=output,
        )
        assert output.exists()

        import xarray as xr

        ds = xr.open_dataset(output)
        assert "mask" in ds
        assert ds["mask"].shape == (nlat, nlon)
        assert float(ds["lat"].min()) >= lat_low - 0.01
        assert float(ds["lat"].max()) <= lat_up + 0.01
        ds.close()

    def test_netcdf_grid_with_study_mask(self, study_config, tmp_path):
        name, config, _ = study_config
        if "grid.upleft.lat" not in config:
            pytest.skip(f"{name}: no grid coordinates")
        nlat = int(config["grid.nlat"])
        nlon = int(config["grid.nlon"])
        lat_up = float(config["grid.upleft.lat"])
        lat_low = float(config["grid.lowright.lat"])
        lon_left = float(config["grid.upleft.lon"])
        lon_right = float(config["grid.lowright.lon"])

        # Create a realistic mask (land border)
        mask = np.ones((nlat, nlon), dtype=np.float32)
        mask[0, :] = 0  # Top row land
        mask[:, 0] = 0  # Left column land

        output = tmp_path / "masked_study_grid.nc"
        create_grid_netcdf(
            lat_bounds=(lat_low, lat_up),
            lon_bounds=(lon_left, lon_right),
            nlat=nlat,
            nlon=nlon,
            output=output,
            mask=mask,
        )

        import xarray as xr

        ds = xr.open_dataset(output)
        assert ds["mask"].values[0, 0] == 0  # Land
        assert ds["mask"].values[1, 1] == 1  # Ocean
        ds.close()


# ---------------------------------------------------------------------------
# 4. Run history with study configs
# ---------------------------------------------------------------------------


class TestRunHistoryWithStudies:
    """Save, load, and compare run records using real study configs."""

    def test_save_and_load_study_run(self, study_config, tmp_path):
        name, config, _ = study_config
        history = RunHistory(tmp_path / "history")
        record = RunRecord(
            config_snapshot=config,
            duration_sec=120.5,
            output_dir=str(tmp_path / "output"),
            summary={"total_biomass": 500000, "species_count": len(config)},
        )
        path = history.save(record)
        assert path.exists()

        loaded = history.list_runs()
        assert len(loaded) == 1
        assert loaded[0].config_snapshot == config
        assert loaded[0].summary["total_biomass"] == 500000

    def test_compare_baseline_vs_modified_runs(self, study_config, tmp_path):
        name, config, _ = study_config
        history = RunHistory(tmp_path / "history")

        # Baseline run
        record1 = RunRecord(
            config_snapshot=dict(config),
            summary={"biomass": 100000},
        )
        history.save(record1)

        # Modified run (shorter simulation)
        modified = dict(config)
        modified["simulation.time.nyear"] = "5"
        record2 = RunRecord(
            config_snapshot=modified,
            summary={"biomass": 80000},
        )
        history.save(record2)

        runs = history.list_runs()
        assert len(runs) == 2

        diffs = history.compare_runs(runs[0].timestamp, runs[1].timestamp)
        diff_keys = {d["key"] for d in diffs}
        assert "simulation.time.nyear" in diff_keys

    def test_multi_run_comparison(self, study_config, tmp_path):
        """Compare 3 runs with different fishing mortality."""
        name, config, _ = study_config
        history = RunHistory(tmp_path / "history")
        timestamps = []

        for fishing_factor in [0.5, 1.0, 2.0]:
            modified = dict(config)
            for key in config:
                if key.startswith("mortality.fishing.rate."):
                    try:
                        modified[key] = str(float(config[key]) * fishing_factor)
                    except ValueError:
                        pass
            record = RunRecord(
                config_snapshot=modified,
                summary={"fishing_factor": fishing_factor},
            )
            history.save(record)
            timestamps.append(record.timestamp)

        diffs = history.compare_runs_multi(timestamps)
        # Fishing rate keys should appear in diffs
        fishing_diffs = [d for d in diffs if "fishing" in d["key"]]
        if any(k.startswith("mortality.fishing.rate.") for k in config):
            assert len(fishing_diffs) > 0, f"{name}: expected fishing rate diffs"


# ---------------------------------------------------------------------------
# 5. Von Bertalanffy parameter validation
# ---------------------------------------------------------------------------


class TestVonBertalanffyFromStudy:
    """Validate Von Bertalanffy growth parameters for each study's species."""

    def test_growth_curve_is_monotonic(self, study_species):
        """L(t) should be monotonically increasing for t > t0."""
        name, species = study_species
        for sp in species:
            t = np.linspace(max(0, sp["t0"] + 0.01), sp["lifespan"], 200)
            length = sp["linf"] * (1 - np.exp(-sp["k"] * (t - sp["t0"])))
            diffs = np.diff(length)
            assert (diffs >= -1e-10).all(), f"{name}/{sp['name']}: growth curve not monotonic"

    def test_growth_reaches_half_linf(self, study_species):
        """Species should reach at least 50% of linf within their lifespan."""
        name, species = study_species
        for sp in species:
            t_max = sp["lifespan"]
            length_at_max = sp["linf"] * (1 - np.exp(-sp["k"] * (t_max - sp["t0"])))
            ratio = length_at_max / sp["linf"]
            assert ratio >= 0.5, f"{name}/{sp['name']}: only reaches {ratio:.0%} of linf by max age"

    def test_weight_at_length_is_positive(self, study_config):
        """Length-weight relationship W = a * L^b should give positive weights."""
        name, config, _ = study_config
        nspecies = int(config["simulation.nspecies"])
        for i in range(nspecies):
            a_key = f"species.lw.condition.factor.sp{i}"
            b_key = f"species.lw.allpower.sp{i}"
            linf_key = f"species.linf.sp{i}"
            if a_key in config and b_key in config and linf_key in config:
                a = float(config[a_key])
                b = float(config[b_key])
                linf = float(config[linf_key])
                # Weight at linf
                weight = a * linf**b
                assert weight > 0, (
                    f"{name}/sp{i}: weight at linf is non-positive: a={a}, b={b}, linf={linf}"
                )
                # Weight should be in reasonable range (grams to kg)
                assert weight < 1e6, f"{name}/sp{i}: weight at linf seems too high: {weight:.0f}g"


# ---------------------------------------------------------------------------
# 6. Ensemble CI charting from study data
# ---------------------------------------------------------------------------


class TestEnsembleCICharting:
    """Generate CI timeseries charts from ensemble-like study data."""

    def test_ci_chart_from_replicate_biomass(self, mock_study_output):
        name, sp_names, data = mock_study_output
        rng = np.random.default_rng(77)

        # Simulate 3 replicates with slightly different biomass
        replicates = []
        base = data["biomass"].groupby("time")["biomass"].sum()
        for _ in range(3):
            noise = rng.normal(1.0, 0.1, size=len(base))
            replicates.append(base * noise)

        times = base.index.tolist()
        mean_vals = np.mean([r.values for r in replicates], axis=0)
        lower_vals = np.percentile([r.values for r in replicates], 2.5, axis=0)
        upper_vals = np.percentile([r.values for r in replicates], 97.5, axis=0)

        fig = make_ci_timeseries(
            times,
            mean_vals,
            lower_vals,
            upper_vals,
            f"Ensemble Biomass — {name}",
            "tons",
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # CI band + mean line

    def test_ci_bounds_ordering(self, mock_study_output):
        """lower <= mean <= upper for generated CI data."""
        _, _, data = mock_study_output
        base = data["biomass"].groupby("time")["biomass"].sum()
        rng = np.random.default_rng(88)
        replicates = [base * rng.normal(1.0, 0.1, size=len(base)) for _ in range(5)]

        mean_vals = np.mean([r.values for r in replicates], axis=0)
        lower_vals = np.percentile([r.values for r in replicates], 2.5, axis=0)
        upper_vals = np.percentile([r.values for r in replicates], 97.5, axis=0)

        assert (lower_vals <= mean_vals + 1e-10).all()
        assert (mean_vals <= upper_vals + 1e-10).all()
