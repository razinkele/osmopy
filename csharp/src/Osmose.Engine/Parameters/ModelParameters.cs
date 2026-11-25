namespace Osmose.Engine;

/// <summary>Species counts and per-species lifespans (bundle group <c>species</c>).</summary>
public sealed class SpeciesParameters
{
    public int NFocal { get; init; }

    public int NBackground { get; init; }

    public int NTotal => NFocal + NBackground;

    /// <summary>Schools per reproduction event, focal species.</summary>
    public int[] NSchools { get; init; } = [];

    /// <summary>Lifespan in steps, every species.</summary>
    public int[] LifespanDt { get; init; } = [];
}

/// <summary>Growth (bundle group <c>growth</c>): every species, except <see cref="Class"/> (focal).</summary>
public sealed class GrowthParameters
{
    /// <summary>0 von Bertalanffy, 1 Gompertz (focal).</summary>
    public byte[] Class { get; init; } = [];

    public double[] Linf { get; init; } = [];

    public double[] K { get; init; } = [];

    public double[] T0 { get; init; } = [];

    public double[] EggSize { get; init; } = [];

    public double[] VbThresholdAge { get; init; } = [];

    public double[] ConditionFactor { get; init; } = [];

    public double[] AllometricPower { get; init; } = [];

    /// <summary>+inf when unset (no cap).</summary>
    public double[] Lmax { get; init; } = [];

    public double[] DeltaLmaxFactor { get; init; } = [];

    public double[] CriticalSuccessRate { get; init; } = [];
}

/// <summary>Predation and feeding stages (bundle group <c>predation</c>).</summary>
public sealed class PredationParameters
{
    public double[] IngestionRate { get; init; } = [];

    public int[] FrShape { get; init; } = [];

    public double[] FrHalfSat { get; init; } = [];

    /// <summary>Width of the size-ratio tables.</summary>
    public int NStage { get; init; }

    /// <summary><c>[species * NStage + stage]</c>.</summary>
    public double[] SizeRatioMin { get; init; } = [];

    public double[] SizeRatioMax { get; init; } = [];

    /// <summary>0 size, 1 age, 2 weight, 3 trophic level.</summary>
    public byte[] StageMetric { get; init; } = [];

    /// <summary>Species <c>s</c>'s sorted thresholds are <c>StageThresholds[StageOffsets[s]..StageOffsets[s + 1]]</c>.</summary>
    public int[] StageOffsets { get; init; } = [];

    public double[] StageThresholds { get; init; } = [];
}

/// <summary>Stage accessibility (bundle group <c>access</c>, mode 2). An empty range means the
/// species is absent from the CSV and resolves to -1, which the kernel treats as FULL access.</summary>
public sealed class AccessParameters
{
    /// <summary>0 none, 1 species matrix, 2 stage matrix.</summary>
    public int Mode { get; init; }

    /// <summary>The matrix in Python's layout, row-major.</summary>
    public double[] Matrix { get; init; } = [];

    public int Rows { get; init; }

    public int Cols { get; init; }

    public int[] PreyOffsets { get; init; } = [];

    public double[] PreyThresholds { get; init; } = [];

    public int[] PreyIndex { get; init; } = [];

    public int[] PredOffsets { get; init; } = [];

    public double[] PredThresholds { get; init; } = [];

    public int[] PredIndex { get; init; } = [];
}

/// <summary>Non-fishing mortality (bundle group <c>mortality</c>).</summary>
public sealed class MortalityParameters
{
    /// <summary><c>[step * NTotal + species]</c>.</summary>
    public double[] AdditionalRateByStep { get; init; } = [];

    /// <summary><c>[step * NFocal + species]</c>: the full per-cohort larva rate.</summary>
    public double[] LarvaRateByStep { get; init; } = [];

    public double[] OutRate { get; init; } = [];

    public double[] StarvationRateMax { get; init; } = [];
}

/// <summary>Fishing (bundle group <c>fishing</c>).</summary>
public sealed class FishingParameters
{
    public bool Enabled { get; init; }

    /// <summary><c>[step * NTotal + species]</c>.</summary>
    public double[] RateByStep { get; init; } = [];

    /// <summary><c>[step * NTotal + species]</c>, or null when the config has no seasonality: the
    /// rate's denominator is then <c>ndt * n_subdt</c> instead of <c>n_subdt</c>.</summary>
    public double[]? SeasonByStep { get; init; }

    public int[] SelectivityType { get; init; } = [];

    public double[] SelectivityA50 { get; init; } = [];

    public double[] SelectivityL50 { get; init; } = [];

    public double[] SelectivitySlope { get; init; } = [];

    /// <summary>Per focal species, its map in <see cref="Maps"/> or -1.</summary>
    public int[] MapIndex { get; init; } = [];

    /// <summary><c>[(map * ny + y) * nx + x]</c>.</summary>
    public double[] Maps { get; init; } = [];

    /// <summary>Null when the config sets no discard rate.</summary>
    public double[]? DiscardRate { get; init; }

    /// <summary>The per-fishery path (Java's model), or null: the per-species tables above then
    /// apply. Present only for configs the per-species tables cannot represent
    /// (osmose/engine/fisheries.py).</summary>
    public FisheryParameters? Fishery { get; init; }
}

/// <summary>Per-fishery fishing (bundle group <c>fishing.fishery</c>, contract version 3).</summary>
public sealed class FisheryParameters
{
    /// <summary><c>MapIndex</c> value: no map covers this step, the fishery does not fish.</summary>
    public const int Off = -1;

    /// <summary><c>MapIndex</c> value: the fishery defines no map, spatial factor 1.</summary>
    public const int NoMap = -2;

    public int Count { get; init; }

    public double[] Rate { get; init; } = [];

    public int[] SelectivityType { get; init; } = [];

    public double[] SelectivityA50 { get; init; } = [];

    public double[] SelectivityL50 { get; init; } = [];

    public double[] SelectivitySlope { get; init; } = [];

    /// <summary><c>[species * Count + fishery]</c>, focal species.</summary>
    public double[] Catchability { get; init; } = [];

    /// <summary><c>[species * Count + fishery]</c>, focal species.</summary>
    public double[] Discard { get; init; } = [];

    /// <summary><c>[fishery * ndt + step_in_year]</c>, or null (no seasonality: the term's
    /// denominator is <c>ndt * n_subdt</c>).</summary>
    public double[]? Season { get; init; }

    /// <summary><c>[fishery * NSteps + step]</c>: a map, <see cref="Off"/> or <see cref="NoMap"/>.</summary>
    public int[] MapIndex { get; init; } = [];

    public int NSteps { get; init; }

    /// <summary><c>[(map * ny + y) * nx + x]</c>.</summary>
    public double[] Maps { get; init; } = [];
}

/// <summary>LTL resources, non-depletable (bundle group <c>resources</c>).</summary>
public sealed class ResourceParameters
{
    public int N { get; init; }

    public double[] SizeMin { get; init; } = [];

    public double[] SizeMax { get; init; } = [];

    public double[] TrophicLevel { get; init; } = [];

    /// <summary>Stage-accessibility prey row per resource, -1 unresolved.</summary>
    public int[] AccessRows { get; init; } = [];

    public int NFrames { get; init; }

    /// <summary><c>[(resource * NFrames + frame) * nCells + cell]</c>, already times the multiplier.</summary>
    public double[] Forcing { get; init; } = [];

    /// <summary><c>[step * N + resource]</c>.</summary>
    public int[] FrameByStep { get; init; } = [];

    /// <summary><c>[step * N + resource]</c>.</summary>
    public double[] AccessByStep { get; init; } = [];

    /// <summary>Feature <c>oxygen_benthos</c>: 1 for each resource whose carrying capacity bottom
    /// oxygen scales; empty when the bundle does not declare the feature.</summary>
    public byte[] OxygenCoupled { get; init; } = [];

    /// <summary><c>[frame * nCells + cell]</c>: the Hill factor, pre-evaluated by the builder with
    /// the Python engine's own function, so applying it is one multiplication (tier E).</summary>
    public double[] OxygenFactor { get; init; } = [];

    /// <summary><c>[step]</c>: the oxygen frame of each step.</summary>
    public int[] OxygenFrameByStep { get; init; } = [];

    /// <summary>Feature <c>depletable_ltl</c>: the biomass predation leaves is carried to the
    /// next step and regrown toward the carrying capacity, where it is otherwise reset to it.</summary>
    public bool Depletable { get; init; }

    /// <summary>The fraction of the carrying capacity regrowth starts from at the least, so that
    /// an emptied cell recovers.</summary>
    public double DepletableFloor { get; init; }

    /// <summary>Per resource: the logistic regrowth rate, per step.</summary>
    public double[] RegrowthRate { get; init; } = [];
}

/// <summary>Background (forced) species (bundle group <c>background</c>, feature
/// <c>background_species</c>): what <c>BackgroundState.get_schools</c> builds a step's schools
/// from. Empty (<see cref="NSpecies"/> 0) without the feature.</summary>
public sealed record BackgroundParameters
{
    public int NSpecies { get; init; }

    /// <summary>Classes of all species together.</summary>
    public int NClasses { get; init; }

    /// <summary><c>[species]</c> .. <c>[species + 1]</c>: a species' classes.</summary>
    public int[] ClassOffsets { get; init; } = [0];

    public double[] ClassLength { get; init; } = [];

    /// <summary>Tonnes per individual.</summary>
    public double[] ClassWeight { get; init; } = [];

    public double[] ClassTrophicLevel { get; init; } = [];

    public int[] ClassAgeDt { get; init; } = [];

    /// <summary><c>[step * NClasses + class]</c>: the class's share of its species' biomass.</summary>
    public double[] ProportionByStep { get; init; } = [];

    /// <summary>The ocean cells (<c>y * nx + x</c>), in the Python state's order.</summary>
    public int[] OceanCells { get; init; } = [];

    /// <summary><c>[frame * nOcean + ocean cell]</c>: species biomass per ocean cell, tonnes.</summary>
    public double[] Base { get; init; } = [];

    /// <summary><c>[step * NSpecies + species]</c>: the frame of <see cref="Base"/> for the step.</summary>
    public int[] FrameByStep { get; init; } = [];
}

/// <summary>Reads the model groups from a bundle through one <see cref="BundleReader"/>, so every
/// missing entry or wrong shape lands in the same problem list as the header's.</summary>
/// <summary>Reproduction (bundle group <c>reproduction</c>, focal species, spec 6.8).</summary>
public sealed record ReproductionParameters
{
    public double[] SexRatio { get; init; } = [];

    public double[] RelativeFecundity { get; init; } = [];

    public double[] MaturitySize { get; init; } = [];

    public int[] MaturityAgeDt { get; init; } = [];

    public double[] SeedingBiomass { get; init; } = [];

    /// <summary>Seeding applies while <c>step &lt; SeedingMaxStep</c>.</summary>
    public int[] SeedingMaxStep { get; init; } = [];

    /// <summary><c>[step, species]</c>: the spawning-season share of the step.</summary>
    public double[] SeasonByStep { get; init; } = [];

    public double[] EggLength { get; init; } = [];

    /// <summary>Egg weight in tonnes, resolved by the builder exactly as <c>create_egg_schools</c>.</summary>
    public double[] EggWeight { get; init; } = [];

    /// <summary>Stock-recruitment curve per species (<c>apply_stock_recruitment</c>): 0 none,
    /// 1 Beverton-Holt, 2 Ricker, 3 hockey stick, 4 Shepherd.</summary>
    public byte[] SrType { get; init; } = [];

    /// <summary>The curve's characteristic SSB in tonnes; empty without feature
    /// <c>stock_recruitment</c>.</summary>
    public double[] SrSsbHalf { get; init; } = [];

    /// <summary>The Shepherd exponent; empty without feature <c>stock_recruitment</c>.</summary>
    public double[] ShepherdBeta { get; init; } = [];

    /// <summary>0 stock_recruitment; 1 linear (feature <c>seeding_linear</c>): a seeded species
    /// skips the curve.</summary>
    public int SeedingMode { get; init; }

    /// <summary>Feature <c>rv_gate</c>: 1 for each gated species; empty without the feature.</summary>
    public byte[] RvEnabled { get; init; } = [];

    /// <summary><c>[year * nFocal + species]</c>: the reproductive-volume egg multiplier of each
    /// model year, resolved by the builder with the Python engine's <c>rv_gate_factor</c>.</summary>
    public double[] RvFactorByYear { get; init; } = [];

    /// <summary>Feature <c>depensation_gate</c>: 1 for each gated species; empty without the
    /// feature.</summary>
    public byte[] DepensationEnabled { get; init; } = [];

    /// <summary>The spawning stock, in tonnes, at which the gate lets half the eggs through.</summary>
    public double[] DepensationS50 { get; init; } = [];

    /// <summary>The gate's steepness, 1 or more.</summary>
    public double[] DepensationTheta { get; init; } = [];
}

/// <summary>Outputs (bundle group <c>output</c>, spec 6.9).</summary>
public sealed class OutputParameters
{
    /// <summary>Per species (every species): schools younger than this many years are left out of
    /// biomass and abundance. 0 when unset.</summary>
    public double[] CutoffAge { get; init; } = [];

    public int RecordFrequency { get; init; } = 1;

    /// <summary>A record of the initial state, at step -1, precedes the run.</summary>
    public bool Step0Include { get; init; }

    /// <summary>Output families beyond the minimal set; none = <c>output.minimal.enabled</c>.</summary>
    public Step.OutputFamilies Families { get; init; }

    /// <summary>Per focal species, where its age classes start in a by-age record (focal + 1
    /// values); empty without a by-age family.</summary>
    public int[] AgeBinOffsets { get; init; } = [];

    /// <summary>The left edges of the length classes; empty without a by-size family.</summary>
    public double[] SizeEdges { get; init; } = [];

    /// <summary>Prey columns of the diet (focal species, then resources); 0 without the diet
    /// family.</summary>
    public int DietColumns { get; init; }

    /// <summary>Java-rule class thresholds of the trophic-level, age and length distributions;
    /// each empty without its family.</summary>
    public double[] TlThresholds { get; init; } = [];

    public double[] AgeThresholds { get; init; } = [];

    public double[] JavaSizeThresholds { get; init; } = [];

    /// <summary>The collector's layout for a model with these species counts.</summary>
    public Step.OutputLayout Layout(int nTotal, int nFocal) =>
        new(nTotal, nFocal, Families)
        {
            AgeBinOffsets = AgeBinOffsets,
            SizeEdges = SizeEdges,
            DietColumns = DietColumns,
            TlThresholds = TlThresholds,
            AgeThresholds = AgeThresholds,
            JavaSizeThresholds = JavaSizeThresholds,
        };

    /// <summary><c>step0 + ceil(n_steps / record_frequency)</c> (spec 8.4).</summary>
    public int Records(int nSteps) => (Step0Include ? 1 : 0) + ((nSteps + RecordFrequency - 1) / RecordFrequency);
}

/// <summary>Movement (bundle group <c>movement</c>, spec 6.11).</summary>
public sealed class MovementParameters
{
    public const byte MethodNone = 0;
    public const byte MethodRandom = 1;
    public const byte MethodMaps = 2;

    /// <summary>Per species (every species): none, random or maps.</summary>
    public byte[] Method { get; init; } = [];

    /// <summary>Walk range in cells, per species.</summary>
    public int[] WalkRange { get; init; } = [];

    /// <summary>The largest walk range, which sizes the candidate scratch: <c>(2r + 1)^2</c>.</summary>
    public int MaxWalkRange { get; init; }

    /// <summary>Number of stacked maps over all species.</summary>
    public int NMaps { get; init; }

    /// <summary><c>[map, y, x]</c>, row-major; a null map is all zeros with <see cref="MapIsNull"/> set.</summary>
    public double[] MapStack { get; init; } = [];

    /// <summary>0 for presence/absence maps, else the map's maximum (<c>MovementMapSet</c>).</summary>
    public double[] MapMaxProba { get; init; } = [];

    public byte[] MapIsNull { get; init; } = [];

    /// <summary>First stacked map of each focal species, or -1 for a species without maps.</summary>
    public int[] MapSpeciesOffset { get; init; } = [];

    /// <summary>Every focal species' <c>index_maps[age_dt, step]</c> table, concatenated row-major.</summary>
    public int[] MapIndex { get; init; } = [];

    /// <summary>Start of each species' table in <see cref="MapIndex"/>, length focal + 1.</summary>
    public long[] MapIndexOffsets { get; init; } = [];

    /// <summary><c>(n_ages, n_steps)</c> of each species' table, flattened.</summary>
    public int[] MapIndexShape { get; init; } = [];

    /// <summary>Start of each focal species' placement domain in <see cref="DomainCells"/>.</summary>
    public int[] DomainOffsets { get; init; } = [];

    /// <summary>Placement domains of random-method species: <c>(x, y)</c> pairs.</summary>
    public int[] DomainCells { get; init; } = [];

    /// <summary>The grid's ocean mask (1 ocean, 0 land), copied for the walk's hot loop.</summary>
    public byte[] Ocean { get; init; } = [];

    /// <summary>Feature <c>salinity_gate</c>: 1 for each gated focal species; empty without the
    /// feature.</summary>
    public byte[] SalinitySpecies { get; init; } = [];

    /// <summary><c>[frame * nCells + cell]</c>: the occupancy weight of each salinity frame
    /// (<c>salinity_weight</c>), NaN where the field is.</summary>
    public double[] SalinityWeight { get; init; } = [];

    /// <summary><c>[step]</c>: the salinity frame of each step.</summary>
    public int[] SalinityFrameByStep { get; init; } = [];

    /// <summary><c>[map * SalinityFrames + frame]</c>: the largest non-NaN <c>map * weight</c>,
    /// 0 when there is none (then the gate does not apply to that map this step).</summary>
    public double[] SalinityWmax { get; init; } = [];

    /// <summary>Number of salinity frames (rows of <see cref="SalinityWeight"/>).</summary>
    public int SalinityFrames { get; init; }
}

internal static class ModelReader
{
    public static SpeciesParameters Species(BundleReader r)
    {
        int nFocal = (int)r.Int("species.n_focal");
        int nBkg = (int)r.Int("species.n_background");
        int nTotal = Math.Max(0, nFocal + nBkg);
        return new SpeciesParameters
        {
            NFocal = nFocal,
            NBackground = nBkg,
            NSchools = I4(r, "species.n_schools", nFocal),
            LifespanDt = I4(r, "species.lifespan_dt", nTotal),
        };
    }

    public static GrowthParameters Growth(BundleReader r, int nFocal, int nTotal) => new()
    {
        Class = r.Array("growth.class", DType.U1, nFocal)?.ToArray<byte>() ?? [],
        Linf = F8(r, "growth.linf", nTotal),
        K = F8(r, "growth.k", nTotal),
        T0 = F8(r, "growth.t0", nTotal),
        EggSize = F8(r, "growth.egg_size", nTotal),
        VbThresholdAge = F8(r, "growth.vb_threshold_age", nTotal),
        ConditionFactor = F8(r, "growth.condition_factor", nTotal),
        AllometricPower = F8(r, "growth.allometric_power", nTotal),
        Lmax = F8(r, "growth.lmax", nTotal),
        DeltaLmaxFactor = F8(r, "growth.delta_lmax_factor", nTotal),
        CriticalSuccessRate = F8(r, "growth.critical_success_rate", nTotal),
    };

    public static PredationParameters Predation(BundleReader r, int nTotal)
    {
        BundleEntry? min = r.Array("predation.size_ratio_min", DType.F8, nTotal, BundleReader.Any);
        int nStage = min is null ? 0 : (int)min.Shape[1];
        return new PredationParameters
        {
            IngestionRate = F8(r, "predation.ingestion_rate", nTotal),
            FrShape = I4(r, "predation.fr_shape", nTotal),
            FrHalfSat = F8(r, "predation.fr_halfsat", nTotal),
            NStage = nStage,
            SizeRatioMin = min?.ToArray<double>() ?? [],
            SizeRatioMax = r.Array("predation.size_ratio_max", DType.F8, nTotal, nStage)?.ToArray<double>() ?? [],
            StageMetric = r.Array("predation.stage.metric", DType.U1, nTotal)?.ToArray<byte>() ?? [],
            StageOffsets = I4(r, "predation.stage.offsets", nTotal + 1),
            StageThresholds = r.Array("predation.stage.thresholds", DType.F8, BundleReader.Any)?.ToArray<double>() ?? [],
        };
    }

    public static AccessParameters Access(BundleReader r, int nTotal)
    {
        int mode = (int)r.Int("access.mode");
        if (mode == 0)
        {
            return new AccessParameters { Mode = 0 };
        }

        BundleEntry? m = r.Array("access.matrix", DType.F8, BundleReader.Any, BundleReader.Any);
        var p = new AccessParameters
        {
            Mode = mode,
            Matrix = m?.ToArray<double>() ?? [],
            Rows = m is null ? 0 : (int)m.Shape[0],
            Cols = m is null ? 0 : (int)m.Shape[1],
        };
        if (mode != 2)
        {
            return p;
        }

        return new AccessParameters
        {
            Mode = mode,
            Matrix = p.Matrix,
            Rows = p.Rows,
            Cols = p.Cols,
            PreyOffsets = I4(r, "access.prey.offsets", nTotal + 1),
            PreyThresholds = r.Array("access.prey.thresholds", DType.F8, BundleReader.Any)?.ToArray<double>() ?? [],
            PreyIndex = r.Array("access.prey.index", DType.I4, BundleReader.Any)?.ToArray<int>() ?? [],
            PredOffsets = I4(r, "access.pred.offsets", nTotal + 1),
            PredThresholds = r.Array("access.pred.thresholds", DType.F8, BundleReader.Any)?.ToArray<double>() ?? [],
            PredIndex = r.Array("access.pred.index", DType.I4, BundleReader.Any)?.ToArray<int>() ?? [],
        };
    }

    public static MortalityParameters Mortality(BundleReader r, int nSteps, int nFocal, int nTotal) => new()
    {
        AdditionalRateByStep = r.Array("mortality.additional.rate_by_step", DType.F8, nSteps, nTotal)?.ToArray<double>() ?? [],
        LarvaRateByStep = r.Array("mortality.larva.rate_by_step", DType.F8, nSteps, nFocal)?.ToArray<double>() ?? [],
        OutRate = F8(r, "mortality.out.rate", nTotal),
        StarvationRateMax = F8(r, "mortality.starvation.rate_max", nTotal),
    };

    public static FishingParameters Fishing(BundleReader r, SimulationConfig config, int nSteps, int nFocal, int nTotal, int ny, int nx) => new()
    {
        Enabled = r.Int("fishing.enabled") != 0,
        RateByStep = r.Array("fishing.rate_by_step", DType.F8, nSteps, nTotal)?.ToArray<double>() ?? [],
        SeasonByStep = config.TryGet("fishing.season_by_step", out _)
            ? r.Array("fishing.season_by_step", DType.F8, nSteps, nTotal)?.ToArray<double>()
            : null,
        SelectivityType = I4(r, "fishing.selectivity.type", nTotal),
        SelectivityA50 = F8(r, "fishing.selectivity.a50", nTotal),
        SelectivityL50 = F8(r, "fishing.selectivity.l50", nTotal),
        SelectivitySlope = F8(r, "fishing.selectivity.slope", nTotal),
        MapIndex = I4(r, "fishing.map_index", nFocal),
        Maps = r.Array("fishing.maps", DType.F8, BundleReader.Any, ny, nx)?.ToArray<double>() ?? [],
        DiscardRate = config.TryGet("fishing.discard_rate", out _) ? F8(r, "fishing.discard_rate", nTotal) : null,
        Fishery = config.TryGet("fishing.fishery.count", out _) ? Fishery(r, config, nSteps, nFocal, ny, nx) : null,
    };

    private static FisheryParameters Fishery(BundleReader r, SimulationConfig config, int nSteps, int nFocal, int ny, int nx)
    {
        int n = (int)r.Int("fishing.fishery.count");
        int ndt = (int)r.Int("time.n_dt_per_year");
        return new FisheryParameters
        {
            Count = n,
            Rate = F8(r, "fishing.fishery.rate", n),
            SelectivityType = I4(r, "fishing.fishery.selectivity.type", n),
            SelectivityA50 = F8(r, "fishing.fishery.selectivity.a50", n),
            SelectivityL50 = F8(r, "fishing.fishery.selectivity.l50", n),
            SelectivitySlope = F8(r, "fishing.fishery.selectivity.slope", n),
            Catchability = r.Array("fishing.fishery.catchability", DType.F8, nFocal, n)?.ToArray<double>() ?? [],
            Discard = r.Array("fishing.fishery.discard", DType.F8, nFocal, n)?.ToArray<double>() ?? [],
            Season = config.TryGet("fishing.fishery.season", out _)
                ? r.Array("fishing.fishery.season", DType.F8, n, ndt)?.ToArray<double>()
                : null,
            MapIndex = r.Array("fishing.fishery.map_index", DType.I4, n, nSteps)?.ToArray<int>() ?? [],
            NSteps = nSteps,
            Maps = r.Array("fishing.fishery.maps", DType.F8, BundleReader.Any, ny, nx)?.ToArray<double>() ?? [],
        };
    }

    public static ResourceParameters Resources(BundleReader r, int nSteps, int nCells, bool oxygen, bool depletable)
    {
        int n = (int)r.Int("resources.n");
        if (n <= 0)
        {
            return new ResourceParameters { N = 0 };
        }

        // As for oxygen: required with the feature, a builder error without it.
        double floor = 0.0;
        double[] regrowth = [];
        string[] depletableEntries = ["resources.depletable.floor", "resources.depletable.regrowth_rate"];
        if (depletable)
        {
            floor = r.Float(depletableEntries[0]);
            regrowth = F8(r, depletableEntries[1], n);
            if (r.Has(depletableEntries[0]) && !(floor >= 0.0 && floor <= 1.0))
            {
                r.Problem($"{depletableEntries[0]}: {floor} must be in [0, 1]");
            }

            if (regrowth.Any(rate => !(rate >= 0.0) || double.IsInfinity(rate)))
            {
                r.Problem($"{depletableEntries[1]}: values must be finite and >= 0");
            }
        }
        else
        {
            foreach (string name in depletableEntries.Where(r.Has))
            {
                r.Problem($"{name}: present, but the bundle does not declare feature depletable_ltl");
            }
        }

        // The oxygen entries exist exactly when the bundle declares oxygen_benthos: required with
        // it, a builder error without it (they would otherwise be silently ignored).
        byte[] coupled = [];
        double[] factor = [];
        int[] oxygenFrame = [];
        string[] oxygenEntries = ["resources.oxygen.coupled", "resources.oxygen.factor", "resources.oxygen.frame_by_step"];
        if (oxygen)
        {
            coupled = r.Array(oxygenEntries[0], DType.U1, n)?.ToArray<byte>() ?? [];
            BundleEntry? f = r.Array(oxygenEntries[1], DType.F8, BundleReader.Any, nCells);
            factor = f?.ToArray<double>() ?? [];
            oxygenFrame = r.Array(oxygenEntries[2], DType.I4, nSteps)?.ToArray<int>() ?? [];
            long frames = f?.Shape[0] ?? 0;
            if (f is not null && oxygenFrame.Any(i => i < 0 || i >= frames))
            {
                r.Problem($"{oxygenEntries[2]}: values must be in [0, {frames})");
            }
        }
        else
        {
            foreach (string name in oxygenEntries.Where(r.Has))
            {
                r.Problem($"{name}: present, but the bundle does not declare feature oxygen_benthos");
            }
        }

        BundleEntry? forcing = r.Array("resources.forcing", DType.F8, n, BundleReader.Any, nCells);
        return new ResourceParameters
        {
            OxygenCoupled = coupled,
            OxygenFactor = factor,
            OxygenFrameByStep = oxygenFrame,
            Depletable = depletable,
            DepletableFloor = floor,
            RegrowthRate = regrowth,
            N = n,
            SizeMin = F8(r, "resources.size_min", n),
            SizeMax = F8(r, "resources.size_max", n),
            TrophicLevel = F8(r, "resources.trophic_level", n),
            AccessRows = I4(r, "resources.access_rows", n),
            NFrames = forcing is null ? 0 : (int)forcing.Shape[1],
            Forcing = forcing?.ToArray<double>() ?? [],
            FrameByStep = r.Array("resources.frame_by_step", DType.I4, nSteps, n)?.ToArray<int>() ?? [],
            AccessByStep = r.Array("resources.accessibility_by_step", DType.F8, nSteps, n)?.ToArray<double>() ?? [],
        };
    }

    /// <summary>The recruitment entries follow the declared features: required with a feature,
    /// a problem without it (a curve, a linear seeding mode or a gate table the engine was not
    /// told about would otherwise be applied, or ignored, silently).</summary>
    public static ReproductionParameters Reproduction(
        BundleReader r, int nSteps, int nFocal, int ndt, bool curves, bool rvGate, bool seedingLinear, bool depensationGate)
    {
        byte[] srType = r.Array("reproduction.sr.type", DType.U1, nFocal)?.ToArray<byte>() ?? [];
        if (srType.Any(t => t > 4))
        {
            r.Problem("reproduction.sr.type: values must be 0 (none) to 4 (shepherd)");
        }

        double[] ssbHalf = [];
        double[] beta = [];
        if (curves)
        {
            ssbHalf = F8(r, "reproduction.sr.ssb_half", nFocal);
            beta = F8(r, "reproduction.sr.shepherd_beta", nFocal);
        }
        else
        {
            if (srType.Any(t => t != 0))
            {
                r.Problem("reproduction.sr.type: a curve is set, but the bundle does not declare feature stock_recruitment");
            }

            Undeclared(r, "stock_recruitment", "reproduction.sr.ssb_half", "reproduction.sr.shepherd_beta");
        }

        int seedingMode = (int)r.Int("reproduction.seeding_mode");
        if (seedingMode is < 0 or > 1)
        {
            r.Problem($"reproduction.seeding_mode: {seedingMode} must be 0 (stock_recruitment) or 1 (linear)");
        }
        else if (seedingMode == 1 && !seedingLinear)
        {
            r.Problem("reproduction.seeding_mode: linear, but the bundle does not declare feature seeding_linear");
        }

        byte[] rvEnabled = [];
        double[] rvFactor = [];
        if (rvGate)
        {
            int nYears = ndt > 0 ? (nSteps + ndt - 1) / ndt : 0;
            rvEnabled = r.Array("reproduction.rv.enabled", DType.U1, nFocal)?.ToArray<byte>() ?? [];
            rvFactor = r.Array("reproduction.rv.factor_by_year", DType.F8, nYears, nFocal)?.ToArray<double>() ?? [];
        }
        else
        {
            Undeclared(r, "rv_gate", "reproduction.rv.enabled", "reproduction.rv.factor_by_year");
        }

        byte[] depEnabled = [];
        double[] depS50 = [];
        double[] depTheta = [];
        if (depensationGate)
        {
            depEnabled = r.Array("reproduction.depensation.enabled", DType.U1, nFocal)?.ToArray<byte>() ?? [];
            depS50 = F8(r, "reproduction.depensation.s50", nFocal);
            depTheta = F8(r, "reproduction.depensation.theta", nFocal);
            int n = Math.Min(depEnabled.Length, Math.Min(depS50.Length, depTheta.Length));
            for (int sp = 0; sp < n; sp++)
            {
                // The Python loader refuses these too; a bundle built by hand must not get past.
                if (depEnabled[sp] != 0 && !(depS50[sp] > 0.0 && depTheta[sp] >= 1.0))
                {
                    r.Problem($"reproduction.depensation: species {sp} has s50 {depS50[sp]} and theta {depTheta[sp]}; a gated species needs s50 > 0 and theta >= 1");
                }
            }
        }
        else
        {
            Undeclared(r, "depensation_gate", "reproduction.depensation.enabled", "reproduction.depensation.s50", "reproduction.depensation.theta");
        }

        return Reproduction(r, nSteps, nFocal) with
        {
            SrType = srType,
            SrSsbHalf = ssbHalf,
            ShepherdBeta = beta,
            SeedingMode = seedingMode,
            RvEnabled = rvEnabled,
            RvFactorByYear = rvFactor,
            DepensationEnabled = depEnabled,
            DepensationS50 = depS50,
            DepensationTheta = depTheta,
        };
    }

    private static void Undeclared(BundleReader r, string feature, params string[] entries)
    {
        foreach (string name in entries.Where(r.Has))
        {
            r.Problem($"{name}: present, but the bundle does not declare feature {feature}");
        }
    }

    private static ReproductionParameters Reproduction(BundleReader r, int nSteps, int nFocal) => new()
    {
        SexRatio = F8(r, "reproduction.sex_ratio", nFocal),
        RelativeFecundity = F8(r, "reproduction.relative_fecundity", nFocal),
        MaturitySize = F8(r, "reproduction.maturity_size", nFocal),
        MaturityAgeDt = I4(r, "reproduction.maturity_age_dt", nFocal),
        SeedingBiomass = F8(r, "reproduction.seeding_biomass", nFocal),
        SeedingMaxStep = I4(r, "reproduction.seeding_max_step", nFocal),
        SeasonByStep = r.Array("reproduction.season_by_step", DType.F8, nSteps, nFocal)?.ToArray<double>() ?? [],
        EggLength = F8(r, "reproduction.egg_length", nFocal),
        EggWeight = F8(r, "reproduction.egg_weight", nFocal),
    };

    /// <summary><c>output.families</c> follows the declared feature <c>full_outputs</c>: with it
    /// the mask holds the life-stage family (every run that is not minimal has it) and only
    /// families this engine collects; without it the mask is 0. The two binning tables exist
    /// exactly with their families: <c>output.age_bin_offsets</c> with a by-age family,
    /// <c>output.size_edges</c> with a by-size one.</summary>
    public static OutputParameters Output(BundleReader r, int nTotal, int nFocal, bool feature, int nResources)
    {
        long families = r.Int("output.families");
        long unknown = families & ~(long)Step.OutputFamilies.Known;
        if (!feature && families != 0)
        {
            r.Problem($"output.families: {families}, but the bundle does not declare feature full_outputs");
        }
        else if (feature && families == 0)
        {
            r.Problem("output.families: full_outputs is declared, but output.families is 0");
        }
        else if (unknown != 0)
        {
            r.Problem($"output.families: unknown bits 0x{unknown:X}; this engine collects 0x{(long)Step.OutputFamilies.Known:X}");
        }
        else if (feature && (families & (long)Step.OutputFamilies.Stage) == 0)
        {
            r.Problem("output.families: the life-stage family (bit 0) is part of every run that is not minimal");
        }

        Step.OutputFamilies on = feature ? (Step.OutputFamilies)(families & (long)Step.OutputFamilies.Known) : Step.OutputFamilies.None;
        int[] ageOffsets = [];
        if ((on & Step.OutputFamilies.ByAge) != 0)
        {
            ageOffsets = I4(r, "output.age_bin_offsets", nFocal + 1);
            bool ok = ageOffsets.Length == nFocal + 1 && ageOffsets[0] == 0;
            for (int sp = 0; ok && sp < nFocal; sp++)
            {
                ok = ageOffsets[sp + 1] > ageOffsets[sp];
            }

            if (ageOffsets.Length == nFocal + 1 && !ok)
            {
                r.Problem("output.age_bin_offsets: must start at 0 and increase (every species has an age class)");
                ageOffsets = [];
            }
        }
        else if (r.Has("output.age_bin_offsets"))
        {
            r.Problem("output.age_bin_offsets: present, but output.families has no by-age family");
        }

        double[] edges = [];
        if ((on & Step.OutputFamilies.BySize) != 0)
        {
            edges = r.Array("output.size_edges", DType.F8, BundleReader.Any)?.ToArray<double>() ?? [];
            bool ok = edges.Length > 0 && !double.IsNaN(edges[0]);
            for (int i = 1; ok && i < edges.Length; i++)
            {
                ok = edges[i] > edges[i - 1];
            }

            if (r.Has("output.size_edges") && !ok)
            {
                r.Problem("output.size_edges: needs at least one edge, strictly increasing");
                edges = [];
            }
        }
        else if (r.Has("output.size_edges"))
        {
            r.Problem("output.size_edges: present, but output.families has no by-size family");
        }

        int dietColumns = 0;
        if ((on & Step.OutputFamilies.Diet) != 0)
        {
            dietColumns = (int)Math.Clamp(r.Int("output.diet_columns"), 0, int.MaxValue);

            // The kernel books a school prey at its species id and a resource in the LAST
            // columns. Any other count would put a resource under a species' label, or the
            // reverse, without a word.
            if (r.Has("output.diet_columns") && dietColumns != nTotal + nResources)
            {
                r.Problem($"output.diet_columns: {dietColumns}, but the prey columns are the {nTotal} species (focal, then background) and the {nResources} resources");
                dietColumns = 0;
            }
        }
        else if (r.Has("output.diet_columns"))
        {
            r.Problem("output.diet_columns: present, but output.families has no diet family");
        }

        // A family whose table is missing or broken is dropped here: the problem is reported, and
        // the layout stays one the collector can build.
        if (dietColumns == 0)
        {
            on &= ~Step.OutputFamilies.Diet;
        }

        if (ageOffsets.Length == 0)
        {
            on &= ~Step.OutputFamilies.ByAge;
        }

        if (edges.Length == 0)
        {
            on &= ~Step.OutputFamilies.BySize;
        }

        // Java-rule thresholds (osmose/engine/output_distrib.py), each exactly with its family.
        double[] Thresholds(string name, Step.OutputFamilies bit)
        {
            if ((on & bit) == 0)
            {
                if (r.Has(name))
                {
                    r.Problem($"{name}: present, but output.families has no family that uses it");
                }

                return [];
            }

            double[] t = r.Array(name, DType.F8, BundleReader.Any)?.ToArray<double>() ?? [];
            bool ok = t.Length > 0;
            for (int i = 1; ok && i < t.Length; i++)
            {
                ok = t[i] > t[i - 1];
            }

            if (!ok)
            {
                r.Problem($"{name}: needs at least one threshold, strictly increasing");
                on &= ~bit;
                return [];
            }

            return t;
        }

        double[] tlThr = Thresholds("output.tl_thresholds", Step.OutputFamilies.BiomassByTl);
        double[] ageThr = Thresholds("output.age_thresholds", Step.OutputFamilies.MeanSizeByAge);
        double[] jSizeThr = Thresholds("output.java_size_thresholds", Step.OutputFamilies.JavaSize);

        return new()
        {
            CutoffAge = F8(r, "output.cutoff_age", nTotal),
            RecordFrequency = (int)Math.Max(1, r.Int("output.record_frequency")),
            Step0Include = r.Int("output.step0_include") != 0,
            Families = on,
            AgeBinOffsets = ageOffsets,
            SizeEdges = edges,
            DietColumns = dietColumns,
            TlThresholds = tlThr,
            AgeThresholds = ageThr,
            JavaSizeThresholds = jSizeThr,
        };
    }

    /// <summary>The background group follows the declared feature: required with it; without
    /// it, background species in the bundle are a problem (they would never be injected).</summary>
    public static BackgroundParameters Background(BundleReader r, int nSteps, int nBackground, int nCells, bool feature)
    {
        string[] entries =
        [
            "background.class.offsets", "background.class.length", "background.class.weight",
            "background.class.trophic_level", "background.class.age_dt", "background.proportion_by_step",
            "background.ocean_cells", "background.base", "background.frame_by_step",
        ];
        if (!feature)
        {
            if (nBackground > 0)
            {
                r.Problem($"species.n_background: {nBackground}, but the bundle does not declare feature background_species");
            }

            Undeclared(r, "background_species", entries);
            return new BackgroundParameters();
        }

        if (nBackground <= 0)
        {
            r.Problem("contract.features: background_species is declared, but species.n_background is 0");
            return new BackgroundParameters();
        }

        int[] offsets = I4(r, entries[0], nBackground + 1);
        bool offsetsOk = offsets.Length == nBackground + 1 && offsets[0] == 0;
        for (int k = 0; offsetsOk && k < nBackground; k++)
        {
            offsetsOk = offsets[k + 1] > offsets[k];
        }

        if (offsets.Length == nBackground + 1 && !offsetsOk)
        {
            r.Problem($"{entries[0]}: must start at 0 and increase (every species has a class)");
        }

        int nClasses = offsetsOk ? offsets[^1] : 0;
        int[] ocean = r.Array(entries[6], DType.I4, BundleReader.Any)?.ToArray<int>() ?? [];
        if (ocean.Any(c => c < 0 || c >= nCells))
        {
            r.Problem($"{entries[6]}: values must be cells in [0, {nCells})");
        }

        BundleEntry? baseEntry = r.Array(entries[7], DType.F8, BundleReader.Any, ocean.Length);
        long nFrames = baseEntry?.Shape[0] ?? 0;
        int[] frame = r.Array(entries[8], DType.I4, nSteps, nBackground)?.ToArray<int>() ?? [];
        if (baseEntry is not null && frame.Any(f => f < 0 || f >= nFrames))
        {
            r.Problem($"{entries[8]}: values must be in [0, {nFrames})");
        }

        return new BackgroundParameters
        {
            NSpecies = nBackground,
            NClasses = nClasses,
            ClassOffsets = offsetsOk ? offsets : [0],
            ClassLength = F8(r, entries[1], nClasses),
            ClassWeight = F8(r, entries[2], nClasses),
            ClassTrophicLevel = F8(r, entries[3], nClasses),
            ClassAgeDt = I4(r, entries[4], nClasses),
            ProportionByStep = r.Array(entries[5], DType.F8, nSteps, nClasses)?.ToArray<double>() ?? [],
            OceanCells = ocean,
            Base = baseEntry?.ToArray<double>() ?? [],
            FrameByStep = frame,
        };
    }

    public static MovementParameters Movement(
        BundleReader r, int nSteps, int nFocal, int nTotal, int ny, int nx, byte[] ocean, bool salinityGate)
    {
        byte[] method = r.Array("movement.method", DType.U1, nTotal)?.ToArray<byte>() ?? [];
        int[] walk = I4(r, "movement.walk_range", nTotal);
        int nMaps = 0;
        double[] stack = [], maxP = [];
        byte[] isNull = [];
        int[] offset = Enumerable.Repeat(-1, nFocal).ToArray(), index = [], shape = new int[nFocal * 2];
        long[] indexOffsets = new long[nFocal + 1];
        if (r.Int("movement.maps.n_species") > 0)
        {
            BundleEntry? s = r.Array("movement.maps.stack", DType.F8, BundleReader.Any, ny, nx);
            nMaps = s is null ? 0 : (int)s.Shape[0];
            stack = s?.ToArray<double>() ?? [];
            maxP = F8(r, "movement.maps.max_proba", nMaps);
            isNull = r.Array("movement.maps.is_null", DType.U1, nMaps)?.ToArray<byte>() ?? [];
            offset = I4(r, "movement.maps.species_offset", nFocal);
            indexOffsets = r.Array("movement.maps.index.offsets", DType.I8, nFocal + 1)?.ToArray<long>() ?? indexOffsets;
            shape = r.Array("movement.maps.index.shape", DType.I4, nFocal, 2)?.ToArray<int>() ?? shape;
            index = I4(r, "movement.maps.index", indexOffsets[^1]);
        }

        // The salinity gate's tables follow the declared feature: required with it, a problem
        // without it.
        byte[] gateSpecies = [];
        double[] gateWeight = [], gateWmax = [];
        int[] gateFrame = [];
        int gateFrames = 0;
        if (salinityGate)
        {
            gateSpecies = r.Array("movement.salinity.species", DType.U1, nFocal)?.ToArray<byte>() ?? [];
            BundleEntry? w = r.Array("movement.salinity.weight", DType.F8, BundleReader.Any, ny * nx);
            gateFrames = w is null ? 0 : (int)w.Shape[0];
            gateWeight = w?.ToArray<double>() ?? [];
            gateFrame = I4(r, "movement.salinity.frame_by_step", nSteps);
            gateWmax = r.Array("movement.salinity.wmax", DType.F8, nMaps, gateFrames)?.ToArray<double>() ?? [];
            if (w is not null && gateFrame.Any(f => f < 0 || f >= gateFrames))
            {
                r.Problem($"movement.salinity.frame_by_step: values must be in [0, {gateFrames})");
            }
        }
        else
        {
            Undeclared(
                r,
                "salinity_gate",
                "movement.salinity.species",
                "movement.salinity.weight",
                "movement.salinity.frame_by_step",
                "movement.salinity.wmax");
        }

        int[] domainOffsets = I4(r, "movement.random.domain.offsets", nFocal + 1);
        long nDomain = domainOffsets.Length == nFocal + 1 ? domainOffsets[^1] : 0;
        return new MovementParameters
        {
            SalinitySpecies = gateSpecies,
            SalinityWeight = gateWeight,
            SalinityFrameByStep = gateFrame,
            SalinityWmax = gateWmax,
            SalinityFrames = gateFrames,
            Method = method,
            WalkRange = walk,
            MaxWalkRange = walk.Length == 0 ? 0 : Math.Max(0, walk.Max()),
            NMaps = nMaps,
            MapStack = stack,
            MapMaxProba = maxP,
            MapIsNull = isNull,
            MapSpeciesOffset = offset,
            MapIndex = index,
            MapIndexOffsets = indexOffsets,
            MapIndexShape = shape,
            DomainOffsets = domainOffsets,
            DomainCells = r.Array("movement.random.domain.cells", DType.I4, nDomain, 2)?.ToArray<int>() ?? [],
            Ocean = ocean,
        };
    }

    private static double[] F8(BundleReader r, string name, long n) => r.Array(name, DType.F8, n)?.ToArray<double>() ?? [];

    private static int[] I4(BundleReader r, string name, long n) => r.Array(name, DType.I4, n)?.ToArray<int>() ?? [];
}
