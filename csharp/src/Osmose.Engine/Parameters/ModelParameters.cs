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
}

/// <summary>Reads the model groups from a bundle through one <see cref="BundleReader"/>, so every
/// missing entry or wrong shape lands in the same problem list as the header's.</summary>
/// <summary>Reproduction (bundle group <c>reproduction</c>, focal species, spec 6.8).</summary>
public sealed class ReproductionParameters
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

    /// <summary>Stock-recruitment curve per species: 0 none (phase 1); others are phase 2.</summary>
    public byte[] SrType { get; init; } = [];

    /// <summary>0 stock_recruitment, 1 linear.</summary>
    public int SeedingMode { get; init; }
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

    /// <summary>Output families beyond the minimal set; 0 = <c>output.minimal.enabled</c>.</summary>
    public long Families { get; init; }

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
    };

    public static ResourceParameters Resources(BundleReader r, int nSteps, int nCells, bool oxygen)
    {
        int n = (int)r.Int("resources.n");
        if (n <= 0)
        {
            return new ResourceParameters { N = 0 };
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

    public static ReproductionParameters Reproduction(BundleReader r, int nSteps, int nFocal) => new()
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
        SrType = r.Array("reproduction.sr.type", DType.U1, nFocal)?.ToArray<byte>() ?? [],
        SeedingMode = (int)r.Int("reproduction.seeding_mode"),
    };

    public static OutputParameters Output(BundleReader r, int nTotal) => new()
    {
        CutoffAge = F8(r, "output.cutoff_age", nTotal),
        RecordFrequency = (int)Math.Max(1, r.Int("output.record_frequency")),
        Step0Include = r.Int("output.step0_include") != 0,
        Families = r.Int("output.families"),
    };

    public static MovementParameters Movement(BundleReader r, int nFocal, int nTotal, int ny, int nx, byte[] ocean)
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

        int[] domainOffsets = I4(r, "movement.random.domain.offsets", nFocal + 1);
        long nDomain = domainOffsets.Length == nFocal + 1 ? domainOffsets[^1] : 0;
        return new MovementParameters
        {
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
