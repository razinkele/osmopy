using Osmose.Engine.Rng;

namespace Osmose.Engine;

/// <summary>A bundle needs features this engine does not implement; every one is named.</summary>
public sealed class UnsupportedFeatureException : Exception
{
    /// <summary>Wraps the unsupported features.</summary>
    public UnsupportedFeatureException(IReadOnlyList<FeatureInfo> features)
        : base("the C# engine cannot run this bundle: " + string.Join("; ", features.Select(f => f.RefusalReason)))
    {
        Features = features;
    }

    /// <summary>The features the bundle needs and the engine lacks.</summary>
    public IReadOnlyList<FeatureInfo> Features { get; }
}

/// <summary>
/// The typed, validated parameters of one run, built ONCE from a <see cref="SimulationConfig"/> at
/// the start of <see cref="OsmoseEngine.RunSimulation"/> (spec section 8.2). After this no code
/// looks an entry up by name.
/// </summary>
/// <remarks>
/// Task 1.1 holds the header groups (contract, run, time, grid) and the feature check. The model
/// groups arrive with the bundle builder (task 1.2) and the tasks that use them.
/// </remarks>
public sealed class EngineParameters
{
    private EngineParameters()
    {
    }

    /// <summary>The bundle's contract version (its header), one of <see cref="FeatureCatalog.ContractVersions"/>.</summary>
    public int ContractVersion { get; private init; }

    /// <summary>The features the bundle declares it needs (<c>contract.features</c>, catalogue ids).</summary>
    public IReadOnlyList<FeatureInfo> Features { get; private init; } = [];

    /// <summary><c>contract.rng.scheme</c>; must equal <see cref="Philox.Scheme"/>.</summary>
    public int RngScheme { get; private init; }

    /// <summary><c>run.seed</c>.</summary>
    public long Seed { get; private init; }

    /// <summary><c>run.threads</c>, at least 1.</summary>
    public int Threads { get; private init; }

    /// <summary><c>run.deadline_seconds</c>; 0 means no deadline.</summary>
    public double DeadlineSeconds { get; private init; }

    /// <summary>Restart (osmose.engine.restart): the first step (after a restart file's step),
    /// the starting school table (<c>init.*</c>, empty without one), and where and when to
    /// write snapshots (<c>restart.dir</c> absent: none).</summary>
    public RestartParameters Restart { get; private init; } = new();

    /// <summary><c>run.checks</c>: debug invariants on.</summary>
    public bool Checks { get; private init; }

    /// <summary><c>time.n_steps</c>.</summary>
    public int NSteps { get; private init; }

    /// <summary><c>time.n_dt_per_year</c>.</summary>
    public int NDtPerYear { get; private init; }

    /// <summary><c>time.n_subdt</c>, the mortality sub-steps.</summary>
    public int NSubdt { get; private init; }

    /// <summary><c>grid.ny</c>.</summary>
    public int Ny { get; private init; }

    /// <summary><c>grid.nx</c>.</summary>
    public int Nx { get; private init; }

    /// <summary><c>grid.ocean_mask</c> (ny * nx, row-major), copied: 1 = ocean, 0 = land.</summary>
    public IReadOnlyList<byte> OceanMask { get; private init; } = [];

    public SpeciesParameters Species { get; private init; } = new();

    public GrowthParameters Growth { get; private init; } = new();

    public MovementParameters Movement { get; private init; } = new();

    /// <summary>Background species (feature <c>background_species</c>); empty without it.</summary>
    public BackgroundParameters Background { get; private init; } = new();

    public ReproductionParameters Reproduction { get; private init; } = new();

    public OutputParameters Output { get; private init; } = new();

    public PredationParameters Predation { get; private init; } = new();

    public AccessParameters Access { get; private init; } = new();

    public MortalityParameters Mortality { get; private init; } = new();

    public FishingParameters Fishing { get; private init; } = new();

    public ResourceParameters Resources { get; private init; } = new();

    /// <summary>
    /// Reads and validates the bundle in three stages: the header (every problem in one
    /// <see cref="BundleValidationException"/>), then the declared features (one
    /// <see cref="UnsupportedFeatureException"/> naming every unimplemented one, so a bundle the
    /// engine cannot run is refused before its model tables are examined), then the model groups
    /// (again every problem at once).
    /// </summary>
    public static EngineParameters From(SimulationConfig config) => From(config, checkFeatures: true);

    /// <summary>As <see cref="From(SimulationConfig)"/>; <paramref name="checkFeatures"/> false lets the
    /// test probes read a bundle whose features are not implemented yet.</summary>
    internal static EngineParameters From(SimulationConfig config, bool checkFeatures)
    {
        ArgumentNullException.ThrowIfNull(config);
        var r = new BundleReader(config, [.. FeatureCatalog.ContractVersions]);
        var problems = new List<string>();

        BundleEntry? featureEntry = r.Array("contract.features", DType.I4, BundleReader.Any);
        int rngScheme = (int)r.Int("contract.rng.scheme");
        long seed = r.Int("run.seed");
        long threads = r.Int("run.threads");
        double deadline = r.Float("run.deadline_seconds");
        long checks = r.Int("run.checks");
        long nSteps = r.Int("time.n_steps");
        long ndt = r.Int("time.n_dt_per_year");
        long nSubdt = r.Int("time.n_subdt");
        long ny = r.Int("grid.ny");
        long nx = r.Int("grid.nx");

        void Range(string name, long value, long min, long max)
        {
            if (value < min || value > max)
            {
                problems.Add($"{name}: {value} is outside [{min}, {max}]");
            }
        }

        if (config.TryGet("run.threads", out _))
        {
            Range("run.threads", threads, 1, 4096);
        }

        if (config.TryGet("time.n_steps", out _))
        {
            Range("time.n_steps", nSteps, 1, int.MaxValue);
        }

        if (config.TryGet("time.n_dt_per_year", out _))
        {
            Range("time.n_dt_per_year", ndt, 1, 365 * 24);
        }

        if (config.TryGet("time.n_subdt", out _))
        {
            Range("time.n_subdt", nSubdt, 1, ushort.MaxValue);
        }

        if (config.TryGet("run.checks", out _))
        {
            Range("run.checks", checks, 0, 1);
        }

        if (config.TryGet("grid.ny", out _))
        {
            Range("grid.ny", ny, 1, 100_000);
        }

        if (config.TryGet("grid.nx", out _))
        {
            Range("grid.nx", nx, 1, 100_000);
        }

        if (config.TryGet("run.deadline_seconds", out _) && !(deadline >= 0 && double.IsFinite(deadline)))
        {
            problems.Add($"run.deadline_seconds: {deadline} must be finite and >= 0 (0 = none)");
        }

        if (config.TryGet("contract.rng.scheme", out _) && rngScheme != Philox.Scheme)
        {
            problems.Add($"contract.rng.scheme: {rngScheme}, this engine implements scheme {Philox.Scheme}");
        }

        byte[] mask = [];
        if (ny >= 1 && nx >= 1 && ny <= 100_000 && nx <= 100_000)
        {
            BundleEntry? maskEntry = r.Array("grid.ocean_mask", DType.U1, ny, nx);
            if (maskEntry is not null)
            {
                mask = maskEntry.ToArray<byte>();
                if (mask.Any(b => b > 1))
                {
                    problems.Add("grid.ocean_mask: values must be 0 (land) or 1 (ocean)");
                }
            }
        }

        var features = new List<FeatureInfo>();
        if (featureEntry is not null)
        {
            var seen = new HashSet<int>();
            foreach (int id in featureEntry.AsSpan<int>())
            {
                if ((uint)id >= (uint)FeatureCatalog.All.Count)
                {
                    problems.Add($"contract.features: {id} is not a catalogue id (0..{FeatureCatalog.All.Count - 1})");
                }
                else if (!seen.Add(id))
                {
                    problems.Add($"contract.features: {FeatureCatalog.All[id].Name} listed twice");
                }
                else
                {
                    features.Add(FeatureCatalog.All[id]);
                }
            }

            if (!seen.Contains(FeatureCatalog.Core.Id))
            {
                problems.Add("contract.features: core is missing; a bundle builder always declares it");
            }
        }

        problems.InsertRange(0, r.Problems);
        if (problems.Count > 0)
        {
            throw new BundleValidationException(problems);
        }

        FeatureInfo[] unsupported = [.. features.Where(f => !f.Implemented).OrderBy(f => f.Id)];
        if (checkFeatures && unsupported.Length > 0)
        {
            throw new UnsupportedFeatureException(unsupported);
        }

        var model = new BundleReader(config, [.. FeatureCatalog.ContractVersions]);
        SpeciesParameters species = ModelReader.Species(model);
        int nTotal = species.NTotal;
        GrowthParameters growth = ModelReader.Growth(model, species.NFocal, nTotal);
        PredationParameters predation = ModelReader.Predation(model, nTotal);
        AccessParameters access = ModelReader.Access(model, nTotal);
        MortalityParameters mortality = ModelReader.Mortality(model, (int)nSteps, species.NFocal, nTotal);
        FishingParameters fishing = ModelReader.Fishing(model, config, (int)nSteps, species.NFocal, nTotal, (int)ny, (int)nx);
        bool oxygen = features.Contains(FeatureCatalog.Named("oxygen_benthos"));
        bool depletable = features.Contains(FeatureCatalog.Named("depletable_ltl"));
        ResourceParameters resources = ModelReader.Resources(model, (int)nSteps, (int)(ny * nx), oxygen, depletable);
        OutputParameters output = ModelReader.Output(model, nTotal, species.NFocal, features.Contains(FeatureCatalog.Named("full_outputs")), resources.N);
        ReproductionParameters reproduction = ModelReader.Reproduction(
            model,
            (int)nSteps,
            species.NFocal,
            (int)ndt,
            curves: features.Contains(FeatureCatalog.Named("stock_recruitment")),
            rvGate: features.Contains(FeatureCatalog.Named("rv_gate")),
            seedingLinear: features.Contains(FeatureCatalog.Named("seeding_linear")),
            depensationGate: features.Contains(FeatureCatalog.Named("depensation_gate")));
        MovementParameters movement = ModelReader.Movement(
            model,
            (int)nSteps,
            species.NFocal,
            nTotal,
            (int)ny,
            (int)nx,
            mask,
            salinityGate: features.Contains(FeatureCatalog.Named("salinity_gate")));
        BackgroundParameters background = ModelReader.Background(
            model,
            (int)nSteps,
            species.NBackground,
            (int)(ny * nx),
            feature: features.Contains(FeatureCatalog.Named("background_species")));
        RestartParameters restart = RestartParameters.Read(config, r, (int)nSteps);
        model.ThrowIfProblems();
        r.ThrowIfProblems();

        return new EngineParameters
        {
            Restart = restart,
            Background = background,
            Species = species,
            Growth = growth,
            Movement = movement,
            Reproduction = reproduction,
            Output = output,
            Predation = predation,
            Access = access,
            Mortality = mortality,
            Fishing = fishing,
            Resources = resources,
            ContractVersion = config.ContractVersion,
            Features = features,
            RngScheme = rngScheme,
            Seed = seed,
            Threads = (int)threads,
            DeadlineSeconds = deadline,
            Checks = checks != 0,
            NSteps = (int)nSteps,
            NDtPerYear = (int)ndt,
            NSubdt = (int)nSubdt,
            Ny = (int)ny,
            Nx = (int)nx,
            OceanMask = mask,
        };
    }
}

/// <summary>Restart settings and the starting school table (bundle groups <c>restart</c> and
/// <c>init</c>, all optional).</summary>
public sealed class RestartParameters
{
    public int StartStep { get; init; }

    public bool Write { get; init; }

    public int Frequency { get; init; }

    public int Spinup { get; init; }

    /// <summary>Directory for the snapshot bundles (Python turns them into Java's NetCDF);
    /// null when nothing is written.</summary>
    public string? Directory { get; init; }

    public int[] Species { get; init; } = [];

    public int[] AgeDt { get; init; } = [];

    public int[] CellX { get; init; } = [];

    public int[] CellY { get; init; } = [];

    public int[] FirstFeedingAgeDt { get; init; } = [];

    public byte[] IsEgg { get; init; } = [];

    public double[] Abundance { get; init; } = [];

    public double[] Length { get; init; } = [];

    public double[] Weight { get; init; } = [];

    public double[] TrophicLevel { get; init; } = [];

    /// <summary>Java's schedule: a snapshot after <paramref name="step"/> (the final one is
    /// added by the caller).</summary>
    public bool Due(int step) => Frequency > 0 && step >= Spinup - 1 && (step - Spinup + 1) % Frequency == 0;

    public static RestartParameters Read(SimulationConfig config, BundleReader r, int nSteps)
    {
        int start = config.TryGet("restart.start_step", out _) ? (int)r.Int("restart.start_step") : 0;
        if (start < 0 || start >= Math.Max(1, nSteps))
        {
            throw new ArgumentException($"restart.start_step: {start} is outside [0, {nSteps})");
        }

        string? dir = null;
        if (config.TryGet("restart.dir", out _))
        {
            byte[] bytes = r.Array("restart.dir", DType.U1, BundleReader.Any)?.ToArray<byte>() ?? [];
            dir = System.Text.Encoding.UTF8.GetString(bytes);
        }

        var p = new RestartParameters
        {
            StartStep = start,
            Write = config.TryGet("restart.write", out _) && r.Int("restart.write") != 0,
            Frequency = config.TryGet("restart.frequency", out _) ? (int)r.Int("restart.frequency") : 0,
            Spinup = config.TryGet("restart.spinup", out _) ? (int)r.Int("restart.spinup") : 0,
            Directory = dir,
        };
        if (!config.TryGet("init.species_id", out _))
        {
            return p;
        }

        long n = r.Array("init.species_id", DType.I4, BundleReader.Any)!.Length;
        return new RestartParameters
        {
            StartStep = p.StartStep,
            Write = p.Write,
            Frequency = p.Frequency,
            Spinup = p.Spinup,
            Directory = p.Directory,
            Species = r.Array("init.species_id", DType.I4, n)!.ToArray<int>(),
            AgeDt = r.Array("init.age_dt", DType.I4, n)!.ToArray<int>(),
            CellX = r.Array("init.cell_x", DType.I4, n)!.ToArray<int>(),
            CellY = r.Array("init.cell_y", DType.I4, n)!.ToArray<int>(),
            FirstFeedingAgeDt = r.Array("init.first_feeding_age_dt", DType.I4, n)!.ToArray<int>(),
            IsEgg = r.Array("init.is_egg", DType.U1, n)!.ToArray<byte>(),
            Abundance = r.Array("init.abundance", DType.F8, n)!.ToArray<double>(),
            Length = r.Array("init.length", DType.F8, n)!.ToArray<double>(),
            Weight = r.Array("init.weight", DType.F8, n)!.ToArray<double>(),
            TrophicLevel = r.Array("init.trophic_level", DType.F8, n)!.ToArray<double>(),
        };
    }
}
