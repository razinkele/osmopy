using Osmose.Engine.Schools;

namespace Osmose.Engine.Step;

/// <summary>The output families beyond the minimal set (bundle <c>output.families</c>, a bit
/// mask; 0 is <c>output.minimal.enabled</c>). Feature <c>full_outputs</c>.</summary>
[Flags]
public enum OutputFamilies : long
{
    None = 0,

    /// <summary>The life-stage split (Eggs, Juvenil, Adult): abundance, deaths by cause and the
    /// per-step mortality rates. Part of every run that is not minimal.</summary>
    Stage = 1,

    /// <summary>Biomass by age class (years), every focal school, no cutoff.</summary>
    BiomassByAge = 2,

    AbundanceByAge = 4,

    /// <summary>Biomass by length class, every focal school, no cutoff.</summary>
    BiomassBySize = 8,

    AbundanceBySize = 16,

    /// <summary>Biomass-weighted mean trophic level; a species with no qualifying school is absent.</summary>
    MeanTl = 32,

    /// <summary>Fishing deaths in numbers.</summary>
    YieldN = 64,

    /// <summary>Abundance-weighted mean length; a species with no qualifying school is absent.</summary>
    MeanSize = 128,

    /// <summary>Spawning-stock biomass.</summary>
    Ssb = 256,

    /// <summary>The diet: biomass eaten per predator species (background ones included) and
    /// prey column (focal species, then resources). Predator pressure is derived from it.</summary>
    Diet = 512,

    /// <summary>Biomass by trophic-level class (Java <c>biomassDistribByTL</c>), every focal school,
    /// no cutoff, Java's class rule (<c>osmose/engine/output_distrib.py</c>).</summary>
    BiomassByTl = 1024,

    /// <summary>Mean length by age class: numerator <c>sum(A * L)</c> and denominator <c>sum(A)</c>,
    /// both window means, divided at write time (Java <c>WeightedDistribOutput</c>).</summary>
    MeanSizeByAge = 2048,

    /// <summary>Mean trophic level by length class: <c>sum(B * TL)</c> over <c>sum(B)</c>.</summary>
    MeanTlBySize = 4096,

    /// <summary>Catch (fishing deaths x weight) by Java size class, summed over the window.</summary>
    YieldBySize = 8192,

    /// <summary>Catch-weighted mean size: <c>sum(F * L)</c> and <c>sum(F)</c> per species, summed.</summary>
    SizeCatch = 16384,

    /// <summary>Catch-weighted mean TL: <c>sum(F * W * TL)</c> and <c>sum(F * W)</c>, summed.</summary>
    TlCatch = 32768,

    /// <summary>Per-step mortality rates by cause and Java size class, taken with the life-stage
    /// snapshot and summed over the window (Java MortalitySpeciesOutput on sizeDistrib).</summary>
    MortalityBySize = 65536,

    /// <summary>Landings and discards (t) by focal species and fishery, summed (Java
    /// FisheryOutputYield): all to the species' fishery on the reduced path, split by the step's
    /// catch shares (<see cref="OutputCollector.SnapshotFisheries"/>) on the per-fishery path.</summary>
    YieldByFishery = 131072,

    /// <summary>The three Java-rule distribution families.</summary>
    JavaDistributions = BiomassByTl | MeanSizeByAge | MeanTlBySize,

    /// <summary>The catch-derived families.</summary>
    CatchFamilies = YieldBySize | SizeCatch | TlCatch | MortalityBySize | YieldByFishery,

    /// <summary>Every family binned by <see cref="OutputLayout.JavaSizeThresholds"/>.</summary>
    JavaSize = MeanTlBySize | YieldBySize | MortalityBySize,

    ByAge = BiomassByAge | AbundanceByAge,
    BySize = BiomassBySize | AbundanceBySize,

    /// <summary>Every family this engine can collect.</summary>
    Known = Stage | ByAge | BySize | MeanTl | YieldN | MeanSize | Ssb | Diet | BiomassByTl | MeanSizeByAge | MeanTlBySize | CatchFamilies,
}

/// <summary>How a record window folds a family's per-step rows (<c>_average_step_outputs</c>).</summary>
public enum WindowFold
{
    /// <summary><c>np.mean(rows, axis=0)</c>.</summary>
    Mean,

    /// <summary><c>np.sum(rows, axis=0)</c>.</summary>
    Sum,

    /// <summary>Per value, the mean over the steps where it is present (not NaN); absent (NaN)
    /// when no step has it. <c>_avg_scalar_dict</c>: a dict that omits a species.</summary>
    MeanOfPresent,
}

/// <summary>One output family beyond the minimal set: its name (the result buffer is
/// <c>result.&lt;name&gt;</c> and the <c>StepOutput</c> field has the same name), the shape of one
/// record, its window rule, and the blocks NumPy reduces separately (one array per species for
/// the distributions; the whole record otherwise).</summary>
public sealed record FamilySpec(string Name, OutputFamilies Bit, WindowFold Fold, int[] Shape, int[] Blocks)
{
    public int Width { get; } = Shape.Aggregate(1, (a, b) => a * b);
}

/// <summary>What the collector needs to lay the families out: the species counts, the mask, and
/// the two tables the distributions are binned with.</summary>
public sealed record OutputLayout(int NTotal, int NFocal, OutputFamilies Families = OutputFamilies.None)
{
    /// <summary>Per focal species, where its age classes start in a by-age record (length
    /// <c>NFocal + 1</c>); species <c>s</c> has <c>ceil(lifespan / steps per year) + 1</c> of
    /// them. Needed with a by-age family.</summary>
    public int[] AgeBinOffsets { get; init; } = [];

    /// <summary>The left edges of the length classes. Needed with a by-size family.</summary>
    public double[] SizeEdges { get; init; } = [];

    /// <summary>Prey columns of the diet (bundle <c>output.diet_columns</c>): the focal
    /// species, then the resources the reference discovered. Needed with the diet family.</summary>
    public int DietColumns { get; init; }

    /// <summary>Java-rule class thresholds (float32 values) of the trophic-level, age and length
    /// distributions: a value's class is the count of thresholds at or below it.</summary>
    public double[] TlThresholds { get; init; } = [];

    public double[] AgeThresholds { get; init; } = [];

    public double[] JavaSizeThresholds { get; init; } = [];

    /// <summary>The number of fisheries of the yield-by-fishery family.</summary>
    public int FisheryCount { get; init; }

    /// <summary>The one fishery catching each focal species on the reduced (per-species) path,
    /// -1 for none; empty on the per-fishery path, where the catch is split by shares.</summary>
    public int[] FisheryOfSpecies { get; init; } = [];

    /// <summary>Every family name, on or off, in record order.</summary>
    public static readonly string[] AllNames =
    [
        "abundance_by_stage", "mortality_by_cause_stage", "mortality_rate_by_cause_stage",
        "biomass_by_age", "abundance_by_age", "biomass_by_size", "abundance_by_size",
        "mean_tl", "yield_n", "mean_size", "ssb",
        "biomass_by_tl", "meansize_byage_num", "meansize_byage_den", "meantl_bysize_num", "meantl_bysize_den",
        "yield_by_size", "size_catch_num", "size_catch_den", "tl_catch_num", "tl_catch_den", "mortality_rate_by_cause_size",
        "yield_by_fishery_landed", "yield_by_fishery_discarded",
        "diet_by_species",
    ];

    /// <summary>The families that are on, in the order of <see cref="AllNames"/>.</summary>
    public IReadOnlyList<FamilySpec> Specs()
    {
        const int stages = OutputCollector.Stages, causes = RunResults.MortalityCauses;
        var specs = new List<FamilySpec>();
        void Add(string name, OutputFamilies bit, WindowFold fold, int[] shape, int[]? blocks = null)
        {
            if ((Families & bit) != 0)
            {
                int width = shape.Aggregate(1, (a, b) => a * b);
                specs.Add(new FamilySpec(name, bit, fold, shape, blocks ?? [0, width]));
            }
        }

        if ((Families & OutputFamilies.ByAge) != 0 && (AgeBinOffsets.Length != NFocal + 1 || AgeBinOffsets[0] != 0))
        {
            throw new ArgumentException("a by-age family needs AgeBinOffsets (NFocal + 1 values, from 0)");
        }

        if ((Families & OutputFamilies.BySize) != 0 && SizeEdges.Length == 0)
        {
            throw new ArgumentException("a by-size family needs SizeEdges");
        }

        if ((Families & OutputFamilies.Diet) != 0 && DietColumns <= 0)
        {
            throw new ArgumentException("the diet family needs DietColumns");
        }

        int nSize = SizeEdges.Length;
        int[] sizeBlocks = [.. Enumerable.Range(0, NFocal + 1).Select(sp => sp * nSize)];
        Add("abundance_by_stage", OutputFamilies.Stage, WindowFold.Mean, [NFocal, stages]);
        Add("mortality_by_cause_stage", OutputFamilies.Stage, WindowFold.Sum, [NFocal, causes, stages]);
        Add("mortality_rate_by_cause_stage", OutputFamilies.Stage, WindowFold.Sum, [NFocal, causes, stages]);
        Add("biomass_by_age", OutputFamilies.BiomassByAge, WindowFold.Mean, [AgeBinOffsets.Length > 0 ? AgeBinOffsets[^1] : 0], AgeBinOffsets);
        Add("abundance_by_age", OutputFamilies.AbundanceByAge, WindowFold.Mean, [AgeBinOffsets.Length > 0 ? AgeBinOffsets[^1] : 0], AgeBinOffsets);
        Add("biomass_by_size", OutputFamilies.BiomassBySize, WindowFold.Mean, [NFocal, nSize], sizeBlocks);
        Add("abundance_by_size", OutputFamilies.AbundanceBySize, WindowFold.Mean, [NFocal, nSize], sizeBlocks);
        Add("mean_tl", OutputFamilies.MeanTl, WindowFold.MeanOfPresent, [NFocal]);
        Add("yield_n", OutputFamilies.YieldN, WindowFold.Sum, [NFocal]);
        Add("mean_size", OutputFamilies.MeanSize, WindowFold.MeanOfPresent, [NFocal]);
        Add("ssb", OutputFamilies.Ssb, WindowFold.Mean, [NFocal]);
        void AddJava(string name, OutputFamilies bit, double[] thresholds)
        {
            int n = thresholds.Length + 1;
            Add(name, bit, WindowFold.Mean, [NFocal, n], [.. Enumerable.Range(0, NFocal + 1).Select(sp => sp * n)]);
        }

        if ((Families & OutputFamilies.BiomassByTl) != 0 && TlThresholds.Length == 0)
        {
            throw new ArgumentException("biomass by TL needs TlThresholds");
        }

        if ((Families & OutputFamilies.MeanSizeByAge) != 0 && AgeThresholds.Length == 0)
        {
            throw new ArgumentException("mean size by age needs AgeThresholds");
        }

        if ((Families & OutputFamilies.JavaSize) != 0 && JavaSizeThresholds.Length == 0)
        {
            throw new ArgumentException("a Java size-class family needs JavaSizeThresholds");
        }

        if ((Families & OutputFamilies.MortalityBySize) != 0 && (Families & OutputFamilies.Stage) == 0)
        {
            throw new ArgumentException("mortality by size is taken with the life-stage snapshot: it needs the stage family");
        }

        AddJava("biomass_by_tl", OutputFamilies.BiomassByTl, TlThresholds);
        AddJava("meansize_byage_num", OutputFamilies.MeanSizeByAge, AgeThresholds);
        AddJava("meansize_byage_den", OutputFamilies.MeanSizeByAge, AgeThresholds);
        AddJava("meantl_bysize_num", OutputFamilies.MeanTlBySize, JavaSizeThresholds);
        AddJava("meantl_bysize_den", OutputFamilies.MeanTlBySize, JavaSizeThresholds);
        int nJSize = JavaSizeThresholds.Length + 1;
        Add("yield_by_size", OutputFamilies.YieldBySize, WindowFold.Sum, [NFocal, nJSize], [.. Enumerable.Range(0, NFocal + 1).Select(sp => sp * nJSize)]);
        Add("size_catch_num", OutputFamilies.SizeCatch, WindowFold.Sum, [NFocal]);
        Add("size_catch_den", OutputFamilies.SizeCatch, WindowFold.Sum, [NFocal]);
        Add("tl_catch_num", OutputFamilies.TlCatch, WindowFold.Sum, [NFocal]);
        Add("tl_catch_den", OutputFamilies.TlCatch, WindowFold.Sum, [NFocal]);
        Add("mortality_rate_by_cause_size", OutputFamilies.MortalityBySize, WindowFold.Sum, [NFocal, causes, nJSize]);
        if ((Families & OutputFamilies.YieldByFishery) != 0 && FisheryCount <= 0)
        {
            throw new ArgumentException("yield by fishery needs FisheryCount");
        }

        Add("yield_by_fishery_landed", OutputFamilies.YieldByFishery, WindowFold.Sum, [NFocal, FisheryCount]);
        Add("yield_by_fishery_discarded", OutputFamilies.YieldByFishery, WindowFold.Sum, [NFocal, FisheryCount]);
        Add("diet_by_species", OutputFamilies.Diet, WindowFold.Sum, [NTotal, DietColumns]); // last
        return specs;
    }
}

/// <summary>The output families for every record (spec 8.4), owned by the run: the minimal set
/// always, and the families of <see cref="Layout"/> in <see cref="Extra"/>.</summary>
public sealed class RunResults
{
    private readonly Dictionary<string, double[]> _extra = new(StringComparer.Ordinal);

    public RunResults(int nRecords, int nTotal, int nFocal, OutputFamilies families = OutputFamilies.None)
        : this(nRecords, new OutputLayout(nTotal, nFocal, families))
    {
    }

    public RunResults(int nRecords, OutputLayout layout)
    {
        NRecords = nRecords;
        Layout = layout;
        NTotal = layout.NTotal;
        NFocal = layout.NFocal;
        RecordStep = new int[nRecords];
        WindowSteps = new int[nRecords];
        Biomass = new double[nRecords * NTotal];
        Abundance = new double[nRecords * NTotal];
        Mortality = new double[nRecords * NFocal * MortalityCauses];
        Yield = new double[nRecords * NFocal];
        Specs = layout.Specs();
        foreach (FamilySpec spec in Specs)
        {
            _extra[spec.Name] = new double[nRecords * spec.Width];
        }
    }

    /// <summary>Columns of <see cref="Mortality"/>, in <c>MortalityCause</c> order. The ninth,
    /// SPAWNING (OSMOSE 4.4.x post-reproduction mortality), is never booked: this engine refuses
    /// semelparous species, so it stays zero, as the Python engine's does without them.</summary>
    public const int MortalityCauses = 9;

    public int NRecords { get; }

    public int NTotal { get; }

    public int NFocal { get; }

    public OutputLayout Layout { get; }

    public OutputFamilies Families => Layout.Families;

    /// <summary>The families beyond the minimal set that this run collects.</summary>
    public IReadOnlyList<FamilySpec> Specs { get; }

    /// <summary>Last step of each record's window; -1 for the initial record.</summary>
    public int[] RecordStep { get; }

    public int[] WindowSteps { get; }

    /// <summary><c>[record, species]</c>, the window mean, tonnes.</summary>
    public double[] Biomass { get; }

    /// <summary><c>[record, species]</c>, the window mean.</summary>
    public double[] Abundance { get; }

    /// <summary><c>[record, focal species, cause]</c>, the window sum of deaths.</summary>
    public double[] Mortality { get; }

    /// <summary><c>[record, focal species]</c>, the window sum, tonnes.</summary>
    public double[] Yield { get; }

    /// <summary>The records of a family, <c>[record * spec.Width + j]</c>; empty when it is off.</summary>
    public double[] Extra(string name) => _extra.TryGetValue(name, out double[]? v) ? v : [];

    /// <summary><c>[record, focal species, stage]</c>, the window mean.</summary>
    public double[] AbundanceByStage => Extra("abundance_by_stage");

    /// <summary><c>[record, focal species, cause, stage]</c>, the window sum of deaths.</summary>
    public double[] DeathsByStage => Extra("mortality_by_cause_stage");

    /// <summary><c>[record, focal species, cause, stage]</c>, the window SUM of the per-step
    /// instantaneous rates (Java's convention): <c>inf</c> when a step emptied the stage.</summary>
    public double[] RatesByStage => Extra("mortality_rate_by_cause_stage");
}

/// <summary>
/// Collects each step's outputs (<c>_collect_outputs</c>) and folds a record window as
/// <c>_average_step_outputs</c> does: biomass and abundance are window means, mortality and yield
/// window sums, and a one-step window passes through. The window keeps every step's row so that
/// the sums reproduce NumPy's: <c>np.sum(rows, axis=0)</c> adds the rows in order, except when a
/// row holds a single value, where the reduction axis is contiguous and NumPy uses its pairwise
/// sum (verified against NumPy on random data, 2026-11-18).
/// <para>With <see cref="OutputFamilies.Stage"/>, each step also has a life-stage snapshot
/// (<see cref="SnapshotStages(EngineParameters, SchoolStore)"/>), taken BEFORE reproduction so
/// that a death is binned by the stage the school held when it died. Its window rule: abundance
/// a mean, deaths a sum, and the rates a sum of the per-step rates.</para>
/// <para>The other families are collected with the minimal set, from the state after
/// reproduction: the distributions (every focal school, eggs included, no cutoff age), SSB,
/// yield in numbers, and the two weighted means, which a species with no qualifying school is
/// absent from (NaN).</para>
/// </summary>
public sealed class OutputCollector
{
    /// <summary>Life stages, in Java's order: Eggs, Juvenil, Adult.</summary>
    public const int Stages = 3;

    private const int DeathCells = RunResults.MortalityCauses * Stages;

    private readonly OutputLayout _layout;
    private readonly int _nTotal, _nFocal, _rowsCapacity;
    private readonly bool _stage, _full, _diet;
    private readonly double[] _biomass, _abundance, _mortality, _yield;
    private readonly IReadOnlyList<FamilySpec> _specs;
    private readonly Dictionary<string, double[]> _rows = new(StringComparer.Ordinal);
    private readonly double[] _stageAbundance, _stageDeaths, _stageRates;
    private readonly double[] _biomassByAge, _abundanceByAge, _biomassBySize, _abundanceBySize, _meanTl, _yieldN, _meanSize, _ssb;
    private readonly double[] _dietRows;
    private readonly double[] _biomassByTl, _sizeAgeNum, _sizeAgeDen, _tlSizeNum, _tlSizeDen;
    private readonly double[] _yieldBySize, _sizeCatchNum, _sizeCatchDen, _tlCatchNum, _tlCatchDen, _sizeRates;
    private readonly double[] _sizeAbundanceScratch, _sizeDeathsScratch; // one step, by Java size class
    private readonly double[] _yfLand, _yfDisc;
    private double[] _shareLand = [], _shareDisc = [], _termScratch = [];
    private int[] _shareSpecies = [];
    private int _shareCount = -1; // -1: no shares taken this step
    private readonly double[] _weighted, _weight; // scratch of the two weighted means, per focal species
    private readonly double[] _present; // scratch of a fold: one value per step of the window
    private int _k;
    private bool _stageTaken, _dietTaken;

    public OutputCollector(int nTotal, int nFocal, int recordFrequency, OutputFamilies families = OutputFamilies.None)
        : this(new OutputLayout(nTotal, nFocal, families), recordFrequency)
    {
    }

    public OutputCollector(OutputLayout layout, int recordFrequency)
    {
        _layout = layout;
        _nTotal = layout.NTotal;
        _nFocal = layout.NFocal;
        _stage = (layout.Families & OutputFamilies.Stage) != 0;
        _full = (layout.Families & ~(OutputFamilies.Stage | OutputFamilies.Diet)) != 0;
        _diet = (layout.Families & OutputFamilies.Diet) != 0;
        _rowsCapacity = Math.Max(1, recordFrequency);
        _biomass = new double[_rowsCapacity * _nTotal];
        _abundance = new double[_rowsCapacity * _nTotal];
        _mortality = new double[_rowsCapacity * _nFocal * RunResults.MortalityCauses];
        _yield = new double[_rowsCapacity * _nFocal];
        _specs = layout.Specs();
        foreach (FamilySpec spec in _specs)
        {
            _rows[spec.Name] = new double[_rowsCapacity * spec.Width];
        }

        double[] Rows(string name) => _rows.TryGetValue(name, out double[]? v) ? v : [];
        _stageAbundance = Rows("abundance_by_stage");
        _stageDeaths = Rows("mortality_by_cause_stage");
        _stageRates = Rows("mortality_rate_by_cause_stage");
        _biomassByAge = Rows("biomass_by_age");
        _biomassByTl = Rows("biomass_by_tl");
        _sizeAgeNum = Rows("meansize_byage_num");
        _sizeAgeDen = Rows("meansize_byage_den");
        _tlSizeNum = Rows("meantl_bysize_num");
        _tlSizeDen = Rows("meantl_bysize_den");
        _yieldBySize = Rows("yield_by_size");
        _sizeCatchNum = Rows("size_catch_num");
        _sizeCatchDen = Rows("size_catch_den");
        _tlCatchNum = Rows("tl_catch_num");
        _tlCatchDen = Rows("tl_catch_den");
        _sizeRates = Rows("mortality_rate_by_cause_size");
        _yfLand = Rows("yield_by_fishery_landed");
        _yfDisc = Rows("yield_by_fishery_discarded");
        int nJ = layout.JavaSizeThresholds.Length + 1;
        _sizeAbundanceScratch = _sizeRates.Length == 0 ? [] : new double[_nFocal * nJ];
        _sizeDeathsScratch = _sizeRates.Length == 0 ? [] : new double[_nFocal * RunResults.MortalityCauses * nJ];
        _abundanceByAge = Rows("abundance_by_age");
        _biomassBySize = Rows("biomass_by_size");
        _abundanceBySize = Rows("abundance_by_size");
        _meanTl = Rows("mean_tl");
        _yieldN = Rows("yield_n");
        _meanSize = Rows("mean_size");
        _ssb = Rows("ssb");
        _dietRows = Rows("diet_by_species");
        _weighted = new double[_full ? 2 * _nFocal : 0];
        _weight = new double[_full ? 2 * _nFocal : 0];
        _present = new double[_specs.Count > 0 ? _rowsCapacity : 0];
    }

    /// <summary>Steps collected into the open window.</summary>
    public int Pending => _k;

    /// <summary>
    /// The step's life-stage split (<c>_collect_by_life_stage</c>) and its per-step rates, into
    /// the open window's next row. Call it once per step, after aging and before reproduction
    /// (which increments ages and clears the egg flag), or just before
    /// <see cref="Collect(EngineParameters, SchoolStore)"/> for a state that no step produced
    /// (the initial record). Nothing to do without <see cref="OutputFamilies.Stage"/>.
    /// </summary>
    public void SnapshotStages(EngineParameters p, SchoolStore store) =>
        SnapshotStages(p.Reproduction.MaturitySize, p.Reproduction.MaturityAgeDt, store);

    /// <summary>As <see cref="SnapshotStages(EngineParameters, SchoolStore)"/>, on the maturity
    /// thresholds alone. A focal school is an egg while its egg flag is set, an adult when
    /// <c>length >= maturity size AND age >= maturity age</c> (the conjunction reproduction and
    /// SSB use), a juvenile otherwise. Sums run in store order.</summary>
    public void SnapshotStages(ReadOnlySpan<double> maturitySize, ReadOnlySpan<int> maturityAgeDt, SchoolStore store)
    {
        if (!_stage)
        {
            return;
        }

        if (_stageTaken)
        {
            throw new InvalidOperationException("the life-stage snapshot was already taken for this step");
        }

        if (_k >= _rowsCapacity)
        {
            throw new InvalidOperationException("the record window is full: flush before collecting");
        }

        Span<double> abundance = _stageAbundance.AsSpan(_k * _nFocal * Stages, _nFocal * Stages);
        Span<double> deaths = _stageDeaths.AsSpan(_k * _nFocal * DeathCells, _nFocal * DeathCells);
        abundance.Clear();
        deaths.Clear();
        ReadOnlySpan<SchoolCore> core = store.Core;
        ReadOnlySpan<SchoolLedger> ledger = store.Ledger;
        for (int i = 0; i < core.Length; i++)
        {
            SchoolCore c = core[i];
            int sp = c.SpeciesId;
            if (sp >= _nFocal || (c.Flags & SchoolFlags.IsBackground) != 0)
            {
                continue;
            }

            int stage = (c.Flags & SchoolFlags.IsEgg) != 0 ? 0
                : c.Length >= maturitySize[sp] && c.AgeDt >= maturityAgeDt[sp] ? 2 : 1;
            abundance[(sp * Stages) + stage] += c.Abundance;
            SchoolLedger l = ledger[i];
            Span<double> d = deaths.Slice((sp * DeathCells) + stage, DeathCells - stage);
            d[0 * Stages] += c.DeadPredation;
            d[1 * Stages] += l.DeadStarvation;
            d[2 * Stages] += l.DeadAdditional;
            d[3 * Stages] += l.DeadFishing;
            d[4 * Stages] += l.DeadOut;
            d[5 * Stages] += l.DeadForaging;
            d[6 * Stages] += l.DeadDiscards;
            d[7 * Stages] += l.DeadAging;
        }

        StageRates(abundance, deaths, _stageRates.AsSpan(_k * _nFocal * DeathCells, _nFocal * DeathCells), _nFocal);
        if (_sizeRates.Length > 0)
        {
            SnapshotSizes(store);
        }

        _stageTaken = true;
    }

    /// <summary>
    /// The step's diet by predator species (<c>aggregate_diet_by_species</c>), into the open
    /// window's next row: the per-school diet rows of the mortality step that just ran, added
    /// per predator species in store order. Call it once per step, right after mortality and
    /// BEFORE the background schools are removed (background predators have rows), or just
    /// before <see cref="Collect(EngineParameters, SchoolStore)"/> for a state that no step
    /// produced. Nothing to do without <see cref="OutputFamilies.Diet"/>.
    /// </summary>
    public void SnapshotDiet(SchoolStore store)
    {
        if (!_diet)
        {
            return;
        }

        if (_dietTaken)
        {
            throw new InvalidOperationException("the diet was already taken for this step");
        }

        if (_k >= _rowsCapacity)
        {
            throw new InvalidOperationException("the record window is full: flush before collecting");
        }

        int cols = _layout.DietColumns;
        if (store.DietCols != cols && store.Count > 0)
        {
            throw new InvalidOperationException($"the store tracks {store.DietCols} diet columns, the layout has {cols}");
        }

        Span<double> row = _dietRows.AsSpan(_k * _nTotal * cols, _nTotal * cols);
        row.Clear();
        ReadOnlySpan<SchoolCore> core = store.Core;
        ReadOnlySpan<double> diet = store.Diet;
        for (int i = 0; i < core.Length; i++)
        {
            int sp = core[i].SpeciesId;
            if (sp >= _nTotal)
            {
                continue;
            }

            ReadOnlySpan<double> eaten = diet.Slice(i * cols, cols);
            Span<double> into = row.Slice(sp * cols, cols);
            for (int j = 0; j < cols; j++)
            {
                into[j] += eaten[j];
            }
        }

        _dietTaken = true;
    }

    /// <summary>
    /// One step's additive instantaneous rates from its death counts
    /// (<c>mortality_rates_from_counts</c>). Per species and stage, with <c>D</c> the deaths
    /// summed over the causes IN CAUSE ORDER (NumPy reduces the engine's transposed, so
    /// non-contiguous, array one cause at a time: not its pairwise sum):
    /// <c>N_start = N_end + D</c>, <c>Z = -ln(N_end / N_start)</c> (survival 1 when
    /// <c>N_start</c> is 0), and <c>m_c = (D_c / D) * Z</c> for a cause that killed, exactly 0 for
    /// one that did not. A stage a step emptied therefore reads <c>inf</c>, and deaths too few to
    /// move the count read <c>-0.0</c>.
    /// </summary>
    /// <param name="abundance"><c>[species, stage]</c>, the survivors.</param>
    /// <param name="deaths"><c>[species, cause, stage]</c>.</param>
    /// <param name="rates"><c>[species, cause, stage]</c>.</param>
    public static void StageRates(ReadOnlySpan<double> abundance, ReadOnlySpan<double> deaths, Span<double> rates, int nSpecies)
    {
        for (int sp = 0; sp < nSpecies; sp++)
        {
            for (int stage = 0; stage < Stages; stage++)
            {
                int at = (sp * DeathCells) + stage;
                double total = deaths[at];
                for (int cause = 1; cause < RunResults.MortalityCauses; cause++)
                {
                    total += deaths[at + (cause * Stages)];
                }

                double end = abundance[(sp * Stages) + stage];
                double start = end + total;
                double z = -Math.Log(start > 0 ? end / start : 1.0);
                for (int cause = 0; cause < RunResults.MortalityCauses; cause++)
                {
                    double share = total > 0 ? deaths[at + (cause * Stages)] / total : 0.0;
                    rates[at + (cause * Stages)] = share > 0 ? share * z : 0.0;
                }
            }
        }
    }

    /// <summary>
    /// One step's outputs from the store (spec 6.9): biomass (<c>A * W</c>) and abundance of focal
    /// schools at or above the cutoff age; deaths by cause and yield (<c>dead_fishing * W</c>) of
    /// every focal school. Sums run in store order.
    /// </summary>
    public void Collect(EngineParameters p, SchoolStore store) => Collect(p, store, [], []);

    /// <summary>As <see cref="Collect(EngineParameters, SchoolStore)"/>, then the step's background
    /// totals (<c>_collect_background_outputs</c>): the biomass and abundance of the background
    /// species, indexed by species like the outputs, taken before those schools were removed and
    /// added WITHOUT the cutoff age. Empty spans add nothing.</summary>
    public void Collect(EngineParameters p, SchoolStore store, ReadOnlySpan<double> backgroundBiomass, ReadOnlySpan<double> backgroundAbundance) =>
        Collect(p.Output.CutoffAge, p.NDtPerYear, p.Reproduction.MaturitySize, p.Reproduction.MaturityAgeDt, store, backgroundBiomass, backgroundAbundance);

    /// <summary>As <see cref="Collect(EngineParameters, SchoolStore, ReadOnlySpan{double}, ReadOnlySpan{double})"/>,
    /// on the cutoff ages (years, per species) and the steps per year alone: enough for the
    /// minimal and the life-stage families.</summary>
    public void Collect(ReadOnlySpan<double> cutoff, int ndt, SchoolStore store, ReadOnlySpan<double> backgroundBiomass, ReadOnlySpan<double> backgroundAbundance) =>
        Collect(cutoff, ndt, [], [], store, backgroundBiomass, backgroundAbundance);

    /// <summary>As <see cref="Collect(EngineParameters, SchoolStore, ReadOnlySpan{double}, ReadOnlySpan{double})"/>,
    /// on the tables it reads: the cutoff ages, the steps per year, and the maturity thresholds
    /// (for SSB).</summary>
    public void Collect(
        ReadOnlySpan<double> cutoff,
        int ndt,
        ReadOnlySpan<double> maturitySize,
        ReadOnlySpan<int> maturityAgeDt,
        SchoolStore store,
        ReadOnlySpan<double> backgroundBiomass,
        ReadOnlySpan<double> backgroundAbundance)
    {
        if (_k >= _rowsCapacity)
        {
            throw new InvalidOperationException("the record window is full: flush before collecting");
        }

        if (_stage && !_stageTaken)
        {
            throw new InvalidOperationException("the life-stage snapshot was not taken for this step: call SnapshotStages before Collect");
        }

        if (_diet && !_dietTaken)
        {
            throw new InvalidOperationException("the diet was not taken for this step: call SnapshotDiet before Collect");
        }

        Span<double> biomass = _biomass.AsSpan(_k * _nTotal, _nTotal);
        Span<double> abundance = _abundance.AsSpan(_k * _nTotal, _nTotal);
        Span<double> mortality = _mortality.AsSpan(_k * _nFocal * RunResults.MortalityCauses, _nFocal * RunResults.MortalityCauses);
        Span<double> yield = _yield.AsSpan(_k * _nFocal, _nFocal);
        biomass.Clear();
        abundance.Clear();
        mortality.Clear();
        yield.Clear();
        double stepsPerYear = ndt;
        ReadOnlySpan<SchoolCore> core = store.Core;
        ReadOnlySpan<SchoolLedger> ledger = store.Ledger;
        for (int i = 0; i < core.Length; i++)
        {
            SchoolCore c = core[i];
            int sp = c.SpeciesId;
            if (sp >= _nFocal || (c.Flags & SchoolFlags.IsBackground) != 0)
            {
                continue;
            }

            if (c.AgeDt / stepsPerYear >= cutoff[sp])
            {
                biomass[sp] += c.Abundance * c.Weight;
                abundance[sp] += c.Abundance;
            }

            SchoolLedger l = ledger[i];
            Span<double> m = mortality.Slice(sp * RunResults.MortalityCauses, RunResults.MortalityCauses);
            m[0] += c.DeadPredation;
            m[1] += l.DeadStarvation;
            m[2] += l.DeadAdditional;
            m[3] += l.DeadFishing;
            m[4] += l.DeadOut;
            m[5] += l.DeadForaging;
            m[6] += l.DeadDiscards;
            m[7] += l.DeadAging;
            yield[sp] += l.DeadFishing * c.Weight;
        }

        for (int sp = _nFocal; sp < backgroundBiomass.Length; sp++)
        {
            biomass[sp] += backgroundBiomass[sp];
            abundance[sp] += backgroundAbundance[sp];
        }

        if (_full)
        {
            CollectFull(cutoff, ndt, maturitySize, maturityAgeDt, store);
        }

        _k++;
        _stageTaken = false;
        _dietTaken = false;
        _shareCount = -1;
    }

    /// <summary>
    /// The families collected with the minimal set, in a pass of their own so that a minimal run
    /// pays nothing for them:
    /// <list type="bullet">
    /// <item>by age (<c>_collect_distributions</c>): the class is <c>age // steps per year</c>,
    /// clipped to the species' classes; every focal school counts, eggs and empty schools too;</item>
    /// <item>by size: the class is the last edge at or below the length
    /// (<c>searchsorted(edges, length, side="right") - 1</c>), clipped to the classes;</item>
    /// <item>SSB (<c>_collect_ssb</c>): <c>A * W</c> of schools with
    /// <c>length >= maturity size AND age >= maturity age AND A > 0</c>;</item>
    /// <item>yield in numbers (<c>_collect_yield_n</c>): the fishing deaths;</item>
    /// <item>mean trophic level (<c>_collect_mean_tl</c>): <c>sum(B * TL) / sum(B)</c> over schools
    /// with <c>TL > 0</c>, <c>B > 0</c> and at or above the cutoff age;</item>
    /// <item>mean size (<c>_collect_mean_size</c>): <c>sum(A * L) / sum(A)</c> over schools with
    /// <c>A > 0</c>, <c>L > 0</c> and at or above the cutoff age.</item>
    /// </list>
    /// </summary>
    private void CollectFull(ReadOnlySpan<double> cutoff, int ndt, ReadOnlySpan<double> maturitySize, ReadOnlySpan<int> maturityAgeDt, SchoolStore store)
    {
        Span<double> Row(double[] rows, int width) => rows.Length == 0 ? default : rows.AsSpan(_k * width, width);
        int[] ageOffsets = _layout.AgeBinOffsets;
        ReadOnlySpan<double> edges = _layout.SizeEdges;
        int nAge = ageOffsets.Length > 0 ? ageOffsets[^1] : 0, nSize = edges.Length;
        Span<double> bAge = Row(_biomassByAge, nAge), aAge = Row(_abundanceByAge, nAge);
        Span<double> bSize = Row(_biomassBySize, _nFocal * nSize), aSize = Row(_abundanceBySize, _nFocal * nSize);
        Span<double> meanTl = Row(_meanTl, _nFocal), yieldN = Row(_yieldN, _nFocal), meanSize = Row(_meanSize, _nFocal), ssb = Row(_ssb, _nFocal);
        bAge.Clear();
        aAge.Clear();
        bSize.Clear();
        aSize.Clear();
        yieldN.Clear();
        ssb.Clear();
        Span<double> tlSum = _weighted.AsSpan(0, _nFocal), tlWeight = _weight.AsSpan(0, _nFocal);
        Span<double> sizeSum = _weighted.AsSpan(_nFocal, _nFocal), sizeWeight = _weight.AsSpan(_nFocal, _nFocal);
        _weighted.AsSpan().Clear();
        _weight.AsSpan().Clear();
        bool byAge = bAge.Length + aAge.Length > 0, bySize = bSize.Length + aSize.Length > 0;
        double[] tlThr = _layout.TlThresholds, ageThr = _layout.AgeThresholds, jSizeThr = _layout.JavaSizeThresholds;
        int nTl = tlThr.Length + 1, nAgeJ = ageThr.Length + 1, nSizeJ = jSizeThr.Length + 1;
        Span<double> bTl = Row(_biomassByTl, _nFocal * nTl);
        Span<double> saNum = Row(_sizeAgeNum, _nFocal * nAgeJ), saDen = Row(_sizeAgeDen, _nFocal * nAgeJ);
        Span<double> tsNum = Row(_tlSizeNum, _nFocal * nSizeJ), tsDen = Row(_tlSizeDen, _nFocal * nSizeJ);
        int nJ = jSizeThr.Length + 1;
        Span<double> ySize = Row(_yieldBySize, _nFocal * nJ);
        Span<double> scNum = Row(_sizeCatchNum, _nFocal), scDen = Row(_sizeCatchDen, _nFocal);
        Span<double> tcNum = Row(_tlCatchNum, _nFocal), tcDen = Row(_tlCatchDen, _nFocal);
        int nFsh = _layout.FisheryCount;
        int[] fos = _layout.FisheryOfSpecies;
        Span<double> yfL = Row(_yfLand, _nFocal * nFsh), yfD = Row(_yfDisc, _nFocal * nFsh);
        yfL.Clear();
        yfD.Clear();
        if (!yfL.IsEmpty && fos.Length == 0 && _shareCount < 0)
        {
            throw new InvalidOperationException("yield by fishery: call SnapshotFisheries before Collect on the per-fishery path");
        }

        ySize.Clear();
        scNum.Clear();
        scDen.Clear();
        tcNum.Clear();
        tcDen.Clear();
        bTl.Clear();
        saNum.Clear();
        saDen.Clear();
        tsNum.Clear();
        tsDen.Clear();
        if (!ssb.IsEmpty && maturitySize.Length < _nFocal)
        {
            throw new InvalidOperationException("SSB needs the maturity thresholds: use a Collect overload that takes them");
        }

        double stepsPerYear = ndt;
        ReadOnlySpan<SchoolCore> core = store.Core;
        ReadOnlySpan<SchoolLedger> ledger = store.Ledger;
        for (int i = 0; i < core.Length; i++)
        {
            SchoolCore c = core[i];
            int sp = c.SpeciesId;
            if (sp >= _nFocal || (c.Flags & SchoolFlags.IsBackground) != 0)
            {
                continue;
            }

            double b = c.Abundance * c.Weight;
            if (byAge)
            {
                int at = ageOffsets[sp] + Math.Clamp(c.AgeDt / ndt, 0, ageOffsets[sp + 1] - ageOffsets[sp] - 1);
                if (!bAge.IsEmpty)
                {
                    bAge[at] += b;
                }

                if (!aAge.IsEmpty)
                {
                    aAge[at] += c.Abundance;
                }
            }

            if (bySize)
            {
                int at = (sp * nSize) + SizeClass(edges, c.Length);
                if (!bSize.IsEmpty)
                {
                    bSize[at] += b;
                }

                if (!aSize.IsEmpty)
                {
                    aSize[at] += c.Abundance;
                }
            }

            // Yield by fishery (_collect_yield_by_fishery).
            if (!yfL.IsEmpty)
            {
                double dw = ledger[i].DeadFishing * c.Weight, dd = ledger[i].DeadDiscards * c.Weight;
                if (fos.Length > 0)
                {
                    int fsh = fos[sp];
                    if (fsh >= 0)
                    {
                        yfL[(sp * nFsh) + fsh] += dw;
                        yfD[(sp * nFsh) + fsh] += dd;
                    }
                }
                else if (i < _shareCount)
                {
                    if (_shareSpecies[i] != sp)
                    {
                        throw new InvalidOperationException("yield by fishery: school order changed since the catch shares");
                    }

                    for (int k = 0; k < nFsh; k++)
                    {
                        yfL[(sp * nFsh) + k] += dw * _shareLand[(i * nFsh) + k];
                        yfD[(sp * nFsh) + k] += dd * _shareDisc[(i * nFsh) + k];
                    }
                }
            }

            // Catch-derived families (_collect_catch_families): every focal school, no cutoff.
            double fDead = ledger[i].DeadFishing;
            if (!ySize.IsEmpty)
            {
                ySize[(sp * nJ) + JavaClass(jSizeThr, c.Length)] += fDead * c.Weight;
            }

            if (!scNum.IsEmpty)
            {
                scNum[sp] += fDead * c.Length;
                scDen[sp] += fDead;
            }

            if (!tcNum.IsEmpty)
            {
                double fBio = fDead * c.Weight;
                tcNum[sp] += fBio * c.TrophicLevel;
                tcDen[sp] += fBio;
            }

            // Java-rule families (_collect_java_distributions): every focal school, no cutoff.
            if (!bTl.IsEmpty)
            {
                bTl[(sp * nTl) + JavaClass(tlThr, c.TrophicLevel)] += b;
            }

            if (!saNum.IsEmpty)
            {
                int at = (sp * nAgeJ) + JavaClass(ageThr, c.AgeDt / (double)ndt);
                saNum[at] += c.Abundance * c.Length;
                saDen[at] += c.Abundance;
            }

            if (!tsNum.IsEmpty)
            {
                int at = (sp * nSizeJ) + JavaClass(jSizeThr, c.Length);
                tsNum[at] += b * c.TrophicLevel;
                tsDen[at] += b;
            }

            if (!ssb.IsEmpty && c.Length >= maturitySize[sp] && c.AgeDt >= maturityAgeDt[sp] && c.Abundance > 0)
            {
                ssb[sp] += b;
            }

            if (!yieldN.IsEmpty)
            {
                yieldN[sp] += ledger[i].DeadFishing;
            }

            if (c.AgeDt / stepsPerYear >= cutoff[sp])
            {
                if (c.TrophicLevel > 0 && b > 0)
                {
                    tlSum[sp] += b * c.TrophicLevel;
                    tlWeight[sp] += b;
                }

                if (c.Abundance > 0 && c.Length > 0)
                {
                    sizeSum[sp] += c.Abundance * c.Length;
                    sizeWeight[sp] += c.Abundance;
                }
            }
        }

        for (int sp = 0; sp < meanTl.Length; sp++)
        {
            meanTl[sp] = tlWeight[sp] > 0 ? tlSum[sp] / tlWeight[sp] : double.NaN;
        }

        for (int sp = 0; sp < meanSize.Length; sp++)
        {
            meanSize[sp] = sizeWeight[sp] > 0 ? sizeSum[sp] / sizeWeight[sp] : double.NaN;
        }
    }

    /// <summary>
    /// The step's catch shares by fishery (<c>_fishery_catch_shares</c>): per focal school, each
    /// fishery's share of its landed deaths, <c>term * (1 - d) / sum</c>, and of its discarded
    /// ones, <c>term * d / sum</c>. Call it right after mortality and the background removal,
    /// before growth changes the lengths the rates saw; the store's order must then hold until
    /// <see cref="Collect(EngineParameters, SchoolStore)"/> (reproduction only appends). Nothing
    /// to do without the family or on the reduced path.
    /// </summary>
    public void SnapshotFisheries(EngineParameters p, SchoolStore store, int t)
    {
        if (_yfLand.Length == 0 || _layout.FisheryOfSpecies.Length > 0 || p.Fishing.Fishery is not { } fm)
        {
            return;
        }

        int nF = fm.Count, n = store.Count;
        if (_shareLand.Length < n * nF)
        {
            _shareLand = new double[n * nF * 2];
            _shareDisc = new double[n * nF * 2];
        }

        if (_shareSpecies.Length < n)
        {
            _shareSpecies = new int[n * 2];
        }

        if (_termScratch.Length < nF)
        {
            _termScratch = new double[nF];
        }

        ReadOnlySpan<SchoolCore> core = store.Core;
        ReadOnlySpan<SchoolLife> life = store.Life;
        Span<double> terms = _termScratch.AsSpan(0, nF);
        for (int i = 0; i < n; i++)
        {
            SchoolCore c = core[i];
            int sp = c.SpeciesId;
            _shareSpecies[i] = sp;
            Span<double> land = _shareLand.AsSpan(i * nF, nF), disc = _shareDisc.AsSpan(i * nF, nF);
            land.Clear();
            disc.Clear();
            if (sp >= _nFocal || (c.Flags & SchoolFlags.IsBackground) != 0)
            {
                continue;
            }

            StepPreparation.FisheryTerms(p, fm, c, life[i].CellId, Math.Max(t, 0), terms);
            double tl = 0.0, td = 0.0;
            for (int k = 0; k < nF; k++)
            {
                double d = fm.Discard[(sp * nF) + k];
                land[k] = terms[k] * (1.0 - d);
                disc[k] = terms[k] * d;
                tl += land[k];
                td += disc[k];
            }

            for (int k = 0; k < nF; k++)
            {
                land[k] = tl > 0 ? land[k] / tl : 0.0;
                disc[k] = td > 0 ? disc[k] / td : 0.0;
            }
        }

        _shareCount = n;
    }

    /// <summary>The mortality-by-size split (<c>_collect_by_java_size</c>): every focal school by
    /// its Java size class at the snapshot, then the per-step rates by
    /// <see cref="ClassRates"/>.</summary>
    private void SnapshotSizes(SchoolStore store)
    {
        double[] thr = _layout.JavaSizeThresholds;
        int nCls = thr.Length + 1;
        int cells = RunResults.MortalityCauses * nCls;
        Span<double> abundance = _sizeAbundanceScratch;
        Span<double> deaths = _sizeDeathsScratch;
        abundance.Clear();
        deaths.Clear();
        ReadOnlySpan<SchoolCore> core = store.Core;
        ReadOnlySpan<SchoolLedger> ledger = store.Ledger;
        for (int i = 0; i < core.Length; i++)
        {
            SchoolCore c = core[i];
            int sp = c.SpeciesId;
            if (sp >= _nFocal || (c.Flags & SchoolFlags.IsBackground) != 0)
            {
                continue;
            }

            int cls = JavaClass(thr, c.Length);
            abundance[(sp * nCls) + cls] += c.Abundance;
            SchoolLedger l = ledger[i];
            Span<double> d = deaths.Slice((sp * cells) + cls);
            d[0 * nCls] += c.DeadPredation;
            d[1 * nCls] += l.DeadStarvation;
            d[2 * nCls] += l.DeadAdditional;
            d[3 * nCls] += l.DeadFishing;
            d[4 * nCls] += l.DeadOut;
            d[5 * nCls] += l.DeadForaging;
            d[6 * nCls] += l.DeadDiscards;
            d[7 * nCls] += l.DeadAging;
        }

        ClassRates(abundance, deaths, _sizeRates.AsSpan(_k * _nFocal * cells, _nFocal * cells), _nFocal, nCls);
    }

    /// <summary><see cref="StageRates"/> for any number of classes: per species and class,
    /// <c>N_start = N_end + deaths</c>, <c>Z = -ln(N_end / N_start)</c>, each cause its share of Z
    /// (<c>mortality_rates_from_counts</c>).</summary>
    public static void ClassRates(ReadOnlySpan<double> abundance, ReadOnlySpan<double> deaths, Span<double> rates, int nSpecies, int nCls)
    {
        int cells = RunResults.MortalityCauses * nCls;
        for (int sp = 0; sp < nSpecies; sp++)
        {
            for (int cls = 0; cls < nCls; cls++)
            {
                int at = (sp * cells) + cls;
                double total = deaths[at];
                for (int cause = 1; cause < RunResults.MortalityCauses; cause++)
                {
                    total += deaths[at + (cause * nCls)];
                }

                double end = abundance[(sp * nCls) + cls];
                double start = end + total;
                double z = -Math.Log(start > 0 ? end / start : 1.0);
                for (int cause = 0; cause < RunResults.MortalityCauses; cause++)
                {
                    double share = total > 0 ? deaths[at + (cause * nCls)] / total : 0.0;
                    rates[at + (cause * nCls)] = share > 0 ? share * z : 0.0;
                }
            }
        }
    }

    /// <summary>Java's <c>OutputDistribution.getClass</c>: the count of thresholds at or below
    /// <paramref name="value"/> (<c>np.searchsorted(thresholds, value, side="right")</c>).</summary>
    public static int JavaClass(ReadOnlySpan<double> thresholds, double value)
    {
        int lo = 0, hi = thresholds.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >>> 1;
            if (thresholds[mid] <= value)
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid;
            }
        }

        return lo;
    }

    /// <summary>The length class of <paramref name="length"/>: the last edge at or below it
    /// (<c>np.searchsorted(edges, length, side="right") - 1</c>), clipped to the classes, so a
    /// length below the first edge falls in the first class and one on an edge in the class that
    /// edge opens.</summary>
    public static int SizeClass(ReadOnlySpan<double> edges, double length)
    {
        int lo = 0, hi = edges.Length; // the count of edges <= length
        while (lo < hi)
        {
            int mid = (lo + hi) >>> 1;
            if (edges[mid] <= length)
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid;
            }
        }

        return Math.Clamp(lo - 1, 0, edges.Length - 1);
    }

    /// <summary>Folds the open window into record <paramref name="record"/> and empties it.</summary>
    public void Flush(RunResults r, int record, int recordStep)
    {
        if (_k == 0)
        {
            throw new InvalidOperationException("nothing to flush");
        }

        Fold(_biomass, _nTotal, mean: true, r.Biomass.AsSpan(record * _nTotal, _nTotal));
        Fold(_abundance, _nTotal, mean: true, r.Abundance.AsSpan(record * _nTotal, _nTotal));
        int w = _nFocal * RunResults.MortalityCauses;
        Fold(_mortality, w, mean: false, r.Mortality.AsSpan(record * w, w));
        Fold(_yield, _nFocal, mean: false, r.Yield.AsSpan(record * _nFocal, _nFocal));
        foreach (FamilySpec spec in _specs)
        {
            Span<double> into = r.Extra(spec.Name).AsSpan(record * spec.Width, spec.Width);
            double[] rows = _rows[spec.Name];
            if (spec.Fold == WindowFold.MeanOfPresent)
            {
                FoldPresent(rows, _k, spec.Width, _present, into);
                continue;
            }

            for (int b = 0; b + 1 < spec.Blocks.Length; b++)
            {
                FoldBlock(rows, _k, spec.Width, spec.Blocks[b], spec.Blocks[b + 1] - spec.Blocks[b], spec.Fold == WindowFold.Mean, _present, into);
            }
        }

        r.RecordStep[record] = recordStep;
        r.WindowSteps[record] = _k;
        _k = 0;
    }

    private void Fold(double[] rows, int width, bool mean, Span<double> into) => FoldWindow(rows, _k, width, mean, into);

    /// <summary>
    /// Folds <paramref name="k"/> rows of <paramref name="width"/> values (<c>[t * width + j]</c>)
    /// into one, as <c>_average_step_outputs</c>: a single row passes through; otherwise NumPy's
    /// axis-0 sum (pairwise for one-value rows, in order otherwise), divided by k for a mean.
    /// </summary>
    public static void FoldWindow(ReadOnlySpan<double> rows, int k, int width, bool mean, Span<double> into)
    {
        if (k == 1)
        {
            rows[..width].CopyTo(into); // a one-step window passes through
            return;
        }

        if (width == 1)
        {
            double s = PairwiseSum(rows[..k]);
            into[0] = mean ? s / k : s;
            return;
        }

        for (int j = 0; j < width; j++)
        {
            double s = rows[j];
            for (int t = 1; t < k; t++)
            {
                s += rows[(t * width) + j];
            }

            into[j] = mean ? s / k : s;
        }
    }

    /// <summary>
    /// As <see cref="FoldWindow"/>, for one block of a wider row: the <paramref name="length"/>
    /// values from <paramref name="start"/> of each of the <paramref name="k"/> rows of
    /// <paramref name="width"/>. A block is one array on the Python side (one species'
    /// distribution, or a whole record), so a block of ONE value is summed pairwise like a
    /// one-value row. <paramref name="scratch"/> holds at least k values.
    /// </summary>
    public static void FoldBlock(ReadOnlySpan<double> rows, int k, int width, int start, int length, bool mean, Span<double> scratch, Span<double> into)
    {
        if (k == 1)
        {
            rows.Slice(start, length).CopyTo(into.Slice(start, length));
            return;
        }

        if (length == 1)
        {
            for (int t = 0; t < k; t++)
            {
                scratch[t] = rows[(t * width) + start];
            }

            double s = PairwiseSum(scratch[..k]);
            into[start] = mean ? s / k : s;
            return;
        }

        for (int j = start; j < start + length; j++)
        {
            double s = rows[j];
            for (int t = 1; t < k; t++)
            {
                s += rows[(t * width) + j];
            }

            into[j] = mean ? s / k : s;
        }
    }

    /// <summary>
    /// Folds a family that a species can be absent from (NaN), as <c>_avg_scalar_dict</c>: per
    /// value, <c>np.mean</c> of the steps that have it (a 1-D array, so the pairwise sum, divided
    /// by their number), and NaN when no step has it. A single row passes through.
    /// </summary>
    public static void FoldPresent(ReadOnlySpan<double> rows, int k, int width, Span<double> scratch, Span<double> into)
    {
        if (k == 1)
        {
            rows[..width].CopyTo(into);
            return;
        }

        for (int j = 0; j < width; j++)
        {
            int m = 0;
            for (int t = 0; t < k; t++)
            {
                double v = rows[(t * width) + j];
                if (!double.IsNaN(v))
                {
                    scratch[m++] = v;
                }
            }

            into[j] = m == 0 ? double.NaN : PairwiseSum(scratch[..m]) / m;
        }
    }

    /// <summary>NumPy's <c>pairwise_sum</c> (<c>loops_utils.h</c>): sequential from 0.0 below 8
    /// values, eight accumulators up to 128, halves (on multiples of 8) above.</summary>
    public static double PairwiseSum(ReadOnlySpan<double> a)
    {
        int n = a.Length;
        if (n < 8)
        {
            double res = 0.0;
            for (int i = 0; i < n; i++)
            {
                res += a[i];
            }

            return res;
        }

        if (n <= 128)
        {
            double r0 = a[0], r1 = a[1], r2 = a[2], r3 = a[3], r4 = a[4], r5 = a[5], r6 = a[6], r7 = a[7];
            int i = 8;
            for (; i < n - (n % 8); i += 8)
            {
                r0 += a[i];
                r1 += a[i + 1];
                r2 += a[i + 2];
                r3 += a[i + 3];
                r4 += a[i + 4];
                r5 += a[i + 5];
                r6 += a[i + 6];
                r7 += a[i + 7];
            }

            double res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
            for (; i < n; i++)
            {
                res += a[i];
            }

            return res;
        }

        int n2 = n / 2;
        n2 -= n2 % 8;
        return PairwiseSum(a[..n2]) + PairwiseSum(a[n2..]);
    }
}
