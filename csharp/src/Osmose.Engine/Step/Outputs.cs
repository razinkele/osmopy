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

    /// <summary>Every family this engine can collect.</summary>
    Known = Stage,
}

/// <summary>The output families for every record (spec 8.4), owned by the run: the minimal set
/// always, the others (empty arrays otherwise) when <see cref="Families"/> has them.</summary>
public sealed class RunResults
{
    public RunResults(int nRecords, int nTotal, int nFocal, OutputFamilies families = OutputFamilies.None)
    {
        NRecords = nRecords;
        NTotal = nTotal;
        NFocal = nFocal;
        Families = families;
        RecordStep = new int[nRecords];
        WindowSteps = new int[nRecords];
        Biomass = new double[nRecords * nTotal];
        Abundance = new double[nRecords * nTotal];
        Mortality = new double[nRecords * nFocal * MortalityCauses];
        Yield = new double[nRecords * nFocal];
        bool stage = (families & OutputFamilies.Stage) != 0;
        AbundanceByStage = new double[stage ? nRecords * nFocal * OutputCollector.Stages : 0];
        DeathsByStage = new double[stage ? nRecords * nFocal * MortalityCauses * OutputCollector.Stages : 0];
        RatesByStage = new double[stage ? nRecords * nFocal * MortalityCauses * OutputCollector.Stages : 0];
    }

    /// <summary>Columns of <see cref="Mortality"/>, in <c>MortalityCause</c> order.</summary>
    public const int MortalityCauses = 8;

    public int NRecords { get; }

    public int NTotal { get; }

    public int NFocal { get; }

    public OutputFamilies Families { get; }

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

    /// <summary><c>[record, focal species, stage]</c>, the window mean.</summary>
    public double[] AbundanceByStage { get; }

    /// <summary><c>[record, focal species, cause, stage]</c>, the window sum of deaths.</summary>
    public double[] DeathsByStage { get; }

    /// <summary><c>[record, focal species, cause, stage]</c>, the window SUM of the per-step
    /// instantaneous rates (Java's convention): <c>inf</c> when a step emptied the stage.</summary>
    public double[] RatesByStage { get; }
}

/// <summary>
/// Collects each step's outputs (<c>_collect_outputs</c>, the <c>output.minimal</c> branch) and
/// folds a record window as <c>_average_step_outputs</c> does: biomass and abundance are window
/// means, mortality and yield window sums, and a one-step window passes through. The window keeps
/// every step's row so that the sums reproduce NumPy's: <c>np.sum(rows, axis=0)</c> adds the rows
/// in order, except when a row holds a single value, where the reduction axis is contiguous and
/// NumPy uses its pairwise sum (verified against NumPy on random data, 2026-11-18).
/// <para>With <see cref="OutputFamilies.Stage"/>, each step also has a life-stage snapshot
/// (<see cref="SnapshotStages(EngineParameters, SchoolStore)"/>), taken BEFORE reproduction so
/// that a death is binned by the stage the school held when it died. Its window rule: abundance
/// a mean, deaths a sum, and the rates a sum of the per-step rates.</para>
/// </summary>
public sealed class OutputCollector
{
    /// <summary>Life stages, in Java's order: Eggs, Juvenil, Adult.</summary>
    public const int Stages = 3;

    private const int DeathCells = RunResults.MortalityCauses * Stages;

    private readonly int _nTotal, _nFocal;
    private readonly bool _stage;
    private readonly double[] _biomass, _abundance, _mortality, _yield;
    private readonly double[] _stageAbundance, _stageDeaths, _stageRates;
    private int _k;
    private bool _stageTaken;

    public OutputCollector(int nTotal, int nFocal, int recordFrequency, OutputFamilies families = OutputFamilies.None)
    {
        _nTotal = nTotal;
        _nFocal = nFocal;
        _stage = (families & OutputFamilies.Stage) != 0;
        int rows = Math.Max(1, recordFrequency);
        _biomass = new double[rows * nTotal];
        _abundance = new double[rows * nTotal];
        _mortality = new double[rows * nFocal * RunResults.MortalityCauses];
        _yield = new double[rows * nFocal];
        _stageAbundance = new double[_stage ? rows * nFocal * Stages : 0];
        _stageDeaths = new double[_stage ? rows * nFocal * DeathCells : 0];
        _stageRates = new double[_stage ? rows * nFocal * DeathCells : 0];
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

        if (_k * _nTotal >= _biomass.Length)
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
        _stageTaken = true;
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
        Collect(p.Output.CutoffAge, p.NDtPerYear, store, backgroundBiomass, backgroundAbundance);

    /// <summary>As <see cref="Collect(EngineParameters, SchoolStore, ReadOnlySpan{double}, ReadOnlySpan{double})"/>,
    /// on the cutoff ages (years, per species) and the steps per year alone.</summary>
    public void Collect(ReadOnlySpan<double> cutoff, double ndt, SchoolStore store, ReadOnlySpan<double> backgroundBiomass, ReadOnlySpan<double> backgroundAbundance)
    {
        if (_k * _nTotal >= _biomass.Length)
        {
            throw new InvalidOperationException("the record window is full: flush before collecting");
        }

        if (_stage && !_stageTaken)
        {
            throw new InvalidOperationException("the life-stage snapshot was not taken for this step: call SnapshotStages before Collect");
        }

        Span<double> biomass = _biomass.AsSpan(_k * _nTotal, _nTotal);
        Span<double> abundance = _abundance.AsSpan(_k * _nTotal, _nTotal);
        Span<double> mortality = _mortality.AsSpan(_k * _nFocal * RunResults.MortalityCauses, _nFocal * RunResults.MortalityCauses);
        Span<double> yield = _yield.AsSpan(_k * _nFocal, _nFocal);
        biomass.Clear();
        abundance.Clear();
        mortality.Clear();
        yield.Clear();
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

            if (c.AgeDt / ndt >= cutoff[sp])
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

        _k++;
        _stageTaken = false;
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
        if (_stage)
        {
            int ws = _nFocal * Stages, wd = _nFocal * DeathCells;
            Fold(_stageAbundance, ws, mean: true, r.AbundanceByStage.AsSpan(record * ws, ws));
            Fold(_stageDeaths, wd, mean: false, r.DeathsByStage.AsSpan(record * wd, wd));
            Fold(_stageRates, wd, mean: false, r.RatesByStage.AsSpan(record * wd, wd));
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
