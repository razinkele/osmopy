using Osmose.Engine.Schools;

namespace Osmose.Engine.Step;

/// <summary>The minimal output families for every record (spec 8.4), owned by the run.</summary>
public sealed class RunResults
{
    public RunResults(int nRecords, int nTotal, int nFocal)
    {
        NRecords = nRecords;
        NTotal = nTotal;
        NFocal = nFocal;
        RecordStep = new int[nRecords];
        WindowSteps = new int[nRecords];
        Biomass = new double[nRecords * nTotal];
        Abundance = new double[nRecords * nTotal];
        Mortality = new double[nRecords * nFocal * MortalityCauses];
        Yield = new double[nRecords * nFocal];
    }

    /// <summary>Columns of <see cref="Mortality"/>, in <c>MortalityCause</c> order.</summary>
    public const int MortalityCauses = 8;

    public int NRecords { get; }

    public int NTotal { get; }

    public int NFocal { get; }

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
}

/// <summary>
/// Collects each step's outputs (<c>_collect_outputs</c>, the <c>output.minimal</c> branch) and
/// folds a record window as <c>_average_step_outputs</c> does: biomass and abundance are window
/// means, mortality and yield window sums, and a one-step window passes through. The window keeps
/// every step's row so that the sums reproduce NumPy's: <c>np.sum(rows, axis=0)</c> adds the rows
/// in order, except when a row holds a single value, where the reduction axis is contiguous and
/// NumPy uses its pairwise sum (verified against NumPy on random data, 2026-11-18).
/// </summary>
public sealed class OutputCollector
{
    private readonly int _nTotal, _nFocal;
    private readonly double[] _biomass, _abundance, _mortality, _yield;
    private int _k;

    public OutputCollector(int nTotal, int nFocal, int recordFrequency)
    {
        _nTotal = nTotal;
        _nFocal = nFocal;
        int rows = Math.Max(1, recordFrequency);
        _biomass = new double[rows * nTotal];
        _abundance = new double[rows * nTotal];
        _mortality = new double[rows * nFocal * RunResults.MortalityCauses];
        _yield = new double[rows * nFocal];
    }

    /// <summary>Steps collected into the open window.</summary>
    public int Pending => _k;

    /// <summary>
    /// One step's outputs from the store (spec 6.9): biomass (<c>A * W</c>) and abundance of focal
    /// schools at or above the cutoff age; deaths by cause and yield (<c>dead_fishing * W</c>) of
    /// every focal school. Sums run in store order.
    /// </summary>
    public void Collect(EngineParameters p, SchoolStore store)
    {
        if (_k * _nTotal >= _biomass.Length)
        {
            throw new InvalidOperationException("the record window is full: flush before collecting");
        }

        Span<double> biomass = _biomass.AsSpan(_k * _nTotal, _nTotal);
        Span<double> abundance = _abundance.AsSpan(_k * _nTotal, _nTotal);
        Span<double> mortality = _mortality.AsSpan(_k * _nFocal * RunResults.MortalityCauses, _nFocal * RunResults.MortalityCauses);
        Span<double> yield = _yield.AsSpan(_k * _nFocal, _nFocal);
        biomass.Clear();
        abundance.Clear();
        mortality.Clear();
        yield.Clear();
        double[] cutoff = p.Output.CutoffAge;
        double ndt = p.NDtPerYear;
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

        _k++;
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
