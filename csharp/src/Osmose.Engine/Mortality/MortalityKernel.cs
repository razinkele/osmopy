using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using Osmose.Engine.Rng;
using Osmose.Engine.Schools;

namespace Osmose.Engine.Mortality;

/// <summary>How a predator's candidate prey are found (spec D5). A struct type parameter whose
/// <see cref="Mode"/> is a JIT-time constant, so each mode compiles to its own kernel.</summary>
public interface IPreyFilter
{
    /// <summary>0: scan every school. 1: vector-built bitmask per predator. 2: #279's sorted
    /// candidate lists.</summary>
    static abstract int Mode { get; }
}

/// <summary>No prefilter: the exact tests see every school (the reference for check 4).</summary>
public struct FullScan : IPreyFilter
{
    public static int Mode => 0;
}

/// <summary>A bitmask per predator, built with Vector256 over the length lane (spec D5).</summary>
public struct MaskPrefilter : IPreyFilter
{
    public static int Mode => 1;
}

/// <summary>Per-predator candidate lists in cell order, as Python builds them (#279).</summary>
public struct CandidateLists : IPreyFilter
{
    public static int Mode => 2;
}

/// <summary>One worker's reusable buffers, grown to the largest cell; never shrunk.</summary>
public sealed class CellScratch
{
    public int[] SeqPredation = [];
    public int[] SeqStarvation = [];
    public int[] SeqFishing = [];
    public int[] SeqAdditional = [];
    public byte[] PreyType = [];
    public int[] PreyId = [];
    public double[] PreyEligible = [];
    public ulong[] Masks = [];
    public int[] CandPtr = [];
    public int[] CandPos = [];
    public int[] LenOrder = [];
    public double[] SortedLen = [];
    public double[] Gather = [];
    public int[] Bounds = [];

    /// <summary>The store's diet rows and where the current cell's begin; <see cref="DietCols"/>
    /// 0 when the diet is not tracked (the usual case, and every minimal run).</summary>
    public double[] Diet = [];
    public int DietStart;
    public int DietCols;

    /// <summary>Makes every buffer big enough for a cell of <paramref name="n"/> schools.</summary>
    public void Ensure(int n, int nResources)
    {
        if (SeqPredation.Length < n)
        {
            int cap = Math.Max(n, SeqPredation.Length * 2);
            SeqPredation = new int[cap];
            SeqStarvation = new int[cap];
            SeqFishing = new int[cap];
            SeqAdditional = new int[cap];
            LenOrder = new int[cap];
            SortedLen = new double[cap];
            Gather = new double[cap];
            CandPtr = new int[cap + 1];
            Bounds = new int[2 * cap];
        }

        if (PreyId.Length < n + nResources)
        {
            int cap = Math.Max(n + nResources, PreyId.Length * 2);
            PreyType = new byte[cap];
            PreyId = new int[cap];
            PreyEligible = new double[cap];
        }
    }

    internal Span<ulong> MaskRows(int n, int words)
    {
        long need = (long)n * words;
        if (Masks.Length < need)
        {
            Masks = new ulong[Math.Max(need, Masks.Length * 2)];
        }

        return Masks.AsSpan(0, (int)need);
    }

    internal Span<int> CandidateBuffer(int total)
    {
        if (CandPos.Length < total)
        {
            CandPos = new int[Math.Max(total, CandPos.Length * 2)];
        }

        return CandPos;
    }
}

/// <summary>
/// The interleaved mortality of ONE cell for ALL its sub-steps, bioenergetics off (spec 4.2,
/// 4.3, 6.4, 6.5; plan task 0.6). The arithmetic is <c>_apply_predation_numba</c> and
/// <c>_apply_single_cause</c> verbatim, including Python's <c>max</c>/<c>min</c> argument order
/// (<see cref="PyMax"/>), so given the same draws it is the Numba oracle to the last bit wherever
/// no <c>exp</c> is evaluated.
/// </summary>
public static class MortalityKernel
{
    /// <summary>Cause codes, as <c>MortalityCause</c> and the cause-order tables use them.</summary>
    public const byte Predation = 0, Starvation = 1, Additional = 2, Fishing = 3;

    private const double WindowSlack = 1e-9;

    public static void RunCell<TCell, TFilter, TDraws>(
        KernelTables t, ref TCell cell, Span<double> resourceBiomass, CellScratch s, ref TDraws draws)
        where TCell : struct, ICellSchools
        where TFilter : struct, IPreyFilter
        where TDraws : struct, IDrawSource
    {
        int n = cell.Count;
        if (n == 0)
        {
            return;
        }

        s.Ensure(n, t.NResources);
        int words = (n + 63) >> 6;
        if (TFilter.Mode == 1)
        {
            BuildMasks(t, ref cell, s.MaskRows(n, words), words);
        }
        else if (TFilter.Mode == 2)
        {
            BuildCandidateLists(t, ref cell, s);
        }

        Span<byte> order = stackalloc byte[4];
        Span<int> seqPred = s.SeqPredation.AsSpan(0, n);
        Span<int> seqStarv = s.SeqStarvation.AsSpan(0, n);
        Span<int> seqFish = s.SeqFishing.AsSpan(0, n);
        Span<int> seqAdd = s.SeqAdditional.AsSpan(0, n);
        for (int sub = 0; sub < t.NSubdt; sub++)
        {
            // Egg release, before the sub-step's draws, from the step-start abundance
            // (mortality(): release = abundance / n_subdt; retained = max(0, retained - release)).
            for (int p = 0; p < n; p++)
            {
                ref SchoolCore c = ref cell.Core(p);
                if ((c.Flags & SchoolFlags.IsEgg) != 0 && c.EggRetained > 0)
                {
                    c.EggRetained = Math.Max(0.0, c.EggRetained - cell.Step(p).EggRelease);
                }
            }

            draws.BeginSubstep(sub);
            draws.Permutation(DrawPurpose.PermPredation, seqPred);
            draws.Permutation(DrawPurpose.PermStarvation, seqStarv);
            draws.Permutation(DrawPurpose.PermFishing, seqFish);
            draws.Permutation(DrawPurpose.PermAdditional, seqAdd);
            for (int i = 0; i < n; i++)
            {
                draws.NextCauseOrder(order);
                for (int k = 0; k < 4; k++)
                {
                    switch (order[k])
                    {
                        case Predation:
                            Predate<TCell, TFilter>(t, ref cell, seqPred[i], resourceBiomass, s, words);
                            break;
                        case Starvation:
                            ApplyRate(ref cell, seqStarv[i], Starvation);
                            break;
                        case Additional:
                            ApplyRate(ref cell, seqAdd[i], Additional);
                            break;
                        case Fishing:
                            ApplyRate(ref cell, seqFish[i], Fishing);
                            break;
                        default:
                            throw new InvalidOperationException($"cause code {order[k]} outside the 4-cause set");
                    }
                }
            }
        }
    }

    /// <summary><c>_apply_single_cause</c> for starvation, additional and fishing, bioen off:
    /// <c>dead = abd * (1 - exp(-D))</c>, fishing split into landings and discards.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplyRate<TCell>(ref TCell cell, int p, byte cause)
        where TCell : struct, ICellSchools
    {
        ref SchoolStep st = ref cell.Step(p);
        double d = cause switch
        {
            Starvation => st.RateStarvation,
            Additional => st.RateAdditional,
            _ => st.RateFishing,
        };
        if (!(d > 0))
        {
            return;
        }

        ref SchoolCore c = ref cell.Core(p);
        double abd = c.Abundance;
        if (!(abd > 0))
        {
            return;
        }

        double dead = abd * (1.0 - Math.Exp(-d));
        ref SchoolLedger l = ref cell.Ledger(p);
        if (cause == Starvation)
        {
            l.DeadStarvation += dead;
        }
        else if (cause == Additional)
        {
            l.DeadAdditional += dead;
        }
        else if (st.DiscardFraction > 0)
        {
            l.DeadFishing += dead * (1.0 - st.DiscardFraction);
            l.DeadDiscards += dead * st.DiscardFraction;
        }
        else
        {
            l.DeadFishing += dead;
        }

        c.Abundance = abd - dead;
    }

    /// <summary><c>_apply_predation_numba</c> for one predator, bioen off.</summary>
    private static void Predate<TCell, TFilter>(KernelTables t, ref TCell cell, int p, Span<double> rsc, CellScratch s, int words)
        where TCell : struct, ICellSchools
        where TFilter : struct, IPreyFilter
    {
        ref SchoolCore pc = ref cell.Core(p);
        if (pc.AgeDt < pc.FirstFeedingAgeDt)
        {
            return;
        }

        double abdP = pc.Abundance;
        if (abdP <= 0)
        {
            return;
        }

        int spPred = pc.SpeciesId;
        double predLen = pc.Length;
        int ratioIdx = (spPred * t.NStage) + pc.FeedingStage;
        double rMin = t.SizeRatioMin[ratioIdx];
        double rMax = t.SizeRatioMax[ratioIdx];
        double biomassP = abdP * pc.Weight;
        double maxEatable = biomassP * t.IngestionRate[spPred] / (t.NDtPerYear * t.NSubdt);
        if (maxEatable <= 0)
        {
            return;
        }

        var scan = new Scan(t, spPred, pc.PredAccessCol, predLen, rMin, rMax, s);
        int n = cell.Count;
        if (TFilter.Mode == 0)
        {
            for (int q = 0; q < n; q++)
            {
                scan.Visit(ref cell, p, q);
            }
        }
        else if (TFilter.Mode == 1)
        {
            ReadOnlySpan<ulong> row = s.Masks.AsSpan(p * words, words);
            for (int w = 0; w < row.Length; w++)
            {
                ulong bits = row[w];
                while (bits != 0)
                {
                    scan.Visit(ref cell, p, (w << 6) + BitOperations.TrailingZeroCount(bits));
                    bits &= bits - 1;
                }
            }
        }
        else
        {
            for (int k = s.CandPtr[p]; k < s.CandPtr[p + 1]; k++)
            {
                scan.Visit(ref cell, p, s.CandPos[k]);
            }
        }

        // 1b: resource prey, in resource order.
        for (int r = 0; r < t.NResources; r++)
        {
            double rscBio = rsc[r];
            if (rscBio <= 0)
            {
                continue;
            }

            if (rMin <= 0 || rMax <= 0)
            {
                continue;
            }

            double preySizeMin = predLen / rMax;
            double preySizeMax = predLen / rMin;
            double overlapMin = PyMax(t.RscSizeMin[r], preySizeMin);
            double overlapMax = PyMin(t.RscSizeMax[r], preySizeMax);
            if (overlapMax <= overlapMin)
            {
                continue;
            }

            double rscRange = t.RscSizeMax[r] - t.RscSizeMin[r];
            if (rscRange <= 0)
            {
                continue;
            }

            double percent = (overlapMax - overlapMin) / rscRange;
            double access = 1.0;
            if (t.UseStageAccess)
            {
                int row = t.ResourceAccessRows[r];
                int col = pc.PredAccessCol;
                if (row >= 0 && col >= 0 && row < t.AccessPreyCount && col < t.AccessPredCount)
                {
                    access = t.AccessByPred[(col * t.AccessPreyCount) + row];
                    if (access <= 0)
                    {
                        continue;
                    }
                }
            }
            else if (t.HasAccess)
            {
                int col = t.NSpeciesFocal + r;
                if (spPred < t.AccessPredCount && col < t.AccessPreyCount)
                {
                    access = t.AccessByPred[(spPred * t.AccessPreyCount) + col];
                    if (access <= 0)
                    {
                        continue;
                    }
                }
            }

            scan.Add(1, r, rscBio * percent * access);
        }

        double total = scan.Total;
        if (total <= 0)
        {
            return;
        }

        // Phase 2: share the eaten biomass in scan order.
        double eatenTotal;
        int shape = t.FrShape[spPred];
        if (shape == 1)
        {
            eatenTotal = PyMin(total, maxEatable);
        }
        else
        {
            double ratio = total / maxEatable;
            double k = t.FrHalfSat[spPred];
            double gForm = shape == 2 ? ratio / (ratio + k) : ratio * ratio / ((ratio * ratio) + (k * k));
            double cap = PyMin(1.0, ratio);
            eatenTotal = maxEatable * PyMin(cap, gForm);
        }

        // The predator's diet row, when the diet is tracked: one column per focal species, then
        // one per resource. A school prey is booked under its SPECIES ID, as the reference does
        // (diet_matrix[p, species_id[q]]), so a background prey of id n_focal + k lands in the
        // column of resource k; ids beyond the columns are dropped.
        int dietCols = s.DietCols;
        Span<double> diet = dietCols > 0 ? s.Diet.AsSpan(s.DietStart + (p * dietCols), dietCols) : default;
        ref SchoolStep ps = ref cell.Step(p);
        for (int k = 0; k < scan.Count; k++)
        {
            double share = s.PreyEligible[k] / total;
            double eaten = eatenTotal * share;
            if (s.PreyType[k] == 0)
            {
                ref SchoolCore qc = ref cell.Core(s.PreyId[k]);
                if (qc.Weight > 0)
                {
                    double deadPrey = eaten / qc.Weight;
                    if (deadPrey > 0.0)
                    {
                        qc.DeadPredation += deadPrey;
                        qc.Abundance -= deadPrey;
                    }
                }

                double preyTl = qc.TrophicLevel;
                if (preyTl <= 0)
                {
                    preyTl = 1.0;
                }

                ps.TlWeightedSum += preyTl * eaten;
                if (qc.SpeciesId < dietCols)
                {
                    diet[qc.SpeciesId] += eaten;
                }
            }
            else
            {
                int r = s.PreyId[k];
                rsc[r] = PyMax(0.0, rsc[r] - eaten);
                double rTl = t.RscTrophicLevel[r];
                if (rTl <= 0)
                {
                    rTl = 1.0;
                }

                ps.TlWeightedSum += rTl * eaten;

                // The resources are the LAST columns, after every species: focal, background,
                // resources, as the reference lays the prey axis out. (With the diet off the
                // column count is 0 and the index is negative.)
                int rscCol = dietCols - t.NResources + r;
                if (rscCol >= 0 && rscCol < dietCols)
                {
                    diet[rscCol] += eaten;
                }
            }
        }

        // Phase 3.
        double success = PyMin(eatenTotal / maxEatable, 1.0);
        ps.PredSuccess += success / t.NSubdt;
        ps.PreyedBiomass += eatenTotal;
    }

    /// <summary>Phase 1's prey buffer and running sum for one predator visit.</summary>
    private ref struct Scan
    {
        private readonly KernelTables _t;
        private readonly int _spPred;
        private readonly int _predCol;
        private readonly double _predLen;
        private readonly double _rMin;
        private readonly double _rMax;
        private readonly Span<byte> _type;
        private readonly Span<int> _id;
        private readonly Span<double> _eligible;

        public Scan(KernelTables t, int spPred, int predCol, double predLen, double rMin, double rMax, CellScratch s)
        {
            _t = t;
            _spPred = spPred;
            _predCol = predCol;
            _predLen = predLen;
            _rMin = rMin;
            _rMax = rMax;
            _type = s.PreyType;
            _id = s.PreyId;
            _eligible = s.PreyEligible;
            Count = 0;
            Total = 0.0;
        }

        public int Count { get; private set; }

        public double Total { get; private set; }

        /// <summary>Phase 1a for one candidate school, in cell order.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Visit<TCell>(ref TCell cell, int p, int q)
            where TCell : struct, ICellSchools
        {
            if (q == p)
            {
                return;
            }

            ref SchoolCore qc = ref cell.Core(q);
            double abdQ = PyMax(qc.Abundance - qc.EggRetained, 0.0);
            if (abdQ <= 0)
            {
                return;
            }

            double preyLen = qc.Length;
            if (preyLen <= 0)
            {
                return;
            }

            double ratio = _predLen / preyLen;
            if (ratio < _rMin || ratio >= _rMax)
            {
                return;
            }

            double access = 1.0;
            if (_t.HasAccess)
            {
                if (_t.UseStageAccess)
                {
                    int col = _predCol;
                    int row = qc.PreyAccessRow;
                    if (col >= 0 && row >= 0)
                    {
                        if (row < _t.AccessPreyCount && col < _t.AccessPredCount)
                        {
                            access = _t.AccessByPred[(col * _t.AccessPreyCount) + row];
                        }

                        if (access <= 0)
                        {
                            return;
                        }
                    }
                }
                else
                {
                    int spPrey = qc.SpeciesId;
                    if (_spPred < _t.AccessPredCount && spPrey < _t.AccessPreyCount)
                    {
                        access = _t.AccessByPred[(_spPred * _t.AccessPreyCount) + spPrey];
                        if (access <= 0)
                        {
                            return;
                        }
                    }
                }
            }

            double preyBio = abdQ * qc.Weight;
            if (preyBio <= 0)
            {
                return;
            }

            Add(0, q, preyBio * access);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Add(byte type, int id, double eligible)
        {
            _type[Count] = type;
            _id[Count] = id;
            _eligible[Count] = eligible;
            Total += eligible;
            Count++;
        }
    }

    /// <summary>Python's <c>max(a, b)</c>: <c>a</c> unless <c>b &gt; a</c> (not IEEE maximum).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double PyMax(double a, double b) => b > a ? b : a;

    /// <summary>Python's <c>min(a, b)</c>: <c>a</c> unless <c>b &lt; a</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double PyMin(double a, double b) => b < a ? b : a;

    /// <summary>One bitmask row per predator over the cell's length lane, the window widened by
    /// 1e-9 relative so the mask is a SUPERSET of what the exact test accepts (spec D5; the
    /// edge cases follow <c>_cell_prey_candidates</c>).</summary>
    internal static void BuildMasks<TCell>(KernelTables t, ref TCell cell, Span<ulong> masks, int words)
        where TCell : struct, ICellSchools
    {
        ReadOnlySpan<double> len = cell.Lengths;
        int n = len.Length;
        bool cellHasNaN = false;
        for (int i = 0; i < n; i++)
        {
            if (double.IsNaN(len[i]))
            {
                cellHasNaN = true;
                break;
            }
        }

        masks.Clear();
        for (int p = 0; p < n; p++)
        {
            Span<ulong> row = masks.Slice(p * words, words);
            ref SchoolCore pc = ref cell.Core(p);
            if (pc.AgeDt < pc.FirstFeedingAgeDt)
            {
                continue;
            }

            int idx = (pc.SpeciesId * t.NStage) + pc.FeedingStage;
            double rMin = t.SizeRatioMin[idx];
            double rMax = t.SizeRatioMax[idx];
            if (cellHasNaN || double.IsNaN(rMin) || double.IsNaN(rMax))
            {
                SetFirst(row, n);
                continue;
            }

            if (rMax <= 0)
            {
                continue;
            }

            double lo = pc.Length / rMax * (1.0 - WindowSlack);
            double hi = rMin > 0 ? pc.Length / rMin * (1.0 + WindowSlack) : double.PositiveInfinity;
            FillRow(len, lo, hi, row);
        }
    }

    private static void FillRow(ReadOnlySpan<double> len, double lo, double hi, Span<ulong> row)
    {
        int n = len.Length;
        int q = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            Vector256<double> vlo = Vector256.Create(lo);
            Vector256<double> vhi = Vector256.Create(hi);
            ref double first = ref MemoryMarshal.GetReference(len);
            for (; q + 4 <= n; q += 4)
            {
                Vector256<double> v = Vector256.LoadUnsafe(ref first, (nuint)q);
                uint bits = (Vector256.GreaterThanOrEqual(v, vlo) & Vector256.LessThanOrEqual(v, vhi)).ExtractMostSignificantBits();
                if (bits != 0)
                {
                    row[q >> 6] |= (ulong)bits << (q & 63);
                }
            }
        }

        for (; q < n; q++)
        {
            if (len[q] >= lo && len[q] <= hi)
            {
                row[q >> 6] |= 1UL << (q & 63);
            }
        }
    }

    private static void SetFirst(Span<ulong> row, int n)
    {
        for (int q = 0; q < n; q++)
        {
            row[q >> 6] |= 1UL << (q & 63);
        }
    }

    /// <summary>#279's candidate lists for one cell: sort lengths, binary-search each predator's
    /// widened window, then sort the slice back into cell order.</summary>
    internal static void BuildCandidateLists<TCell>(KernelTables t, ref TCell cell, CellScratch s)
        where TCell : struct, ICellSchools
    {
        ReadOnlySpan<double> len = cell.Lengths;
        int n = len.Length;
        Span<int> order = s.LenOrder.AsSpan(0, n);
        Span<double> sorted = s.SortedLen.AsSpan(0, n);
        bool cellHasNaN = false;
        for (int i = 0; i < n; i++)
        {
            order[i] = i;
            sorted[i] = len[i];
            cellHasNaN |= double.IsNaN(len[i]);
        }

        sorted.Sort(order);
        Span<int> bounds = s.Bounds.AsSpan(0, 2 * n);
        int total = 0;
        for (int p = 0; p < n; p++)
        {
            int i0 = 0, i1 = 0;
            ref SchoolCore pc = ref cell.Core(p);
            if (pc.AgeDt >= pc.FirstFeedingAgeDt)
            {
                int idx = (pc.SpeciesId * t.NStage) + pc.FeedingStage;
                double rMin = t.SizeRatioMin[idx];
                double rMax = t.SizeRatioMax[idx];
                if (cellHasNaN || double.IsNaN(rMin) || double.IsNaN(rMax))
                {
                    i1 = n;
                }
                else if (rMax > 0)
                {
                    i0 = LowerBound(sorted, pc.Length / rMax * (1.0 - WindowSlack));
                    i1 = rMin > 0 ? UpperBound(sorted, pc.Length / rMin * (1.0 + WindowSlack)) : n;
                    i1 = Math.Max(i1, i0);
                }
            }

            bounds[2 * p] = i0;
            bounds[(2 * p) + 1] = i1;
            total += i1 - i0;
        }

        Span<int> cand = s.CandidateBuffer(total);
        s.CandPtr[0] = 0;
        for (int p = 0; p < n; p++)
        {
            int i0 = bounds[2 * p], m = bounds[(2 * p) + 1] - i0, at = s.CandPtr[p];
            order.Slice(i0, m).CopyTo(cand.Slice(at, m));
            cand.Slice(at, m).Sort();
            s.CandPtr[p + 1] = at + m;
        }
    }

    private static int LowerBound(ReadOnlySpan<double> a, double v)
    {
        int lo = 0, hi = a.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >>> 1;
            if (a[mid] < v)
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

    private static int UpperBound(ReadOnlySpan<double> a, double v)
    {
        int lo = 0, hi = a.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >>> 1;
            if (a[mid] <= v)
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
}
