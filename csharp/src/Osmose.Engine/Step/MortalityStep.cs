using System.Runtime.CompilerServices;
using Osmose.Engine.Mortality;
using Osmose.Engine.Rng;
using Osmose.Engine.Schools;

namespace Osmose.Engine.Step;

/// <summary>Supplies the draw source for each cell of a step (a struct type parameter, so calls are
/// devirtualised): Philox in production, injected draws in tests.</summary>
public interface ICellDraws<TDraws>
    where TDraws : struct, IDrawSource
{
    TDraws ForCell(int cell);
}

/// <summary>Production draws: every value a pure function of the run key, the step and the cell.</summary>
public readonly struct PhiloxCellDraws(PhiloxKey key, int step) : ICellDraws<PhiloxDrawSource>
{
    public PhiloxDrawSource ForCell(int cell) => new(key, step, cell);
}

/// <summary>Test draws: one recorded source per cell.</summary>
public readonly struct InjectedCellDraws(InjectedDrawSource[] perCell) : ICellDraws<InjectedDrawSource>
{
    public InjectedDrawSource ForCell(int cell) => perCell[cell];
}

/// <summary>
/// One whole mortality step (plan task 1.5, spec 6.6–6.7), in <c>mortality()</c>'s order:
/// larva pre-pass, egg retention, the per-step preparation, the interleaved loop cell by cell
/// (all sub-steps per cell, spec D3), the commit, out-of-domain mortality, the lagged starvation
/// rate and the trophic level. The store must be regrouped (cells contiguous) beforehand.
/// </summary>
public static class MortalityStep
{
    /// <summary>Runs the step serially: <see cref="Prepare"/>, every cell, <see cref="Commit"/>.
    /// <paramref name="larva"/> is per-school scratch of at least <c>store.Count</c> values.</summary>
    public static void Run<TFactory, TDraws>(
        EngineParameters p, KernelTables t, SchoolStore store, int step, Span<double> rscCellMajor,
        CellScratch scratch, Span<double> larva, ref TFactory draws)
        where TFactory : struct, ICellDraws<TDraws>
        where TDraws : struct, IDrawSource
    {
        Prepare(p, store, step, rscCellMajor, larva);
        for (int cell = 0; cell < store.CellCount; cell++)
        {
            RunCell<TFactory, TDraws>(t, store, cell, rscCellMajor, scratch, ref draws);
        }

        Commit(p, store, step, larva);
    }

    /// <summary>Before the cells: larva pre-pass, egg retention and release, the per-step
    /// preparation (stages, access indices, rates, resource capacity).</summary>
    public static void Prepare(EngineParameters p, SchoolStore store, int step, Span<double> rscCellMajor, Span<double> larva)
    {
        int n = store.Count, nFocal = p.Species.NFocal, nsub = p.NSubdt;
        store.ClearDiet();
        Span<SchoolCore> core = store.Core;
        Span<SchoolStep> st = store.Step;
        Span<SchoolLedger> led = store.Ledger;
        Span<SchoolLife> life = store.Life;

        // Larva pre-pass: the full per-cohort rate, once; deaths kept apart and added to the
        // ADDITIONAL counter after the loop, as mortality() merges larva_deaths (loop + larva).
        for (int i = 0; i < n; i++)
        {
            ref SchoolCore c = ref core[i];
            larva[i] = 0.0;
            if ((c.Flags & SchoolFlags.IsEgg) != 0)
            {
                double fraction = 1 - Math.Exp(-p.Mortality.LarvaRateByStep[(step * nFocal) + c.SpeciesId]);
                double dead = c.Abundance * fraction;
                larva[i] = dead;
                c.Abundance -= dead;
            }

            led[i] = default;
            led[i].AbundanceAtMortalityStart = c.Abundance;
            c.DeadPredation = 0.0;
            bool egg = (c.Flags & SchoolFlags.IsEgg) != 0;
            c.EggRetained = egg ? c.Abundance : 0.0;
            st[i].EggRelease = egg ? c.Abundance / nsub : 0.0;
            st[i].TlWeightedSum = 0.0;
        }

        StepPreparation.StagesAndAccess(p, core);
        StepPreparation.Rates(p, core, life, st, step);
        StepPreparation.UpdateResources(p, step, rscCellMajor);
    }

    /// <summary>One cell, every sub-step (spec D3). Cells are disjoint in schools and resources
    /// and their draws are counter-based, so cells may run on any thread in any order.</summary>
    public static void RunCell<TFactory, TDraws>(KernelTables t, SchoolStore store, int cell, Span<double> rscCellMajor, CellScratch scratch, ref TFactory draws)
        where TFactory : struct, ICellDraws<TDraws>
        where TDraws : struct, IDrawSource
    {
        ReadOnlySpan<int> cellStart = store.CellStart;
        int start = cellStart[cell], count = cellStart[cell + 1] - start;
        if (count == 0)
        {
            return;
        }

        int nRsc = t.NResources;
        var schools = new ContiguousCell(store.CoreArray, store.StepArray, store.LedgerArray, store.LaneArray, start, count);
        TDraws d = draws.ForCell(cell);
        Span<double> rsc = nRsc > 0 ? rscCellMajor.Slice(cell * nRsc, nRsc) : [];
        scratch.Diet = store.DietArray; // the cell's schools are contiguous: so are their rows
        scratch.DietStart = start * store.DietCols;
        scratch.DietCols = store.DietCols;
        MortalityKernel.RunCell<ContiguousCell, MaskPrefilter, TDraws>(t, ref schools, rsc, scratch, ref d);
    }

    /// <summary>After the cells: the commit, out-of-domain mortality, the lagged starvation rate,
    /// the trophic level, and the death-accounting check under <c>run.checks</c>.</summary>
    public static void Commit(EngineParameters p, SchoolStore store, int step, Span<double> larva)
    {
        int n = store.Count, ndt = p.NDtPerYear;
        Span<SchoolCore> core = store.Core;
        Span<SchoolStep> st = store.Step;
        Span<SchoolLedger> led = store.Ledger;
        Span<SchoolLife> life = store.Life;
        for (int i = 0; i < n; i++)
        {
            ref SchoolCore c = ref core[i];
            ref SchoolLedger l = ref led[i];
            double total = Pairwise8(
                c.DeadPredation, l.DeadStarvation, l.DeadAdditional, l.DeadFishing,
                l.DeadOut, l.DeadForaging, l.DeadDiscards, l.DeadAging);
            c.Abundance = Math.Max(0.0, l.AbundanceAtMortalityStart - total); // np.maximum(0.0, ...)
            l.DeadAdditional += larva[i];

            if ((c.Flags & SchoolFlags.IsOut) != 0)
            {
                double fraction = 1 - Math.Exp(-(p.Mortality.OutRate[c.SpeciesId] / ndt));
                double dead = c.Abundance * fraction;
                c.Abundance -= dead;
                l.DeadOut += dead;
            }

            // Lagged starvation rate for the next step (update_starvation_rate).
            double sr = st[i].PredSuccess;
            double csr = p.Growth.CriticalSuccessRate[c.SpeciesId];
            double rate = sr <= csr ? p.Mortality.StarvationRateMax[c.SpeciesId] * (1 - (sr / (csr > 0 ? csr : 1.0))) : 0.0;
            rate = Math.Max(0.0, rate);
            life[i].StarvationRate = (c.Flags & SchoolFlags.IsBackground) != 0 ? 0.0 : rate;

            if (st[i].PreyedBiomass > 0 && st[i].TlWeightedSum > 0)
            {
                c.TrophicLevel = 1.0 + (st[i].TlWeightedSum / st[i].PreyedBiomass);
            }
        }

        if (p.Checks)
        {
            CheckDeathAccounting(core, led, life, larva[..n], step);
        }
    }

    /// <summary>Relative tolerance of <see cref="CheckDeathAccounting"/> (spec, Phase 1 acceptance 3).</summary>
    public const double DeathAccountingTolerance = 1e-9;

    /// <summary>
    /// The death-accounting invariant (spec 10.4): for every school, the abundance removed this step
    /// (pre-larva abundance minus the committed abundance) equals the sum of its eight death counters,
    /// to <see cref="DeathAccountingTolerance"/> relative to the pre-larva abundance. NaN anywhere is a
    /// violation. Catches overkill, which the commit's <c>max(0, ...)</c> would otherwise hide.
    /// </summary>
    public static void CheckDeathAccounting(
        ReadOnlySpan<SchoolCore> core, ReadOnlySpan<SchoolLedger> led, ReadOnlySpan<SchoolLife> life,
        ReadOnlySpan<double> larva, int step)
    {
        for (int i = 0; i < core.Length; i++)
        {
            SchoolLedger l = led[i];
            double start = l.AbundanceAtMortalityStart + larva[i];
            double removed = start - core[i].Abundance;
            double counted = Pairwise8(
                core[i].DeadPredation, l.DeadStarvation, l.DeadAdditional, l.DeadFishing,
                l.DeadOut, l.DeadForaging, l.DeadDiscards, l.DeadAging);
            if (!(Math.Abs(removed - counted) <= DeathAccountingTolerance * start))
            {
                throw new InvariantViolationException(
                    life[i].Id, step, $"death accounting: removed {removed:R} of {start:R}, counters sum to {counted:R}");
            }
        }
    }

    /// <summary>NumPy's pairwise sum of 8 contiguous values, which is how
    /// <c>n_dead.sum(axis=1)</c> adds a school's 8 cause counters. Left to right would round
    /// differently.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Pairwise8(double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7) =>
        ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
}
