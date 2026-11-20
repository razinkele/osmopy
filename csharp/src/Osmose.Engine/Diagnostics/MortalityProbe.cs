using System.Runtime.InteropServices;
using Osmose.Engine.Mortality;
using Osmose.Engine.Rng;
using Osmose.Engine.Schools;
using Osmose.Engine.Step;

namespace Osmose.Engine.Diagnostics;

/// <summary>
/// Test harness for plan task 1.5, callable through pythonnet: one whole mortality step on a
/// captured state with recorded draws. Schools are appended in Python's order, so each school's
/// <see cref="SchoolLife.Id"/> is its Python index and the results are written back in that
/// order whatever the regroup did.
/// </summary>
public static class MortalityProbe
{
    public static void Run(string modelPath, string statePath, string drawsPath, int step, string outPath)
    {
        using SimulationConfig model = SimulationConfig.Load(modelPath);
        using SimulationConfig state = SimulationConfig.Load(statePath);
        using SimulationConfig drawsBundle = SimulationConfig.Load(drawsPath);
        EngineParameters p = EngineParameters.From(model, checkFeatures: false);
        KernelTables tables = KernelTables.From(p);

        var r = new BundleReader(state, 1);
        long n = r.Array("state.species_id", DType.I4, BundleReader.Any)!.Length;
        int[] species = r.Array("state.species_id", DType.I4, n)!.ToArray<int>();
        int[] age = r.Array("state.age_dt", DType.I4, n)!.ToArray<int>();
        double[] length = r.Array("state.length", DType.F8, n)!.ToArray<double>();
        double[] weight = r.Array("state.weight", DType.F8, n)!.ToArray<double>();
        double[] abundance = r.Array("state.abundance", DType.F8, n)!.ToArray<double>();
        double[] tl = r.Array("state.trophic_level", DType.F8, n)!.ToArray<double>();
        int[] cx = r.Array("state.cell_x", DType.I4, n)!.ToArray<int>();
        int[] cy = r.Array("state.cell_y", DType.I4, n)!.ToArray<int>();
        byte[] bkg = r.Array("state.is_background", DType.U1, n)!.ToArray<byte>();
        byte[] egg = r.Array("state.is_egg", DType.U1, n)!.ToArray<byte>();
        byte[] isOut = r.Array("state.is_out", DType.U1, n)!.ToArray<byte>();
        int[] ffa = r.Array("state.first_feeding_age_dt", DType.I4, n)!.ToArray<int>();
        double[] starv = r.Array("state.starvation_rate", DType.F8, n)!.ToArray<double>();
        double[] psr = r.Array("state.pred_success_rate", DType.F8, n)!.ToArray<double>();
        double[] preyed = r.Array("state.preyed_biomass", DType.F8, n)!.ToArray<double>();
        r.ThrowIfProblems();

        int nCells = p.Ny * p.Nx;
        var store = new SchoolStore(nCells, initialCapacity: Math.Max(16, (int)n));
        for (int i = 0; i < n; i++)
        {
            if (abundance[i] == 0.0)
            {
                throw new InvalidOperationException($"school {i} has zero abundance: the regroup would drop it and the draws would no longer line up");
            }

            int at = store.Append(
                new SchoolCore
                {
                    SpeciesId = species[i],
                    AgeDt = age[i],
                    Length = length[i],
                    Weight = weight[i],
                    Abundance = abundance[i],
                    TrophicLevel = tl[i],
                    FirstFeedingAgeDt = checked((short)ffa[i]),
                    Flags = (bkg[i] != 0 ? SchoolFlags.IsBackground : 0) | (egg[i] != 0 ? SchoolFlags.IsEgg : 0)
                        | (isOut[i] != 0 ? SchoolFlags.IsOut : 0),
                },
                new SchoolLife { StarvationRate = starv[i], CellId = cx[i] >= 0 && cy[i] >= 0 ? (cy[i] * p.Nx) + cx[i] : -1 });
            store.Step[at].PredSuccess = psr[i];
            store.Step[at].PreyedBiomass = preyed[i];
        }

        // Step data is per-step scratch and does not survive the regroup; carry the two
        // accumulators across it by id.
        var keepPsr = new double[n];
        var keepPreyed = new double[n];
        for (int i = 0; i < n; i++)
        {
            keepPsr[i] = store.Step[i].PredSuccess;
            keepPreyed[i] = store.Step[i].PreyedBiomass;
        }

        store.Regroup();
        for (int i = 0; i < store.Count; i++)
        {
            ulong id = store.Life[i].Id;
            store.Step[i] = default;
            store.Step[i].PredSuccess = keepPsr[id];
            store.Step[i].PreyedBiomass = keepPreyed[id];
        }

        InjectedDrawSource[] perCell = InjectedDraws.FromBundle(drawsBundle, store.CellStart, nCells, p.NSubdt);
        var draws = new InjectedCellDraws(perCell);
        double[] rsc = new double[Math.Max(1, p.Resources.N * nCells)];
        double[] larva = new double[store.Count];
        MortalityStep.Run<InjectedCellDraws, InjectedDrawSource>(p, tables, store, step, rsc, new CellScratch(), larva, ref draws);

        double[] outAbundance = new double[n], outStarv = new double[n], outTl = new double[n];
        double[] outPsr = new double[n], outPreyed = new double[n], outEgg = new double[n];
        double[] outDead = new double[n * RunResults.MortalityCauses];
        for (int i = 0; i < store.Count; i++)
        {
            long id = (long)store.Life[i].Id;
            SchoolCore c = store.Core[i];
            SchoolLedger l = store.Ledger[i];
            outAbundance[id] = c.Abundance;
            outStarv[id] = store.Life[i].StarvationRate;
            outTl[id] = c.TrophicLevel;
            outPsr[id] = store.Step[i].PredSuccess;
            outPreyed[id] = store.Step[i].PreyedBiomass;
            outEgg[id] = c.EggRetained;
            double[] row = [c.DeadPredation, l.DeadStarvation, l.DeadAdditional, l.DeadFishing, l.DeadOut, l.DeadForaging, l.DeadDiscards, l.DeadAging];
            row.CopyTo(outDead, id * RunResults.MortalityCauses);
        }

        int nRsc = p.Resources.N;
        double[] rscOut = new double[Math.Max(1, nRsc * nCells)];
        for (int rr = 0; rr < nRsc; rr++)
        {
            for (int c = 0; c < nCells; c++)
            {
                rscOut[(rr * nCells) + c] = rsc[(c * nRsc) + rr]; // back to Python's (n_rsc, n_cells)
            }
        }

        var handles = new List<GCHandle>();
        try
        {
            long Pin(Array a)
            {
                GCHandle h = GCHandle.Alloc(a, GCHandleType.Pinned);
                handles.Add(h);
                return h.AddrOfPinnedObject();
            }

            var output = new SimulationConfig(1);
            output.SetArray("mort.abundance", DType.F8, Pin(outAbundance), n);
            output.SetArray("mort.n_dead", DType.F8, Pin(outDead), n, RunResults.MortalityCauses);
            output.SetArray("mort.starvation_rate", DType.F8, Pin(outStarv), n);
            output.SetArray("mort.trophic_level", DType.F8, Pin(outTl), n);
            output.SetArray("mort.pred_success_rate", DType.F8, Pin(outPsr), n);
            output.SetArray("mort.preyed_biomass", DType.F8, Pin(outPreyed), n);
            output.SetArray("mort.egg_retained", DType.F8, Pin(outEgg), n);
            output.SetArray("mort.resources", DType.F8, Pin(rscOut), Math.Max(1, nRsc), nCells);
            output.Save(outPath);
        }
        finally
        {
            foreach (GCHandle h in handles)
            {
                h.Free();
            }
        }
    }
}

/// <summary>Builds one <see cref="InjectedDrawSource"/> per cell from a draws bundle: per sub-step
/// permutations (<c>draws.perm_*</c>, <c>(n_subdt, n_located)</c>) and cause orders
/// (<c>draws.cause_orders</c>, <c>(n_subdt, n_located, 4)</c>), in cell order.</summary>
internal static class InjectedDraws
{
    public static InjectedDrawSource[] FromBundle(SimulationConfig b, ReadOnlySpan<int> cellStart, int nCells, int nSubdt)
    {
        var r = new BundleReader(b, 1);
        long nLocated = cellStart[nCells];
        ReadOnlySpan<int> pp = r.Array("draws.perm_predation", DType.I4, nSubdt, nLocated)!.AsSpan<int>();
        ReadOnlySpan<int> ps = r.Array("draws.perm_starvation", DType.I4, nSubdt, nLocated)!.AsSpan<int>();
        ReadOnlySpan<int> pf = r.Array("draws.perm_fishing", DType.I4, nSubdt, nLocated)!.AsSpan<int>();
        ReadOnlySpan<int> pa = r.Array("draws.perm_additional", DType.I4, nSubdt, nLocated)!.AsSpan<int>();
        ReadOnlySpan<byte> co = r.Array("draws.cause_orders", DType.U1, nSubdt, nLocated, 4)!.AsSpan<byte>();
        r.ThrowIfProblems();
        var result = new InjectedDrawSource[nCells];
        for (int c = 0; c < nCells; c++)
        {
            int s = cellStart[c], m = cellStart[c + 1] - s;
            var perms = new int[nSubdt][][];
            var orders = new byte[nSubdt][];
            for (int sub = 0; sub < nSubdt; sub++)
            {
                int o = (int)(sub * nLocated) + s;
                perms[sub] = [pp.Slice(o, m).ToArray(), ps.Slice(o, m).ToArray(), pf.Slice(o, m).ToArray(), pa.Slice(o, m).ToArray()];
                orders[sub] = co.Slice(o * 4, m * 4).ToArray();
            }

            result[c] = new InjectedDrawSource(perms, orders);
        }

        return result;
    }
}
