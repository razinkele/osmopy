using System.Runtime.InteropServices;
using Osmose.Engine.Schools;
using Osmose.Engine.Step;

namespace Osmose.Engine.Diagnostics;

/// <summary>
/// Test harness for plan task 1.4, callable through pythonnet: runs the per-step preparation on a
/// school state captured from the Python engine and writes the results as a bundle file, so a
/// Python test can compare them with the engine's own functions. Not part of a simulation.
/// </summary>
public static class PrepProbe
{
    /// <summary>
    /// <paramref name="modelPath"/>: a bundle from <c>build_bundle</c>. <paramref name="statePath"/>:
    /// a bundle of <c>state.*</c> arrays (one entry per school, Python's order). Writes
    /// <c>prep.*</c> arrays to <paramref name="outPath"/>.
    /// </summary>
    public static void Run(string modelPath, string statePath, int step, string outPath)
    {
        using SimulationConfig model = SimulationConfig.Load(modelPath);
        using SimulationConfig state = SimulationConfig.Load(statePath);
        EngineParameters p = EngineParameters.From(model, checkFeatures: false);
        var r = new BundleReader(state, 1);
        BundleEntry? sp = r.Array("state.species_id", DType.I4, BundleReader.Any);
        r.ThrowIfProblems();
        long n = sp!.Length;
        int[] species = sp.ToArray<int>();
        int[] age = r.Array("state.age_dt", DType.I4, n)!.ToArray<int>();
        double[] length = r.Array("state.length", DType.F8, n)!.ToArray<double>();
        double[] weight = r.Array("state.weight", DType.F8, n)!.ToArray<double>();
        double[] abundance = r.Array("state.abundance", DType.F8, n)!.ToArray<double>();
        double[] tl = r.Array("state.trophic_level", DType.F8, n)!.ToArray<double>();
        int[] cx = r.Array("state.cell_x", DType.I4, n)!.ToArray<int>();
        int[] cy = r.Array("state.cell_y", DType.I4, n)!.ToArray<int>();
        byte[] bkg = r.Array("state.is_background", DType.U1, n)!.ToArray<byte>();
        byte[] egg = r.Array("state.is_egg", DType.U1, n)!.ToArray<byte>();
        int[] ffa = r.Array("state.first_feeding_age_dt", DType.I4, n)!.ToArray<int>();
        double[] starv = r.Array("state.starvation_rate", DType.F8, n)!.ToArray<double>();
        r.ThrowIfProblems();

        var store = new SchoolStore(p.Ny * p.Nx, initialCapacity: Math.Max(16, (int)n));
        for (int i = 0; i < n; i++)
        {
            store.Append(
                new SchoolCore
                {
                    SpeciesId = species[i],
                    AgeDt = age[i],
                    Length = length[i],
                    Weight = weight[i],
                    Abundance = abundance[i],
                    TrophicLevel = tl[i],
                    Flags = (bkg[i] != 0 ? SchoolFlags.IsBackground : 0) | (egg[i] != 0 ? SchoolFlags.IsEgg : 0),
                    FirstFeedingAgeDt = checked((short)ffa[i]),
                },
                new SchoolLife
                {
                    StarvationRate = starv[i],
                    CellId = cx[i] >= 0 && cy[i] >= 0 ? (cy[i] * p.Nx) + cx[i] : -1,
                });
        }

        StepPreparation.StagesAndAccess(p, store.Core);
        StepPreparation.Rates(p, store.Core, store.Life, store.Step, step);
        int nRsc = p.Resources.N, nCells = p.Ny * p.Nx;
        double[] cellMajor = new double[Math.Max(1, nRsc * nCells)];

        // Optional: the biomass carried into the step, Python's (n_rsc, n_cells). Only depletable
        // resources read it; without the entry the buffer is empty, as at the start of a run.
        if (state.TryGet("state.resources", out _))
        {
            double[] carried = r.Array("state.resources", DType.F8, nRsc, nCells)!.ToArray<double>();
            r.ThrowIfProblems();
            for (int rr = 0; rr < nRsc; rr++)
            {
                for (int c = 0; c < nCells; c++)
                {
                    cellMajor[(c * nRsc) + rr] = carried[(rr * nCells) + c];
                }
            }
        }

        StepPreparation.UpdateResources(p, step, cellMajor);

        double[] es = new double[n], ea = new double[n], ef = new double[n], ed = new double[n];
        int[] stage = new int[n], prey = new int[n], pred = new int[n];
        for (int i = 0; i < n; i++)
        {
            es[i] = store.Step[i].RateStarvation;
            ea[i] = store.Step[i].RateAdditional;
            ef[i] = store.Step[i].RateFishing;
            ed[i] = store.Step[i].DiscardFraction;
            stage[i] = store.Core[i].FeedingStage;
            prey[i] = store.Core[i].PreyAccessRow;
            pred[i] = store.Core[i].PredAccessCol;
        }

        double[] rsc = new double[Math.Max(1, nRsc * nCells)];
        for (int rr = 0; rr < nRsc; rr++)
        {
            for (int c = 0; c < nCells; c++)
            {
                rsc[(rr * nCells) + c] = cellMajor[(c * nRsc) + rr]; // back to Python's (n_rsc, n_cells)
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
            output.SetArray("prep.eff_starvation", DType.F8, Pin(es), n);
            output.SetArray("prep.eff_additional", DType.F8, Pin(ea), n);
            output.SetArray("prep.eff_fishing", DType.F8, Pin(ef), n);
            output.SetArray("prep.discard", DType.F8, Pin(ed), n);
            output.SetArray("prep.feeding_stage", DType.I4, Pin(stage), n);
            output.SetArray("prep.prey_row", DType.I4, Pin(prey), n);
            output.SetArray("prep.pred_col", DType.I4, Pin(pred), n);
            output.SetArray("prep.resources", DType.F8, Pin(rsc), Math.Max(1, nRsc), nCells);
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
