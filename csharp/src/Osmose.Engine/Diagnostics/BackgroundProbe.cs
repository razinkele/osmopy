using System.Runtime.InteropServices;
using Osmose.Engine.Schools;
using Osmose.Engine.Step;

namespace Osmose.Engine.Diagnostics;

/// <summary>
/// Test harness for plan task 2.1 (background species), callable through pythonnet.
/// </summary>
public static class BackgroundProbe
{
    /// <summary>The background schools <see cref="Background.Inject"/> appends at
    /// <paramref name="step"/>, in injection order, for comparison with
    /// <c>BackgroundState.get_schools</c>.</summary>
    public static void Schools(string modelPath, int step, string outPath)
    {
        using SimulationConfig model = SimulationConfig.Load(modelPath);
        EngineParameters p = EngineParameters.From(model, checkFeatures: false);
        var store = new SchoolStore(p.Ny * p.Nx);
        ulong first = Background.Inject(p, store, step);
        int n = store.Count;
        int[] species = new int[n], age = new int[n], feeding = new int[n], cx = new int[n], cy = new int[n];
        double[] length = new double[n], weight = new double[n], abundance = new double[n], level = new double[n];
        byte[] flags = new byte[n];
        long[] ordinal = new long[n];
        for (int i = 0; i < n; i++)
        {
            SchoolCore c = store.Core[i];
            int cell = store.Life[i].CellId;
            species[i] = c.SpeciesId;
            age[i] = c.AgeDt;
            feeding[i] = c.FirstFeedingAgeDt;
            length[i] = c.Length;
            weight[i] = c.Weight;
            abundance[i] = c.Abundance;
            level[i] = c.TrophicLevel;
            flags[i] = (byte)c.Flags;
            cx[i] = cell % p.Nx;
            cy[i] = cell / p.Nx;
            ordinal[i] = (long)(store.Life[i].Id - first);
        }

        Write(outPath, o =>
        {
            o.Int("bkg.per_step", Background.SchoolsPerStep(p.Background));
            o.Array("bkg.species_id", DType.I4, species);
            o.Array("bkg.age_dt", DType.I4, age);
            o.Array("bkg.first_feeding_age_dt", DType.I4, feeding);
            o.Array("bkg.cell_x", DType.I4, cx);
            o.Array("bkg.cell_y", DType.I4, cy);
            o.Array("bkg.length", DType.F8, length);
            o.Array("bkg.weight", DType.F8, weight);
            o.Array("bkg.abundance", DType.F8, abundance);
            o.Array("bkg.trophic_level", DType.F8, level);
            o.Array("bkg.flags", DType.U1, flags);
            o.Array("bkg.ordinal", DType.I8, ordinal);
        });
    }

    /// <summary>
    /// The background totals and the removal, on a state as it stands after mortality:
    /// <paramref name="statePath"/> holds <c>state.species_id</c>, <c>abundance</c>, <c>weight</c>,
    /// <c>cell_x</c>, <c>cell_y</c> and <c>is_background</c>, focal schools first and the step's
    /// background schools after them in injection order (as the Python state holds them). The
    /// store is regrouped by cell before the totals are taken, as in a real step. Writes the
    /// totals per species and the ids (input positions) of the schools left, in store order.
    /// </summary>
    public static void Totals(string modelPath, string statePath, string outPath)
    {
        using SimulationConfig model = SimulationConfig.Load(modelPath);
        using SimulationConfig state = SimulationConfig.Load(statePath);
        EngineParameters p = EngineParameters.From(model, checkFeatures: false);
        var r = new BundleReader(state, 1);
        long n = r.Array("state.species_id", DType.I4, BundleReader.Any)!.Length;
        int[] species = r.Array("state.species_id", DType.I4, n)!.ToArray<int>();
        double[] abundance = r.Array("state.abundance", DType.F8, n)!.ToArray<double>();
        double[] weight = r.Array("state.weight", DType.F8, n)!.ToArray<double>();
        int[] cx = r.Array("state.cell_x", DType.I4, n)!.ToArray<int>();
        int[] cy = r.Array("state.cell_y", DType.I4, n)!.ToArray<int>();
        byte[] bkg = r.Array("state.is_background", DType.U1, n)!.ToArray<byte>();
        r.ThrowIfProblems();

        var store = new SchoolStore(p.Ny * p.Nx, initialCapacity: Math.Max(16, (int)n));
        ulong first = 0;
        bool seen = false;
        for (int i = 0; i < n; i++)
        {
            if (bkg[i] != 0 && !seen)
            {
                first = store.NextId;
                seen = true;
            }

            store.Append(
                new SchoolCore { SpeciesId = species[i], Abundance = abundance[i], Weight = weight[i], Flags = bkg[i] != 0 ? SchoolFlags.IsBackground : 0 },
                new SchoolLife { CellId = cx[i] >= 0 && cy[i] >= 0 ? (cy[i] * p.Nx) + cx[i] : -1 });
        }

        store.Regroup();
        int nTotal = p.Species.NTotal;
        double[] biomass = new double[nTotal], total = new double[nTotal];
        double[] scratch = new double[2 * Background.SchoolsPerStep(p.Background)];
        int before = store.Count;
        Background.CollectAndRemove(p, store, first, scratch, biomass, total);
        long[] left = new long[Math.Max(1, store.Count)];
        for (int i = 0; i < store.Count; i++)
        {
            left[i] = (long)store.Life[i].Id;
        }

        int kept = store.Count;
        Write(outPath, o =>
        {
            o.Int("bkg.removed", before - kept);
            o.Int("bkg.left_count", kept);
            o.Array("bkg.biomass", DType.F8, biomass);
            o.Array("bkg.abundance", DType.F8, total);
            o.Array("bkg.left_ids", DType.I8, left);
        });
    }

    private static void Write(string outPath, Action<Writer> fill)
    {
        var handles = new List<GCHandle>();
        try
        {
            var output = new SimulationConfig(1);
            fill(new Writer(output, handles));
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

    private sealed class Writer(SimulationConfig output, List<GCHandle> handles)
    {
        public void Int(string name, long value) => output.SetInt(name, value);

        public void Array(string name, DType dtype, System.Array values)
        {
            System.Array pinned = values.Length > 0 ? values : System.Array.CreateInstance(values.GetType().GetElementType()!, 1);
            GCHandle h = GCHandle.Alloc(pinned, GCHandleType.Pinned);
            handles.Add(h);
            output.SetArray(name, dtype, h.AddrOfPinnedObject(), values.Length);
        }
    }
}
