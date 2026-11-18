using System.Runtime.InteropServices;
using Osmose.Engine.Schools;
using Osmose.Engine.Step;

namespace Osmose.Engine.Diagnostics;

/// <summary>
/// Test harness for plan task 1.7, callable through pythonnet: runs ONE life-cycle stage
/// (<c>floor</c>, <c>growth</c>, <c>aging</c> or <c>reproduction</c>) on a school state captured
/// at that stage's entry in a Python run, and writes every school's fields in store order (the
/// input's order, then any new egg schools), so each stage is compared on Python's own input.
/// </summary>
public static class LifeCycleProbe
{
    public static void Run(string modelPath, string statePath, string stage, int step, string outPath)
    {
        using SimulationConfig model = SimulationConfig.Load(modelPath);
        using SimulationConfig state = SimulationConfig.Load(statePath);
        EngineParameters p = EngineParameters.From(model, checkFeatures: false);
        var r = new BundleReader(state, 1);
        long n = r.Array("state.species_id", DType.I4, BundleReader.Any)!.Length;
        int[] species = r.Array("state.species_id", DType.I4, n)!.ToArray<int>();
        int[] age = r.Array("state.age_dt", DType.I4, n)!.ToArray<int>();
        double[] length = r.Array("state.length", DType.F8, n)!.ToArray<double>();
        double[] weight = r.Array("state.weight", DType.F8, n)!.ToArray<double>();
        double[] abundance = r.Array("state.abundance", DType.F8, n)!.ToArray<double>();
        double[] tl = r.Array("state.trophic_level", DType.F8, n)!.ToArray<double>();
        byte[] isOut = r.Array("state.is_out", DType.U1, n)!.ToArray<byte>();
        byte[] isEgg = r.Array("state.is_egg", DType.U1, n)!.ToArray<byte>();
        byte[] fromSeeding = r.Array("state.from_seeding", DType.U1, n)!.ToArray<byte>();
        int[] ffa = r.Array("state.first_feeding_age_dt", DType.I4, n)!.ToArray<int>();
        double[] psr = r.Array("state.pred_success_rate", DType.F8, n)!.ToArray<double>();
        int[] cx = r.Array("state.cell_x", DType.I4, n)!.ToArray<int>();
        int[] cy = r.Array("state.cell_y", DType.I4, n)!.ToArray<int>();
        double[] starv = r.Array("state.starvation_rate", DType.F8, n)!.ToArray<double>();
        double[] aging = r.Array("state.dead_aging", DType.F8, n)!.ToArray<double>();
        r.ThrowIfProblems();

        int nx = p.Nx;
        var store = new SchoolStore(p.Ny * nx, initialCapacity: Math.Max(16, (int)n));
        for (int i = 0; i < n; i++)
        {
            SchoolFlags flags = (isOut[i] != 0 ? SchoolFlags.IsOut : 0) | (isEgg[i] != 0 ? SchoolFlags.IsEgg : 0)
                | (fromSeeding[i] != 0 ? SchoolFlags.FromSeeding : 0);
            int at = store.Append(
                new SchoolCore
                {
                    SpeciesId = species[i],
                    AgeDt = age[i],
                    Length = length[i],
                    Weight = weight[i],
                    Abundance = abundance[i],
                    TrophicLevel = tl[i],
                    Flags = flags,
                    FirstFeedingAgeDt = checked((short)ffa[i]),
                },
                new SchoolLife { StarvationRate = starv[i], CellId = cx[i] >= 0 && cy[i] >= 0 ? (cy[i] * nx) + cx[i] : -1 });
            store.Step[at].PredSuccess = psr[i];
            store.Ledger[at].DeadAging = aging[i];
        }

        int nFocal = p.Species.NFocal;
        double[] ssb = new double[nFocal], eggs = new double[nFocal];
        bool[] seeded = new bool[nFocal];
        switch (stage)
        {
            case "floor":
                LifeCycle.Floor(store.Core);
                break;
            case "growth":
                LifeCycle.Growth(p, store.Core, store.Step);
                break;
            case "aging":
                LifeCycle.Aging(p, store.Core, store.Ledger);
                break;
            case "reproduction":
                LifeCycle.Reproduction(p, store, step, ssb, eggs, seeded);
                break;
            default:
                throw new ArgumentException($"unknown stage {stage}", nameof(stage));
        }

        int m = store.Count;
        int[] oSp = new int[m], oAge = new int[m], oFfa = new int[m], oX = new int[m], oY = new int[m];
        double[] oLen = new double[m], oW = new double[m], oA = new double[m], oTl = new double[m], oStarv = new double[m], oAging = new double[m];
        byte[] oEgg = new byte[m], oSeed = new byte[m], oOut = new byte[m];
        for (int i = 0; i < m; i++)
        {
            SchoolCore c = store.Core[i];
            int cell = store.Life[i].CellId;
            oSp[i] = c.SpeciesId;
            oAge[i] = c.AgeDt;
            oFfa[i] = c.FirstFeedingAgeDt;
            oX[i] = cell >= 0 ? cell % nx : -1;
            oY[i] = cell >= 0 ? cell / nx : -1;
            oLen[i] = c.Length;
            oW[i] = c.Weight;
            oA[i] = c.Abundance;
            oTl[i] = c.TrophicLevel;
            oStarv[i] = store.Life[i].StarvationRate;
            oAging[i] = store.Ledger[i].DeadAging;
            oEgg[i] = (byte)((c.Flags & SchoolFlags.IsEgg) != 0 ? 1 : 0);
            oSeed[i] = (byte)((c.Flags & SchoolFlags.FromSeeding) != 0 ? 1 : 0);
            oOut[i] = (byte)((c.Flags & SchoolFlags.IsOut) != 0 ? 1 : 0);
        }

        byte[] oSeeded = seeded.Select(b => (byte)(b ? 1 : 0)).ToArray();
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
            output.SetArray("life.species_id", DType.I4, Pin(oSp), m);
            output.SetArray("life.age_dt", DType.I4, Pin(oAge), m);
            output.SetArray("life.first_feeding_age_dt", DType.I4, Pin(oFfa), m);
            output.SetArray("life.cell_x", DType.I4, Pin(oX), m);
            output.SetArray("life.cell_y", DType.I4, Pin(oY), m);
            output.SetArray("life.length", DType.F8, Pin(oLen), m);
            output.SetArray("life.weight", DType.F8, Pin(oW), m);
            output.SetArray("life.abundance", DType.F8, Pin(oA), m);
            output.SetArray("life.trophic_level", DType.F8, Pin(oTl), m);
            output.SetArray("life.starvation_rate", DType.F8, Pin(oStarv), m);
            output.SetArray("life.dead_aging", DType.F8, Pin(oAging), m);
            output.SetArray("life.is_egg", DType.U1, Pin(oEgg), m);
            output.SetArray("life.from_seeding", DType.U1, Pin(oSeed), m);
            output.SetArray("life.is_out", DType.U1, Pin(oOut), m);
            output.SetArray("life.ssb", DType.F8, Pin(ssb), nFocal);
            output.SetArray("life.eggs", DType.F8, Pin(eggs), nFocal);
            output.SetArray("life.seeded", DType.U1, Pin(oSeeded), nFocal);
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
