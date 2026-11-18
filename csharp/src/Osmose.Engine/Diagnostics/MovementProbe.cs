using System.Runtime.InteropServices;
using Osmose.Engine.Rng;
using Osmose.Engine.Schools;
using Osmose.Engine.Step;

namespace Osmose.Engine.Diagnostics;

/// <summary>
/// Test harness for plan task 1.6, callable through pythonnet: moves a captured school state for
/// one step and writes, per school in the input's order, where it went and the tier-E pieces
/// (map index, same-map flag, walk candidate set from its position before the move).
/// </summary>
public static class MovementProbe
{
    /// <summary>
    /// <paramref name="statePath"/>: a bundle of <c>state.species_id</c>, <c>age_dt</c>,
    /// <c>cell_x</c>, <c>cell_y</c>, <c>is_out</c>. With <paramref name="regroup"/> the store is
    /// regrouped (reordered by cell) before moving, which must not change any school's result.
    /// </summary>
    public static void Run(string modelPath, string statePath, int step, long seed, bool regroup, string outPath)
    {
        using SimulationConfig model = SimulationConfig.Load(modelPath);
        using SimulationConfig state = SimulationConfig.Load(statePath);
        EngineParameters p = EngineParameters.From(model, checkFeatures: false);
        MovementParameters m = p.Movement;
        var r = new BundleReader(state, 1);
        long n = r.Array("state.species_id", DType.I4, BundleReader.Any)!.Length;
        int[] species = r.Array("state.species_id", DType.I4, n)!.ToArray<int>();
        int[] age = r.Array("state.age_dt", DType.I4, n)!.ToArray<int>();
        int[] cx = r.Array("state.cell_x", DType.I4, n)!.ToArray<int>();
        int[] cy = r.Array("state.cell_y", DType.I4, n)!.ToArray<int>();
        byte[] isOut = r.Array("state.is_out", DType.U1, n)!.ToArray<byte>();
        r.ThrowIfProblems();

        int nx = p.Nx, ny = p.Ny;
        var store = new SchoolStore(ny * nx, initialCapacity: Math.Max(16, (int)n));
        int[] cellBefore = new int[n];
        for (int i = 0; i < n; i++)
        {
            cellBefore[i] = cx[i] >= 0 && cy[i] >= 0 ? (cy[i] * nx) + cx[i] : -1;
            store.Append(
                new SchoolCore { SpeciesId = species[i], AgeDt = age[i], Abundance = 1.0, Flags = isOut[i] != 0 ? SchoolFlags.IsOut : 0 },
                new SchoolLife { CellId = cellBefore[i] });
        }

        // Tier-E pieces from the position before the move, through the functions the walk uses.
        int[] mapIndex = new int[n];
        byte[] sameMap = new byte[n];
        long[] candOffsets = new long[n + 1];
        var candCells = new List<int>();
        int[] scratch = new int[Math.Max(1, Movement.ScratchLength(m))];
        for (int i = 0; i < n; i++)
        {
            int sp = species[i];
            byte method = sp < m.Method.Length ? m.Method[sp] : MovementParameters.MethodNone;
            mapIndex[i] = -1;
            int count = 0;
            if (method == MovementParameters.MethodMaps)
            {
                int idx = Movement.MapIndex(m, sp, age[i], step);
                bool same = Movement.SameMap(m, sp, age[i], step, idx);
                mapIndex[i] = idx;
                sameMap[i] = (byte)(same ? 1 : 0);
                int off = m.MapSpeciesOffset[sp];
                bool live = idx >= 0 && off >= 0 && off + idx < m.NMaps && m.MapIsNull[off + idx] == 0;
                if (live && same && cellBefore[i] >= 0)
                {
                    count = Movement.MapWalkCandidates(m, ny, nx, off + idx, cellBefore[i], m.WalkRange[sp], scratch);
                }
            }
            else if (method == MovementParameters.MethodRandom && cellBefore[i] >= 0 && m.WalkRange[sp] > 0)
            {
                count = Movement.RandomWalkCandidates(m, ny, nx, cellBefore[i], m.WalkRange[sp], scratch);
            }

            for (int k = 0; k < count; k++)
            {
                candCells.Add(scratch[k]);
            }

            candOffsets[i + 1] = candOffsets[i] + count;
        }

        if (regroup)
        {
            store.Regroup();
        }

        Movement.Run(p, store, step, PhiloxKey.ForRun(seed, Philox.Scheme), scratch);

        int[] outX = new int[n], outY = new int[n];
        byte[] outOut = new byte[n];
        for (int i = 0; i < store.Count; i++)
        {
            long id = (long)store.Life[i].Id;
            int cell = store.Life[i].CellId;
            outX[id] = cell >= 0 ? cell % nx : -1;
            outY[id] = cell >= 0 ? cell / nx : -1;
            outOut[id] = (byte)((store.Core[i].Flags & SchoolFlags.IsOut) != 0 ? 1 : 0);
        }

        int[] cand = candCells.Count > 0 ? candCells.ToArray() : [0];
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
            output.SetArray("move.cell_x", DType.I4, Pin(outX), n);
            output.SetArray("move.cell_y", DType.I4, Pin(outY), n);
            output.SetArray("move.is_out", DType.U1, Pin(outOut), n);
            output.SetArray("move.map_index", DType.I4, Pin(mapIndex), n);
            output.SetArray("move.same_map", DType.U1, Pin(sameMap), n);
            output.SetArray("move.cand.offsets", DType.I8, Pin(candOffsets), n + 1);
            output.SetArray("move.cand.cells", DType.I4, Pin(cand), cand.Length);
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
