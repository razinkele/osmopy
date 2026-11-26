using System.Runtime.InteropServices;

using Osmose.Engine.Schools;

namespace Osmose.Engine.Step;

/// <summary>A restart snapshot as a bundle (<c>snap.*</c>): the focal schools of the store, in
/// store order, after a step. Python reads it and writes Java's NetCDF
/// (<c>osmose.engine.restart</c>); the engine itself writes no NetCDF.</summary>
public static class SnapshotWriter
{
    public static void Write(SchoolStore store, int nFocal, int step, string path)
    {
        ReadOnlySpan<SchoolCore> core = store.Core;
        ReadOnlySpan<SchoolLife> life = store.Life;
        var keep = new List<int>(core.Length);
        for (int i = 0; i < core.Length; i++)
        {
            if (core[i].SpeciesId < nFocal && (core[i].Flags & SchoolFlags.IsBackground) == 0)
            {
                keep.Add(i);
            }
        }

        int n = keep.Count;
        var species = new int[n];
        var age = new int[n];
        var cellId = new int[n];
        var abundance = new double[n];
        var length = new double[n];
        var weight = new double[n];
        var tl = new double[n];
        for (int k = 0; k < n; k++)
        {
            SchoolCore c = core[keep[k]];
            species[k] = c.SpeciesId;
            age[k] = c.AgeDt;
            cellId[k] = life[keep[k]].CellId;
            abundance[k] = c.Abundance;
            length[k] = c.Length;
            weight[k] = c.Weight;
            tl[k] = c.TrophicLevel;
        }

        var handles = new List<GCHandle>();
        try
        {
            var output = new SimulationConfig(1);
            output.SetArray("snap.step", DType.I4, Pin(handles, new[] { step }), 1);
            output.SetArray("snap.species_id", DType.I4, Pin(handles, species), Math.Max(n, 0));
            output.SetArray("snap.age_dt", DType.I4, Pin(handles, age), Math.Max(n, 0));
            output.SetArray("snap.cell_id", DType.I4, Pin(handles, cellId), Math.Max(n, 0));
            output.SetArray("snap.abundance", DType.F8, Pin(handles, abundance), Math.Max(n, 0));
            output.SetArray("snap.length", DType.F8, Pin(handles, length), Math.Max(n, 0));
            output.SetArray("snap.weight", DType.F8, Pin(handles, weight), Math.Max(n, 0));
            output.SetArray("snap.trophic_level", DType.F8, Pin(handles, tl), Math.Max(n, 0));
            output.Save(path);
        }
        finally
        {
            foreach (GCHandle h in handles)
            {
                h.Free();
            }
        }
    }

    private static long Pin(List<GCHandle> handles, Array a)
    {
        GCHandle h = GCHandle.Alloc(a, GCHandleType.Pinned);
        handles.Add(h);
        return h.AddrOfPinnedObject();
    }
}
