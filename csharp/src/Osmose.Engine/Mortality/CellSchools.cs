using Osmose.Engine.Schools;

namespace Osmose.Engine.Mortality;

/// <summary>How the kernel reaches a cell's schools. A struct type parameter, so the JIT
/// specialises the kernel for each layout (kernel spike, plan task 0.6).</summary>
public interface ICellSchools
{
    /// <summary>Schools in the cell.</summary>
    int Count { get; }

    /// <summary>The cell's school lengths, contiguous and in cell order.</summary>
    ReadOnlySpan<double> Lengths { get; }

    ref SchoolCore Core(int position);

    ref SchoolStep Step(int position);

    ref SchoolLedger Ledger(int position);
}

/// <summary>Regrouped layout (spec D2): the cell's schools are one contiguous range of the
/// store, and the length lane is a slice of the store's lane.</summary>
public readonly struct ContiguousCell : ICellSchools
{
    private readonly SchoolCore[] _core;
    private readonly SchoolStep[] _step;
    private readonly SchoolLedger[] _ledger;
    private readonly double[] _lane;
    private readonly int _start;

    public ContiguousCell(SchoolCore[] core, SchoolStep[] step, SchoolLedger[] ledger, double[] lane, int start, int count)
    {
        _core = core;
        _step = step;
        _ledger = ledger;
        _lane = lane;
        _start = start;
        Count = count;
    }

    public int Count { get; }

    public ReadOnlySpan<double> Lengths => new(_lane, _start, Count);

    public ref SchoolCore Core(int position) => ref _core[_start + position];

    public ref SchoolStep Step(int position) => ref _step[_start + position];

    public ref SchoolLedger Ledger(int position) => ref _ledger[_start + position];
}

/// <summary>Python's layout: schools stay in creation order and a cell is an index list. The
/// lengths are gathered into scratch once per cell (counted in the kernel's time).</summary>
public readonly struct IndexedCell : ICellSchools
{
    private readonly SchoolCore[] _core;
    private readonly SchoolStep[] _step;
    private readonly SchoolLedger[] _ledger;
    private readonly int[] _indices;
    private readonly int _start;
    private readonly double[] _gathered;

    public IndexedCell(SchoolCore[] core, SchoolStep[] step, SchoolLedger[] ledger, int[] indices, int start, int count, double[] gatherBuffer)
    {
        _core = core;
        _step = step;
        _ledger = ledger;
        _indices = indices;
        _start = start;
        Count = count;
        _gathered = gatherBuffer;
        for (int i = 0; i < count; i++)
        {
            gatherBuffer[i] = core[indices[start + i]].Length;
        }
    }

    public int Count { get; }

    public ReadOnlySpan<double> Lengths => new(_gathered, 0, Count);

    public ref SchoolCore Core(int position) => ref _core[_indices[_start + position]];

    public ref SchoolStep Step(int position) => ref _step[_indices[_start + position]];

    public ref SchoolLedger Ledger(int position) => ref _ledger[_indices[_start + position]];
}
