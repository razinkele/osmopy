namespace Osmose.Engine.Schools;

/// <summary>
/// All schools of a run in the split array-of-structs layout (spec D2, plan task 1.3): parallel
/// arrays of <see cref="SchoolCore"/>, <see cref="SchoolStep"/>, <see cref="SchoolLedger"/> and
/// <see cref="SchoolLife"/>, plus a contiguous length lane.
/// </summary>
/// <remarks>
/// <para><see cref="Regroup"/> is the one reorder per step (spec 3.4). It is a stable counting sort
/// on <see cref="SchoolLife.CellId"/> from the front buffers into the back buffers that, in the same
/// sweep, drops schools whose abundance is exactly zero (the compaction), lays each cell's schools
/// out contiguously, puts unlocated and out-of-domain schools (<c>CellId == -1</c>) in a tail
/// segment, and copies the lengths into the lane. Within a cell, schools keep their previous
/// relative order, so the order is deterministic.</para>
/// <para>Capacity doubles when exhausted, for every array together, and never shrinks during a run,
/// so after warm-up neither <see cref="Append"/> nor <see cref="Regroup"/> allocates.
/// <see cref="SchoolStep"/> is rebuilt every step and is not carried across a regroup.</para>
/// </remarks>
public sealed class SchoolStore
{
    private SchoolCore[] _core;
    private SchoolLedger[] _ledger;
    private SchoolLife[] _life;
    private SchoolCore[] _coreBack;
    private SchoolLedger[] _ledgerBack;
    private SchoolLife[] _lifeBack;
    private SchoolStep[] _step;
    private double[] _lane;
    private readonly int[] _cellStart;
    private readonly int[] _cursor;
    private ulong _nextId;

    /// <summary>A store for a grid of <paramref name="nCells"/> cells.</summary>
    public SchoolStore(int nCells, int initialCapacity = 1024)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(nCells);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(initialCapacity);
        CellCount = nCells;
        _core = new SchoolCore[initialCapacity];
        _ledger = new SchoolLedger[initialCapacity];
        _life = new SchoolLife[initialCapacity];
        _coreBack = new SchoolCore[initialCapacity];
        _ledgerBack = new SchoolLedger[initialCapacity];
        _lifeBack = new SchoolLife[initialCapacity];
        _step = new SchoolStep[initialCapacity];
        _lane = new double[initialCapacity];
        _cellStart = new int[nCells + 2]; // cells, then the tail segment's end
        _cursor = new int[nCells + 1];
    }

    /// <summary>Grid cells.</summary>
    public int CellCount { get; }

    /// <summary>Live schools (including the unlocated tail).</summary>
    public int Count { get; private set; }

    /// <summary>Allocated slots per array.</summary>
    public int Capacity => _core.Length;

    /// <summary>The next id <see cref="Append"/> will assign.</summary>
    public ulong NextId => _nextId;

    public Span<SchoolCore> Core => _core.AsSpan(0, Count);

    public Span<SchoolStep> Step => _step.AsSpan(0, Count);

    public Span<SchoolLedger> Ledger => _ledger.AsSpan(0, Count);

    public Span<SchoolLife> Life => _life.AsSpan(0, Count);

    /// <summary>A copy of each school's length, contiguous per cell after a regroup.</summary>
    public ReadOnlySpan<double> LengthLane => _lane.AsSpan(0, Count);

    /// <summary>After a regroup: cell <c>c</c>'s schools are <c>[CellStart[c], CellStart[c + 1])</c>, and
    /// the unlocated tail is <c>[CellStart[CellCount], CellStart[CellCount + 1])</c>.</summary>
    public ReadOnlySpan<int> CellStart => _cellStart;

    /// <summary>The underlying arrays, for kernels that address a cell by offset.</summary>
    internal SchoolCore[] CoreArray => _core;

    internal SchoolStep[] StepArray => _step;

    internal SchoolLedger[] LedgerArray => _ledger;

    internal double[] LaneArray => _lane;

    /// <summary>Prey columns of the per-school diet rows (focal species, then resources);
    /// 0 when the diet is not tracked.</summary>
    public int DietCols { get; private set; }

    /// <summary>The per-school diet rows of the current step, <c>[position * DietCols + column]</c>:
    /// the biomass each school ate of each prey column, in tonnes. Per-step scratch like
    /// <see cref="Step"/>: valid from <see cref="ClearDiet"/> (the start of a mortality step, on
    /// the regrouped store) until the store is next reordered or compacted.</summary>
    internal double[] DietArray { get; private set; } = [];

    /// <summary>The diet rows of the schools, for reading.</summary>
    public ReadOnlySpan<double> Diet => DietArray.AsSpan(0, Count * DietCols);

    /// <summary>Turns diet tracking on, with <paramref name="columns"/> prey columns.</summary>
    public void EnableDiet(int columns)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(columns);
        DietCols = columns;
    }

    /// <summary>Zeroes the diet rows of every school, growing the array with the store. Nothing
    /// to do when the diet is not tracked.</summary>
    public void ClearDiet()
    {
        if (DietCols == 0)
        {
            return;
        }

        if (DietArray.Length < Count * DietCols)
        {
            DietArray = new double[_core.Length * DietCols];
        }
        else
        {
            DietArray.AsSpan(0, Count * DietCols).Clear();
        }
    }

    /// <summary>Columns of the staged diet rows (osmose.engine.diet_stage); 0 when not tracked.</summary>
    public int StagedCols { get; private set; }

    /// <summary>The per-school staged diet rows of the current step,
    /// <c>[position * StagedCols + staged column]</c>, valid like <see cref="DietArray"/>.</summary>
    internal double[] StagedArray { get; private set; } = [];

    /// <summary>Each school's staged prey column this step: its species' first column plus its
    /// stage at predation (Java's Prey snapshot).</summary>
    internal int[] StagedPreyCol { get; private set; } = [];

    public ReadOnlySpan<double> Staged => StagedArray.AsSpan(0, Count * StagedCols);

    /// <summary>Turns staged diet tracking on.</summary>
    public void EnableStagedDiet(int columns)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(columns);
        StagedCols = columns;
    }

    /// <summary>Zeroes the staged rows and sizes the prey columns for the current schools.</summary>
    public void ClearStagedDiet()
    {
        if (StagedCols == 0)
        {
            return;
        }

        if (StagedArray.Length < Count * StagedCols)
        {
            StagedArray = new double[_core.Length * StagedCols];
        }
        else
        {
            StagedArray.AsSpan(0, Count * StagedCols).Clear();
        }

        if (StagedPreyCol.Length < Count)
        {
            StagedPreyCol = new int[_core.Length];
        }
    }

    /// <summary>Appends one school, assigning the next id. Reproduction and incoming flux append
    /// serially, so ids are deterministic. Returns the new school's index.</summary>
    public int Append(in SchoolCore core, in SchoolLife life)
    {
        if (Count == _core.Length)
        {
            Grow();
        }

        int i = Count++;
        _core[i] = core;
        _ledger[i] = default;
        _step[i] = default;
        _life[i] = life;
        _life[i].Id = _nextId++;
        _lane[i] = core.Length;
        return i;
    }

    /// <summary>The one reorder per step (class remarks).</summary>
    public void Regroup()
    {
        int n = Count;
        int nCells = CellCount;
        Array.Clear(_cursor);
        int tail = 0;
        for (int i = 0; i < n; i++)
        {
            if (_core[i].Abundance == 0.0)
            {
                continue;
            }

            int c = _life[i].CellId;
            if ((uint)c < (uint)nCells)
            {
                _cursor[c + 1]++;
            }
            else
            {
                tail++;
            }
        }

        for (int c = 0; c < nCells; c++)
        {
            _cursor[c + 1] += _cursor[c];
        }

        int located = _cursor[nCells];
        _cursor.AsSpan(0, nCells + 1).CopyTo(_cellStart);
        _cellStart[nCells + 1] = located + tail;
        int tailAt = located;
        for (int i = 0; i < n; i++)
        {
            if (_core[i].Abundance == 0.0)
            {
                continue;
            }

            int c = _life[i].CellId;
            int at = (uint)c < (uint)nCells ? _cursor[c]++ : tailAt++;
            _coreBack[at] = _core[i];
            _ledgerBack[at] = _ledger[i];
            _lifeBack[at] = _life[i];
            _lane[at] = _core[i].Length;
        }

        (_core, _coreBack) = (_coreBack, _core);
        (_ledger, _ledgerBack) = (_ledgerBack, _ledger);
        (_life, _lifeBack) = (_lifeBack, _life);
        Count = located + tail;
    }

    /// <summary>
    /// Removes every background school, keeping the others in their order and with their step
    /// data (the background schools of a step leave right after mortality: the life cycle and the
    /// outputs must not see them). The cell index (<see cref="CellStart"/>) is stale afterwards,
    /// until the next <see cref="Regroup"/>. Returns how many were removed.
    /// </summary>
    public int RemoveBackground()
    {
        int n = Count, kept = 0;
        for (int i = 0; i < n; i++)
        {
            if ((_core[i].Flags & SchoolFlags.IsBackground) != 0)
            {
                continue;
            }

            if (kept != i)
            {
                _core[kept] = _core[i];
                _ledger[kept] = _ledger[i];
                _life[kept] = _life[i];
                _step[kept] = _step[i];
                _lane[kept] = _lane[i];
            }

            kept++;
        }

        Count = kept;
        return n - kept;
    }

    private void Grow()
    {
        int capacity = _core.Length * 2;
        Array.Resize(ref _core, capacity);
        Array.Resize(ref _ledger, capacity);
        Array.Resize(ref _life, capacity);
        Array.Resize(ref _coreBack, capacity);
        Array.Resize(ref _ledgerBack, capacity);
        Array.Resize(ref _lifeBack, capacity);
        Array.Resize(ref _step, capacity);
        Array.Resize(ref _lane, capacity);
    }
}
