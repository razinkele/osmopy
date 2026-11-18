namespace Osmose.Engine;

/// <summary>How a run ended (spec section 8.5).</summary>
public enum RunStatus
{
    /// <summary>Every step ran.</summary>
    Completed,

    /// <summary>The control block's cancel slot was set; the first records are valid.</summary>
    Cancelled,

    /// <summary><c>run.deadline_seconds</c> passed; the first records are valid.</summary>
    TimedOut,
}

/// <summary>
/// What <see cref="OsmoseEngine.RunSimulation"/> returns besides the result buffers
/// (spec section 8.3). The remaining fields arrive with the step loop in Phase 1.
/// </summary>
public sealed class RunSummary
{
    internal RunSummary(RunStatus status, int stepsCompleted, int recordsWritten)
        : this(status, stepsCompleted, recordsWritten, 0, 0.0, 0)
    {
    }

    internal RunSummary(RunStatus status, int stepsCompleted, int recordsWritten, int peakSchools, double elapsedSeconds, int rngScheme)
    {
        Status = status;
        StepsCompleted = stepsCompleted;
        RecordsWritten = recordsWritten;
        PeakSchools = peakSchools;
        ElapsedSeconds = elapsedSeconds;
        RngScheme = rngScheme;
    }

    /// <summary>How the run ended.</summary>
    public RunStatus Status { get; }

    /// <summary>Steps fully simulated.</summary>
    public int StepsCompleted { get; }

    /// <summary>Output records written into the result buffers.</summary>
    public int RecordsWritten { get; }

    /// <summary>The largest live school count at any step.</summary>
    public int PeakSchools { get; }

    public double ElapsedSeconds { get; }

    /// <summary>The draw scheme the run used (spec 5).</summary>
    public int RngScheme { get; }
}
