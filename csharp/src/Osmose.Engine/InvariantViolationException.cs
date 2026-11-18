namespace Osmose.Engine;

/// <summary>A whole-engine invariant checked under <c>run.checks</c> (spec 10.4) does not hold.
/// The Python wrapper raises it as <c>CSharpEngineError</c>.</summary>
public sealed class InvariantViolationException : Exception
{
    /// <summary>Names the school, the step and what failed.</summary>
    public InvariantViolationException(ulong schoolId, int step, string cause)
        : base($"invariant violated at step {step}, school {schoolId}: {cause}")
    {
        SchoolId = schoolId;
        Step = step;
        Cause = cause;
    }

    public ulong SchoolId { get; }

    public int Step { get; }

    public string Cause { get; }
}
