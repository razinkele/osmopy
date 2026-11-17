namespace Osmose.Engine;

/// <summary>
/// The engine's single entry point (spec section 8.3). Python resolves every key, file and
/// name into a <see cref="SimulationConfig"/>; this method runs the whole simulation and
/// writes its results into buffers the caller owns.
/// </summary>
public static class OsmoseEngine
{
    /// <summary>
    /// Runs the whole simulation. Blocking, thread-safe and re-entrant: no static mutable
    /// state, one run object per call.
    /// </summary>
    /// <exception cref="NotImplementedException">
    /// Until Phase 1 of docs/superpowers/specs/2026-11-17-csharp-engine-implementation-plan.md.
    /// </exception>
    public static RunSummary RunSimulation(SimulationConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        throw new NotImplementedException(
            "The C# engine has no model yet: RunSimulation arrives in Phase 1 of "
            + "docs/superpowers/specs/2026-11-17-csharp-engine-implementation-plan.md.");
    }
}
