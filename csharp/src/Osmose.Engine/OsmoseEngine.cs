using System.Runtime.CompilerServices;

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
    /// <exception cref="BundleValidationException">The bundle is malformed (every problem listed).</exception>
    /// <exception cref="UnsupportedFeatureException">
    /// The bundle needs a feature this engine does not implement. Until task 1.8 that includes
    /// <c>core</c>, so every valid bundle is refused here, by name (fail closed, spec D11).
    /// </exception>
    public static RunSummary RunSimulation(SimulationConfig config)
    {
        try
        {
            return RunOnce(config);
        }
        finally
        {
            // The run's own arrays (the school store, the scratch, the tables) are dead here.
            // Left to the collector's schedule they stay for two or three more runs: measured on
            // Baltic (2026-11-19), about 30 MB a run, and a process that runs many simulations
            // sat at 270 to 320 MB resident against 204 MB with this collection, which took
            // 4 ms. Without it the engine can exceed the Python engine's peak (spec 10.4).
            GC.Collect();
        }
    }

    /// <summary>
    /// One run, in its own frame: the run object must be unreachable when
    /// <see cref="RunSimulation"/> collects, so this method is never inlined into it.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static RunSummary RunOnce(SimulationConfig config)
    {
        EngineParameters parameters = EngineParameters.From(config);
        return new Step.SimulationRun(parameters, config).Run();
    }
}
