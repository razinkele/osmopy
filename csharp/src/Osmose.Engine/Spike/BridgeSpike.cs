using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Osmose.Engine.Spike;

/// <summary>
/// Throwaway targets for the pythonnet bridge spike (plan task 0.2, gate G0). Each method
/// exercises one mechanism the real entry point will rely on: borrowed NumPy memory passed
/// as an <see cref="long"/> address, the GIL released around a call, the shared control
/// block, exception propagation and concurrent callers. Delete once Phase 1's
/// <see cref="OsmoseEngine.RunSimulation"/> covers the same checks.
/// </summary>
public static class BridgeSpike
{
    /// <summary>The runtime pythonnet actually loaded, e.g. ".NET 10.0.15".</summary>
    public static string FrameworkDescription() => RuntimeInformation.FrameworkDescription;

    /// <summary>Multiplies a caller-owned float64 buffer in place (no copy either way).</summary>
    public static unsafe void ScaleInPlace(long address, long length, double factor)
    {
        var values = new Span<double>((void*)address, checked((int)length));
        for (int i = 0; i < values.Length; i++)
        {
            values[i] *= factor;
        }
    }

    /// <summary>Sums a caller-owned float64 buffer after a delay, so two callers overlap.</summary>
    public static unsafe double SumAfterDelay(long address, long length, int delayMilliseconds)
    {
        Thread.Sleep(delayMilliseconds);
        var values = new ReadOnlySpan<double>((void*)address, checked((int)length));
        double total = 0.0;
        foreach (double v in values)
        {
            total += v;
        }

        return total;
    }

    /// <summary>Busy-waits for <paramref name="seconds"/>, standing in for a compute-bound run.</summary>
    public static long Spin(double seconds)
    {
        var clock = Stopwatch.StartNew();
        long iterations = 0;
        while (clock.Elapsed.TotalSeconds < seconds)
        {
            iterations++;
        }

        return iterations;
    }

    /// <summary>
    /// Loops until slot 0 of the caller's int64 control block is non-zero, writing the
    /// iteration count to slot 1 after each iteration (the protocol of spec section 8.5).
    /// Returns the number of completed iterations.
    /// </summary>
    public static unsafe long LoopUntilCancelled(long controlAddress, long maxIterations, int sleepMilliseconds)
    {
        long* control = (long*)controlAddress;
        long done = 0;
        while (done < maxIterations && Volatile.Read(ref control[0]) == 0)
        {
            Thread.Sleep(sleepMilliseconds);
            done++;
            Volatile.Write(ref control[1], done);
        }

        return done;
    }

    /// <summary>Throws, so the spike can check what Python receives.</summary>
    public static void Throw(string message) => throw new InvalidOperationException(message);
}
