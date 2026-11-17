using BenchmarkDotNet.Running;

namespace Osmose.Engine.Bench;

/// <summary>
/// BenchmarkDotNet host. The kernel spike (plan task 0.6) adds the captured-step benchmarks.
/// Run only on an otherwise idle machine (plan rule 5):
/// <c>dotnet run -c Release --project csharp/bench/Osmose.Engine.Bench -- --filter '*'</c>.
/// </summary>
public static class Program
{
    public static void Main(string[] args) =>
        BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);
}
