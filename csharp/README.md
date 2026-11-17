# C# engine

A third simulation backend beside the Python (NumPy/Numba) and Java engines: a .NET 10 library
behind one entry point, `OsmoseEngine.RunSimulation(SimulationConfig)`, called from Python
through pythonnet. Python keeps the Shiny UI, the Copernicus downloads, the xarray reading and
the config parsing.

- Design: `docs/superpowers/specs/2026-11-17-csharp-engine-design.md`
- Plan: `docs/superpowers/specs/2026-11-17-csharp-engine-implementation-plan.md`

**Status: Phase 0.** The skeleton and the pythonnet bridge spike are done. `RunSimulation` throws
`NotImplementedException` until Phase 1.

## Layout

```
csharp/
  Osmose.Engine.sln
  Directory.Build.props        net10.0, nullable, warnings as errors, invariant globalization
  src/Osmose.Engine/           the engine library (unsafe allowed; emits runtimeconfig.json)
  tests/Osmose.Engine.Tests/   xUnit
  bench/Osmose.Engine.Bench/   BenchmarkDotNet
osmose/engine_cs/              Python side; importing it never loads the CLR
  __init__.py                  runtime/assembly resolution, csharp_engine_available()
  bridge.py                    the only module that imports pythonnet
  bin/                         build output (gitignored)
scripts/build_csharp_engine.sh dotnet publish into osmose/engine_cs/bin
```

## Developer setup (no sudo)

1. Install the .NET 10 SDK into your home directory with Microsoft's installer:
   ```bash
   curl -sSL -o /tmp/dotnet-install.sh https://dot.net/v1/dotnet-install.sh
   bash /tmp/dotnet-install.sh --channel 10.0 --install-dir "$HOME/.dotnet"
   ```
   The installer only writes into `--install-dir`. Nothing needs to be on `PATH`, and
   `DOTNET_ROOT` does not need to be set: the build script and the Python bridge both look in
   `$DOTNET_ROOT`, then `~/.dotnet`, then `dotnet` on `PATH`.
2. Install the Python side: `.venv/bin/pip install -e ".[dev]"` (`[dev]` includes pythonnet;
   end users can use `pip install -e ".[csharp]"` instead).
3. Build the engine into the package: `bash scripts/build_csharp_engine.sh`.
   Extra arguments go to `dotnet publish`. On a busy machine, add `-m:1`.
4. Run the tests:
   ```bash
   OSMOSE_REQUIRE_CSHARP=1 .venv/bin/python -m pytest tests/test_csharp_*.py
   ~/.dotnet/dotnet test csharp/Osmose.Engine.sln -c Release
   ```
   Without `OSMOSE_REQUIRE_CSHARP=1`, the Python bridge tests skip when the engine is not built.
   With it, a missing engine is a failure, which is how CI runs them.

Before committing C# changes, run `~/.dotnet/dotnet format csharp/Osmose.Engine.sln`. CI runs it
with `--verify-no-changes`.

## Rules the bridge depends on

- **Never host the CLR in a process that later forks.** CoreCLR's runtime threads do not survive
  `fork`. The bridge tests therefore run their CLR checks in a subprocess. Calibration pools
  use spawn/forkserver, so each worker hosts its own runtime.
- **pythonnet releases the GIL around every call** (measured by the bridge spike: a Python thread
  kept ticking through a 2 s C# call). The UI relies on this.
- **pythonnet is given both `runtime_config` and `dotnet_root`.** clr_loader 0.3.1 ignores
  `dotnet_root` when no runtime config is passed (spec section 9.4).
- **Buffers are borrowed.** Python owns every NumPy array whose address it passes, and must keep
  it alive and unmoved until the call returns.
