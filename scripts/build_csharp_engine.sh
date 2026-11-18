#!/usr/bin/env bash
# Build the C# engine into osmose/engine_cs/bin (spec section 9.1): a framework-dependent,
# portable IL assembly plus the runtimeconfig.json pythonnet hosts it with.
#
# Finds dotnet the same way osmose.engine_cs.resolve_dotnet_root does: $DOTNET_ROOT, then
# ~/.dotnet, then dotnet on PATH. Publishes where the package loads from: $OSMOSE_CSHARP_ENGINE_DIR
# when set (osmose.engine_cs.assembly_dir), else osmose/engine_cs/bin. Extra arguments go to
# `dotnet publish` (e.g. -m:1 on a busy box).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -n "${DOTNET_ROOT:-}" && -x "$DOTNET_ROOT/dotnet" ]]; then
  DOTNET="$DOTNET_ROOT/dotnet"
elif [[ -x "$HOME/.dotnet/dotnet" ]]; then
  DOTNET="$HOME/.dotnet/dotnet"
elif command -v dotnet >/dev/null 2>&1; then
  DOTNET="$(command -v dotnet)"
else
  echo "error: no dotnet found (set DOTNET_ROOT, install to ~/.dotnet, or put dotnet on PATH)." >&2
  echo "See csharp/README.md for the no-sudo install." >&2
  exit 1
fi

export DOTNET_CLI_TELEMETRY_OPTOUT=1
export DOTNET_NOLOGO=1

OUT="${OSMOSE_CSHARP_ENGINE_DIR:-$ROOT/osmose/engine_cs/bin}"
"$DOTNET" publish "$ROOT/csharp/src/Osmose.Engine/Osmose.Engine.csproj" \
  -c Release -o "$OUT" "$@"
echo "C# engine built into $OUT"
