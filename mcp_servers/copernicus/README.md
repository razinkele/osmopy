# Copernicus Marine MCP Server

Downloads CMEMS datasets as OSMOSE-compatible NetCDF forcing files.

## Setup

1. Register at https://data.marine.copernicus.eu to obtain credentials.
2. Export credentials in your shell **before** launching Claude Code:

   ```bash
   export CMEMS_USERNAME="you@example.com"
   export CMEMS_PASSWORD="your-password"
   ```

   Or copy `.env.example` to `.env` and source it from your shell init.
3. Credentials are consumed by `server.py` via `os.environ.get`.
   There is **no hardcoded fallback** — missing env vars raise `RuntimeError`.

## Security

Never commit `CMEMS_PASSWORD` into `.mcp.json`, `server.py`, or any tracked file.
`tests/test_copernicus_mcp_env.py` and `tests/test_mcp_config_hygiene.py` enforce this.
