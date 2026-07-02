---
name: Baltic grid validated against BITS + HELCOM
description: 2026-04-21 validation run; mask 100% correct vs 1,296 ICES BITS haul positions. Grid scope is SD 22-32 by design.
type: project
originSessionId: 1234202a-3436-4b5c-8595-2206a969a1ef
---
Baltic example grid (`data/baltic/baltic_grid.nc`, 50×40, lon 10-30°E, lat 54-66°N, 616 ocean cells) passed independent validation on 2026-04-21:

- **Mask correctness: 100% on 1,296 ICES BITS haul positions** (BITS 2021 Q1 + 2022 Q4 + 2023 Q1 + 2023 Q4). Zero hauls landed in cells marked as land.
- **Coverage**: 10/17 HELCOM sub-basins fully covered; Kattegat, Kiel Bay, Great Belt partially truncated west of 10°E; Gulf of Finland tip east of 29.8°E truncated.
- **ICES stock areas**: SD 22-32 stocks all covered. SD 20-21 stocks (`cod.27.21`, `her.27.20-24`, `ple.27.21-23`, `sol.27.20-24`) are out of scope — design choice for Central + Eastern Baltic.
- **Outside-grid hauls**: ~5-6 per BITS quarter in the Danish Straits (lat 55.07-55.37°N, lon 9.58-9.99°E). Consistent across years.

**Why:** follow-up to the LTL overlay fix on 2026-04-21. User wanted to know whether the grid we're overlaying LTL onto is actually geographically correct. Answer: yes, for its stated scope.

**How to apply:** don't re-run the full BITS cross-check unless (a) the grid mask changes or (b) the user wants to extend the grid westward to include Skagerrak/Kattegat (would need `grid.upleft.lon` 10→8 + mask regeneration). Validation harness is a short standalone Python script using `https://datras.ices.dk/WebServices/DATRASWebService.asmx/getHHdata` (no auth). DATRAS is public — ICES creds in .env are not needed for this check.

**Related fix shipped same day:** `/home/razinka/ices-mcp-server/ices/datras.py:374` corrected `getHH` → `getHHdata` (one-char typo; canonical R package `icesDatras` uses `getHHdata`). MCP server needs restart to reload.
