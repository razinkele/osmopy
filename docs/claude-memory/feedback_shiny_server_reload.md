---
name: OSMOSE runs as standalone Uvicorn, not via shiny-server
description: App runs as osmose-shiny.service on port 8838, proxied by nginx. Restart service after code changes.
type: feedback
originSessionId: a0e6a56e-90ae-4dac-b2d1-150ae013ea8f
---
OSMOSE runs as a standalone Uvicorn service (`osmose-shiny.service`) on port 8838, proxied directly by nginx. It does NOT use shiny-server (which has WebSocket 403 issues with Python Shiny 1.5+ / Starlette 0.52+).

**Why:** shiny-server v1.5.22 rejects WebSocket upgrades from Python Shiny's Uvicorn backend with HTTP 403. Direct Uvicorn works perfectly.

**How to apply:** After code changes, restart the service:
- `sudo systemctl restart osmose-shiny` or `sudo bash deploy.sh --restart`
- Check logs: `journalctl -u osmose-shiny -f`
- The app does NOT auto-reload — always restart after changes.
- Deploy URL: `https://laguna.ku.lt/osmose/`
