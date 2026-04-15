---
name: deploy
description: Deploy the OSMOSE app to the production server (direct Uvicorn service, not shiny-server)
disable-model-invocation: true
---

Deploy OSMOSE to the production server.

## Architecture

OSMOSE runs as a standalone Uvicorn service (`osmose-shiny.service`) on port 8838,
proxied directly by nginx. This bypasses shiny-server which has WebSocket compatibility
issues with Python Shiny 1.5+ / Starlette 0.52+.

## Steps

1. Run tests from the project root (`/home/razinka/osmose/osmose-python/`):
   ```
   .venv/bin/python -m pytest -x -q
   ```

2. Run lint:
   ```
   .venv/bin/ruff check osmose/ ui/ tests/
   ```

3. If BOTH pass, deploy:
   ```
   sudo bash deploy.sh
   ```

4. Verify deployment is accessible at `https://laguna.ku.lt/osmose/`

## Quick restart (after code changes)

```
sudo bash deploy.sh --restart
```

Or equivalently:
```
sudo systemctl restart osmose-shiny
```

## Logs

```
journalctl -u osmose-shiny -f
```

## Rules

- Do NOT deploy if tests or lint fail. Report failures and stop.
- The deploy script requires `sudo` — confirm with the user before running.
- Always run from `/home/razinka/osmose/osmose-python/`.
- After code changes, restart the service — Uvicorn does NOT auto-reload in production.
