---
name: deploy
description: Deploy the OSMOSE app to the local Shiny Server after verifying tests and lint pass
disable-model-invocation: true
---

Deploy OSMOSE to the local Shiny Server.

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

4. Verify deployment is accessible at `http://localhost:3838/osmose/`

## Rules

- Do NOT deploy if tests or lint fail. Report failures and stop.
- The deploy script requires `sudo` — confirm with the user before running.
- Always run from `/home/razinka/osmose/osmose-python/`.
