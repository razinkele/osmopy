# Deploying the OSMOSE web app

The production web app is a **systemd service** running Uvicorn directly:

- **Service:** `osmose-shiny.service` ("OSMOSE Python Shiny App (direct Uvicorn)")
- **Command:** `uvicorn app:app --host 127.0.0.1 --port 8838 --root-path /osmose`
- **Runs as:** user `shiny`, from this source tree (osmose is imported from the working
  copy — it is **not** pip-installed in the env), using `/opt/micromamba/envs/shiny`.
- **Public URL:** behind nginx at `/osmose/`.

## Deploying a change — ALWAYS restart after pulling

```bash
git pull                                      # update the source
sudo systemctl restart osmose-shiny.service   # REQUIRED — load the new code
systemctl status osmose-shiny.service         # confirm it came back up
```

**A `git pull` alone does NOT update the running app.** The service starts Uvicorn
**without** `--reload`, so it loads Python modules once at startup and keeps them in
memory until restarted. Skipping the restart leaves stale code serving requests.

## Why skipping the restart causes confusing errors (not just "old behaviour")

The engine imports some modules **lazily, at run time** — e.g. `simulate.py` imports
`osmose.engine.processes.mortality` *inside* `_mortality()`, not at module load. In a
long-running process this can mix versions: the OLD `osmose.engine.config`
(`EngineConfig`) imported at startup is used to build the config, while the FIRST
simulation after a `git pull` lazily imports the NEW `mortality.py`. If the new code
reads a field the old `EngineConfig` doesn't have, you get a run-time error like:

```
'EngineConfig' object has no attribute 'fr_shape'
```

…that appears **only through the long-running GUI** and cannot be reproduced from a
fresh `.venv` script run. The cause is always the stale service, never a code bug
(`EngineConfig.from_dict` is the only constructor and sets every field). The fix is to
restart the service.

**Rule of thumb:** if a GUI-only "missing attribute / unexpected field" error appears
that you can't reproduce with `.venv/bin/python`, check the service start time
(`ps -o lstart= -p $(systemctl show -p MainPID --value osmose-shiny.service)`) against
when the field was added — then `sudo systemctl restart osmose-shiny.service`.
