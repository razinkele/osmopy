# Deploying the OSMOSE web app

The production web app is a **systemd service** running Uvicorn directly:

- **Service:** `osmose-shiny.service` ("OSMOSE Python Shiny App (direct Uvicorn)")
- **Command:** `uvicorn app:app --host 127.0.0.1 --port 8838 --root-path /osmose`
- **Runs as:** user `shiny`, from this source tree (osmose is imported from the working
  copy — it is **not** pip-installed in the env), using `/opt/micromamba/envs/shiny`.
- **Public URL:** behind nginx at `/osmose/`.
- **Supported runtime:** shiny **1.6.x** (`shiny>=1.6.3,<1.7`), shinyswatch ≥0.11, shinywidgets ≥0.7, cma ≥4.0, shiny_deckgl `v1.9.2`. To upgrade the shared env in place: `pip install --upgrade "cma>=4.0" "shinyswatch>=0.11" "shinywidgets>=0.7"` and reinstall `shiny_deckgl @v1.9.2`, then restart the service.
- **Writable state (calibration):** calibration checkpoints must NOT use the package default (`<repo>/data/baltic/calibration_results`). The unit sets `StateDirectory=osmose/calibration_results` + `Environment=OSMOSE_RESULTS_DIR=/var/lib/osmose/calibration_results` (systemd creates the dir, owned by `shiny`, on every start). Without this you'll see `RESULTS_DIR probe failed: [Errno 13] Permission denied` at startup and calibration checkpoint writes will fail.
  - **The unit's own comment says the source tree "lives under another user's home and is read-only to `shiny`". That is STALE — do not reason from it.** Measured 2026-09-14: `/srv/shiny-server/osmose` is a symlink to `/srv/shiny-server/osmose-src`, owned `shiny:shiny`; `.../osmose/data` is `drwxr-xr-x shiny shiny`; and the unit sets neither `ProtectSystem=` nor `ReadOnlyPaths=`. The service user CAN write inside the tree. The `OSMOSE_RESULTS_DIR` setting above is still correct and should stay — but a prediction built on the read-only premise was made from this comment and turned out to be wrong, so verify before relying on it.
- **Feedback system — set `OSMOSE_FEEDBACK_TOKEN` or the feature is write-only.** Measured 2026-09-14: `systemctl show osmose-shiny.service -p Environment` returns only `PYTHONUNBUFFERED=1` and `OSMOSE_RESULTS_DIR=...`. With the token unset, `check_feedback_token` returns `False` for every caller, so **both `/feedback/review` and `/api/feedback` answer `403`** — the form accepts and stores feedback that nobody can read back through the app. Add `Environment=OSMOSE_FEEDBACK_TOKEN=<long random string>` and restart.
- **Behind nginx, set `OSMOSE_TRUSTED_PROXY=1`.** nginx forwards `X-Forwarded-For`, but the app ignores the header unless this variable is set, and then keys the rate limiter on `client.host` — which behind nginx is nginx. Measured consequence: **every user of the feedback form currently shares one bucket of 5 submissions per hour.** The app already logs `X-Forwarded-For present but OSMOSE_TRUSTED_PROXY is unset` once per process when this happens.
- **Feedback store paths (optional).** The defaults (`<repo>/data/feedback/{feedback,contacts}.jsonl`) work — the tree is service-user writable (see above). Pointing them outside the source tree is hygiene, not a fix: `StateDirectory=osmose/feedback` plus `Environment=OSMOSE_FEEDBACK_FILE=/var/lib/osmose/feedback/feedback.jsonl` and `Environment=OSMOSE_CONTACTS_FILE=/var/lib/osmose/feedback/contacts.jsonl`. `contacts.jsonl` holds reporter **email addresses** — keep it off any path that gets copied, backed up publicly, or served. Full operator guide: [`docs/feedback-runbook.md`](docs/feedback-runbook.md).

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
