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
- **Feedback system — set `OSMOSE_FEEDBACK_TOKEN` or the feature is write-only.** Measured 2026-09-14: `systemctl show osmose-shiny.service -p Environment` returns only `PYTHONUNBUFFERED=1` and `OSMOSE_RESULTS_DIR=...`. With the token unset, `check_feedback_token` returns `False` for every caller, so **both `/feedback/review` and `/api/feedback` answer `403`** — the form accepts and stores feedback that nobody can read back through the app. **Do NOT add it as `Environment=` in the unit** — `deploy.sh` rewrites the unit wholesale (`cat > "$SERVICE_FILE"`) on every deploy, and `systemctl show -p Environment` prints such values to any local user (demonstrated 2026-09-15 from an unprivileged account). Put it in a root-owned `0600` `EnvironmentFile` referenced from a `.d/` drop-in, which `deploy.sh` never touches: see [`docs/feedback-v2-deploy.md`](docs/feedback-v2-deploy.md) steps 1-2.
- **Behind nginx, set `OSMOSE_TRUSTED_PROXY=1`.** nginx forwards `X-Forwarded-For`, but the app ignores the header unless this variable is set, and then keys the rate limiter on `client.host` — which behind nginx is nginx. Measured consequence: **every user of the feedback form currently shares one bucket of 5 submissions per hour.** The app already logs `X-Forwarded-For present but OSMOSE_TRUSTED_PROXY is unset` once per process when this happens.
- **Feedback store paths (optional).** The defaults (`<repo>/data/feedback/{feedback,contacts}.jsonl`) work — the tree is service-user writable (see above). Pointing them outside the source tree is hygiene, not a fix — but if you do, note that `/var/lib/osmose` is `root:root` and the service user cannot create a subdirectory there, so the `StateDirectory=osmose/feedback` half is what makes the path writable and is NOT optional (without it `append_feedback`'s `mkdir` fails at the first submission). Set the two paths via the same EnvironmentFile as the token, not `Environment=`; see [`docs/feedback-v2-deploy.md`](docs/feedback-v2-deploy.md). `contacts.jsonl` holds reporter **email addresses** — keep it off any path that gets copied, backed up publicly, or served. Full operator guide: [`docs/feedback-runbook.md`](docs/feedback-runbook.md).

## What `deploy.sh` runs as root, and what as `shiny`

`sudo bash deploy.sh` runs as root, but **root never executes, or writes through, anything the service user can modify.**

**Run it only from the deploy host's own checkout** (`/home/razinka/osmopy`), never from `/srv/shiny-server/osmose` or `osmose-src`. Those trees are `shiny`-writable, so `sudo bash` there would run a script the service account controls. The script's own "not from the prod clone" check runs too late to help: it is part of the script being run. The prod clone (`.git` included) and the shared env (`/opt/micromamba/envs/shiny`, with its `python3`, `pip` and `site-packages`) are owned by `shiny`. Run as root:
- `git` would execute the clone's hooks, `core.fsmonitor` and filter drivers;
- the env's interpreter would execute its `.pth` files, or a replaced binary;
- `cp` into the clone would follow a planted symlink.

Any of those would let a compromised service account get root at the next deploy. So every `git` command on the clone, every run of the env's `python3` and `pip`, and every write into the clone goes through `runuser -u shiny`, under `setsid` so the child has no controlling terminal (it could otherwise inject keystrokes into root's terminal with the `TIOCSTI` ioctl). Root keeps only what it owns: the systemd unit, `/opt/dotnet`, the `/srv/shiny-server/osmose` symlink, and creating the empty clone directory. `tests/test_deploy_privileges.py` enforces this line by line.

Deploys before this change let root operate on the clone through a `safe.directory` grant in root's git config. The deploy now withdraws it.

Root does not re-own anything in the clone either: a hard link `shiny` planted there would carry the change to a file outside it. Instead, a read-only check stops the deploy, before the fetch, if anything in the clone is root-owned (the one-time step below covers it). On 2026-11-18 nothing was.

The Python dependencies (Step 0) are installed **before** the checkout. If pip fails, the deploy stops with the previous ref still checked out. Installs that completed before the failure stay in the shared env. They are version floors, which the running code already satisfies.

### One-time step before the first deploy with this change
Earlier deploys ran pip as root, so a few packages in the shared env are root-owned, and pip running as `shiny` cannot upgrade them. Measured 2026-11-18: `cma`, `pyarrow` and `shiny_deckgl`, with their `dist-info` directories (`pyodbc` too, but that is installed by conda for another app and not managed here). Nothing fails until one of them needs an upgrade. Then Step 0 stops the deploy, before the checkout, with a pip permission error. To hand them to the env's owner once, do it **with the service stopped**, from the deploy host's checkout:

```bash
SP=/opt/micromamba/envs/shiny/lib/python3.13/site-packages
sudo find "$SP" -maxdepth 1 -user root                 # what is root-owned in the env
sudo find /srv/shiny-server/osmose-src -user root      # and in the clone (empty on 2026-11-18)
sudo systemctl stop osmose-shiny.service
sudo chown -R --no-dereference shiny "$SP"/cma "$SP"/cma-*.dist-info "$SP"/pyarrow "$SP"/pyarrow-*.dist-info "$SP"/shiny_deckgl "$SP"/shiny_deckgl-*.dist-info
sudo bash deploy.sh                                     # restarts the service
```

If the clone listing showed anything, add those paths to the `chown`. Any `~`-prefixed leftover in `$SP` is the debris of an interrupted pip install; the deploy reports one it cannot remove, and it can be deleted.

**If that deploy stops before the restart** (for example at Step 0), the service is still stopped. The previous ref is still checked out, so bring it back as it was, and check it:

```bash
sudo systemctl start osmose-shiny.service
systemctl status osmose-shiny.service
```

Step 0 installs only version floors, which the old code already satisfies. If the env is nevertheless broken, repair it with pip, run as `shiny`, before starting the service.

This grants `shiny` nothing new. It owns `site-packages`, so it could already rename those directories away. It is a manual step on purpose: an automatic root `chown` on every deploy would itself be root acting on a path that runs through `shiny`-owned directories.

## The C# engine (optional, opt-in)

The third engine (`csharp/`, spec `docs/superpowers/specs/2026-11-17-csharp-engine-design.md` section 9.7) is **off by default**. The web app does not offer it before the Phase 2 UI work, so a plain `sudo bash deploy.sh` installs nothing for it. Every deploy also removes any assembly a previous deploy built, as `shiny` and before the checkout, so an old build never outlives its ref.

To build it on the server:

```bash
sudo OSMOSE_DEPLOY_CSHARP=1 bash deploy.sh
```

After the service is up and answering HTTP, the deploy:
1. installs the .NET 10 SDK into `/opt/dotnet` with Microsoft's `dotnet-install.sh`, if no 10.x SDK is there;
2. installs `pythonnet` into the shiny env;
3. builds the assembly from the deployed ref **as `shiny`**, never as root, into a staging directory;
4. runs the self-test on the staged build **as the service runs**: user `shiny`, the env's interpreter, the working copy and `DOTNET_ROOT=/opt/dotnet`;
5. only if it passes, moves the build into `osmose/engine_cs/bin` and checks that the service's own path can load it.

The unit always carries `Environment=DOTNET_ROOT=/opt/dotnet`, which is inert when the directory is absent. A failure in any step exits 1 with a `C# engine:` error, and a build that fails never becomes loadable. **The app keeps running the new code without the engine:** the steps run after the restart, so a C# failure can never leave new code on disk under the old process.

To re-check by hand:

```bash
cd /srv/shiny-server/osmose
sudo runuser -u shiny -- env DOTNET_ROOT=/opt/dotnet /opt/micromamba/envs/shiny/bin/python3 -m osmose.engine_cs.selftest
```

`--uninstall` leaves `/opt/dotnet` in place, since other software may use it. The Docker and Apptainer images build the engine unconditionally, and CI runs the same self-test in both.

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
