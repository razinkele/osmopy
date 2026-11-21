# Deploying the OSMOSE web app

The production web app is a **systemd service** running Uvicorn directly:

- **Service:** `osmose-shiny.service` ("OSMOSE Python Shiny App (direct Uvicorn)")
- **Command:** `uvicorn app:app --host 127.0.0.1 --port 8838 --root-path /osmose`
- **Runs as:** user `shiny`, from this source tree (osmose is imported from the working
  copy — it is **not** pip-installed in the env), using `/opt/micromamba/envs/shiny`.
- **Public URL:** behind nginx at `/osmose/`.
- **Supported runtime:** CI tests shiny **1.9.x** (`shiny>=1.9.0,<1.10`, `pyproject.toml`) with shiny_deckgl `v1.16.1`, the versions the shared env runs (measured 2026-11-21: shiny 1.9.0, shiny_deckgl 1.16.1, shinyswatch 0.14.0, shinywidgets 0.8.2; other apps installed them, and CI follows them). The deploy enforces floors only: shiny ≥1.9.0 (checked, never installed), shinyswatch ≥0.11, shinywidgets ≥0.7, cma ≥4.0, pyarrow ≥14, shiny_deckgl ≥1.16.1, pyvis ≥4.2. If the env's owner moves shiny or shiny_deckgl again, move `pyproject.toml` with it so CI keeps testing what prod runs. Do not upgrade or downgrade the shared env for osmose: see "The shared environment" below.
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

Root does not re-own anything in the clone either: a hard link `shiny` planted there would carry the change to a file outside it. Instead, a read-only check stops the deploy, before the fetch, if anything in the clone is root-owned (see "Root-owned files in the clone" below). On 2026-11-18 and 2026-11-21 nothing was.

The Python dependencies (Step 0) are installed **before** the checkout. If pip fails, the deploy stops with the previous ref still checked out. Step 0 installs only what is missing or below a floor (pip may also raise a dependency that a newly installed package requires at a newer version), so installs that completed before a failure only raised packages to floors the running code already satisfies.

### The shared environment
`/opt/micromamba/envs/shiny` is shared with the server's other apps and managed with micromamba, not by this deploy. Measured 2026-11-21: 740 of the 869 entries in its `site-packages` are root-owned, from micromamba runs as root (2026-11-18 22:05; `shiny_deckgl` 1.16.1 on 2026-11-20, which AQUABC and CENOP require at ≥1.16).

So Step 0 installs only what is missing or below a floor (`tests/test_deploy_floors.py`); pip may also raise a dependency that such an install requires at a newer version, but never replaces a package that already meets its floor just to reach a newer release. On that env it installs nothing. Every pip call passes `--no-user`, so an unmet floor that pip cannot write into the env fails the deploy instead of landing in `shiny`'s home. **Do not `chown` packages to `shiny` for osmose's sake.** An earlier version of this section did, so that pip could *upgrade* them; with the Step 0 of that time (`--upgrade` to the newest release, and `shiny_deckgl` held at exactly 1.9.2) it would have downgraded `shiny_deckgl` under the other apps. If a floor is ever unmet on a root-owned package, Step 0 stops the deploy before the checkout with a pip permission error, the service keeps running the previous ref, and the fix belongs to whoever manages the env: raise the package with micromamba, then deploy again.

Any `~`-prefixed leftover in `site-packages` is the debris of an interrupted pip install; the deploy reports one it cannot remove, and it can be deleted.

### Root-owned files in the clone
Old root-git deploys could leave root-owned files in the prod clone, and git as `shiny` would then fail half-way, so the deploy refuses first (nothing was root-owned on 2026-11-18 or 2026-11-21). **Do not repair it with a root `chown` inside the clone**: `shiny` controls every directory there, and can swap one for a symlink between a `find` and the `chown`, which would then re-own a file outside the clone (`--no-dereference` protects only the last path component). Set the clone aside instead and let the deploy make a fresh one. The rename happens in root's own `/srv/shiny-server` and reads nothing inside the clone:

```bash
sudo systemctl stop osmose-shiny.service
sudo mv /srv/shiny-server/osmose-src /srv/shiny-server/osmose-src.old
sudo bash deploy.sh                    # clones afresh as shiny (public repo, no credentials), provisions the JARs, restarts
```

**If that deploy fails before the new clone is complete**, the service is stopped and `/srv/shiny-server/osmose` points at nothing. Put the old clone back and start the service: `sudo rm -rf /srv/shiny-server/osmose-src` (only if the failed clone left a directory), `sudo mv /srv/shiny-server/osmose-src.old /srv/shiny-server/osmose-src`, `sudo systemctl start osmose-shiny.service`.

Before deleting `osmose-src.old`, list what it held that git does not track, **as `shiny`**: `sudo runuser -u shiny -- git -C /srv/shiny-server/osmose-src.old status --short --ignored --untracked-files=all`. On 2026-11-21 that was the UI run history (`data/history/*.json`), the two JARs (provisioned again by the deploy) and `__pycache__`; there was no `data/feedback/` (the feedback store is configured elsewhere). Copy anything worth keeping **as `shiny`**, never as root, e.g. `sudo runuser -u shiny -- cp -a /srv/shiny-server/osmose-src.old/data/history /srv/shiny-server/osmose-src/data/`.

### If a deploy stops
- **Before the checkout** (Step 0, or the clone checks): the previous ref is still checked out and the service is as it was. If you stopped it by hand (above), start it again: `sudo systemctl start osmose-shiny.service`, then `systemctl status osmose-shiny.service`.
- **After the checkout, before the restart** (for example the unit write in Step 3): the new ref is on disk under the old process. `systemctl start` does nothing to a running service. Fix the cause and run the deploy again, or load what is on disk with `sudo systemctl restart osmose-shiny.service`, then verify as usual.

## The C# engine (optional, opt-in)

The third engine (`csharp/`, spec `docs/superpowers/specs/2026-11-17-csharp-engine-design.md` section 9.7) is **off by default**: a plain `sudo bash deploy.sh` installs nothing for it. Where a build is installed, the web app offers it as a third engine button; production has had one since 2026-11-21.

**Every deploy removes the build a previous deploy made**, as `shiny` and before the checkout, so an old build never outlives its ref. So once the engine is on a server, **every later deploy must carry the flag too**: one without it leaves the app without the C# engine. Such a deploy warns where it deletes the build, its summary reads `C# engine: REMOVED`, and its last lines give the command that restores the engine. (A server that never had the engine prints `not built`.) The release procedure's deploy command carries the flag (`.claude/skills/release/SKILL.md`).

The ref is resolved before anything is deleted for it. A deploy for a ref that does not exist, typically a tag that has not been pushed yet, stops with "Was the tag pushed?" and leaves the checkout and the build as they were. Before 2026-11-21 it deleted the build first and then died at the checkout.

To build it on the server:

```bash
sudo OSMOSE_DEPLOY_REF=<tag> OSMOSE_DEPLOY_CSHARP=1 bash deploy.sh
```

After the service is up and answering HTTP, the deploy:
1. installs the .NET 10 SDK into `/opt/dotnet` with Microsoft's `dotnet-install.sh`, if no 10.x SDK is there;
2. installs `pythonnet` into the shiny env: pip installs or updates whatever `pythonnet>=3.2.1` needs, in an env other apps share (on production, 2026-11-21, that added `pythonnet` and `clr_loader` and replaced nothing);
3. builds the assembly from the deployed ref **as `shiny`**, never as root, into a staging directory;
4. runs the self-test on the staged build **as the service runs**: user `shiny`, the env's interpreter, the working copy and `DOTNET_ROOT=/opt/dotnet`;
5. only if it passes, moves the build into `osmose/engine_cs/bin` and checks that the service's own path can load it;
6. **restarts the service once more** and requires the served page to offer the engine. The app decides once, at startup, whether to offer it, and the process the deploy started earlier came up before the build existed. On the first production deploy (2026-11-21, before this step existed) the summary said "activated" while the app showed no C# button until a restart by hand.

So a C# deploy interrupts the service **twice**, a few seconds each. If the app does not offer the engine after that restart, the deploy removes the build, restarts the service without it and exits 1; if the removal itself fails, it says the rollback is incomplete and names the directory to remove by hand.

The unit always carries `Environment=DOTNET_ROOT=/opt/dotnet`, which is inert when the directory is absent. A failure in any step exits 1 with a `C# engine:` error, and a build that fails never becomes loadable. **The app keeps running the new code without the engine:** the steps run after the restart, so a C# failure can never leave new code on disk under the old process.

CI and the Docker and Apptainer images test the Python bridge on Python 3.12. Production runs it on Python 3.13; that combination was checked by hand on 2026-11-21 (the self-test, and C#, Python and C# runs in one app process).

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
