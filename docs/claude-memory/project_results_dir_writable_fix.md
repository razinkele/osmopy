---
name: project-results-dir-writable-fix
description: "Prod RESULTS_DIR resolved into read-only home via symlink → startup probe + checkpoint failures; fixed via OSMOSE_RESULTS_DIR env + StateDirectory, shipped+rolled out 2026-06-16 (PR #63, 8b48545)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3c92cf51-9a04-490e-b26d-2e39302eb27f
---

**RESULTS_DIR-not-writable fix — SHIPPED + ROLLED OUT 2026-06-16** (PR #63 rebase-merged, master `8b48545`, branch deleted, local synced; all 6 CI legs green).

Prod `RESULTS_DIR` = `default_results_dir()` resolved to `/home/razinka/osmose/osmose-python/data/baltic/calibration_results` (via the `/srv/shiny-server/osmose`→home symlink that `Path(__file__).resolve()` follows), read-only to service user `shiny` → "RESULTS_DIR probe failed: [Errno 13] Permission denied" at startup + calibration checkpoint-write failures.

Fix: `default_results_dir()` now honors `OSMOSE_RESULTS_DIR` env (mirrors `OSMOSE_FEEDBACK_FILE`; empty ignored; `RESULTS_DIR` const reads it at import so systemd `Environment=` works); `deploy.sh` unit template sets `StateDirectory=osmose/calibration_results` + `Environment=OSMOSE_RESULTS_DIR=/var/lib/osmose/calibration_results`.

**Rollout DONE** via systemd drop-in `/etc/systemd/system/osmose-shiny.service.d/results-dir.conf` (those 2 lines) + daemon-reload + restart; verified: env resolved, `/var/lib/osmose/calibration_results` owned `shiny:shiny`, startup clean (probe error gone).

Gotcha: `probe_writable`/`write_checkpoint` need a PRE-EXISTING writable dir (neither mkdirs) → StateDirectory (not just env) is required.
