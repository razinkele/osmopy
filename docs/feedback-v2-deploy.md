# Feedback v2 deploy sequence

> **STATUS: EXECUTED 2026-09-15 ~22:45. This deploy is DONE.**
> Prod moved `c21349d1` (2026-07-19) -> `54ccac8`, 481 commits. All four steps verified: v1 gone
> (`feedback_review.py` present, HTTP 200, `NRestarts=0`); token set in a root-owned 0600
> EnvironmentFile and absent from `systemctl show -p Environment`; `OSMOSE_TRUSTED_PROXY=1`; and a
> real UI submission whose record carried `has_contact: true` with NO `contact` key, address in a
> separate `contacts.jsonl`. The store was then hardened to 0700/0600 (see below) and the smoke-test
> record removed.
>
> **Do not re-run this sequence.** It is kept as the record of what was done and why, and for the
> traps in it that generalise — steps 1-2 still describe the live configuration. Retire it once
> `DEPLOY.md` fully describes the steady state.
>
> **One thing changed AFTER this ran:** the store's 0600 file mode is now enforced in code
> (`_append_json_line`, PR #154), not only by this host's `StateDirectoryMode=0700`. systemd's
> StateDirectory defaults to 0755 and `osmose/feedback.py` set no mode at all, so `contacts.jsonl` —
> which holds reporter email addresses — landed world-readable on first write. Found by checking
> the modes after everything else already looked green.


A one-time, ordered runbook for moving production from feedback **v1** to **v2**. Written
2026-09-15 against master @ `55cadb6`.

**How this relates to the other two documents.** [`DEPLOY.md`](../DEPLOY.md) describes the
deployment's steady state — what the service is, which settings exist and why.
[`docs/feedback-runbook.md`](feedback-runbook.md) is the operator guide for *using* the feature once
it runs (triage, promoting to a GitHub issue, handling contacts). This file is neither: it is the
ordered sequence for the specific transition, with the verification that proves each step landed.
Delete it once the deploy is done and `DEPLOY.md` describes reality.

**Where this CORRECTS `DEPLOY.md`.** That file says to add `Environment=OSMOSE_FEEDBACK_TOKEN=...`
to the unit. Do not: (1) `deploy.sh:223` runs `cat > "$SERVICE_FILE"`, rewriting the unit wholesale
on every deploy, so the line is destroyed the next time anyone deploys; (2) `systemctl show -p
Environment` prints `Environment=` values to **any local user** — demonstrated 2026-09-15 from an
unprivileged account. Use the EnvironmentFile + drop-in in steps 1-2 instead. `DEPLOY.md` has been
updated to point here.

**State at the time of writing** (measured, not assumed): prod is at `c21349d1` (2026-07-19), **481
commits behind**, and is running feedback **v1** — `osmose/feedback.py` there writes the reporter's
address inline as a `"contact"` key, there is no `feedback_review.py`, and `data/feedback/` does not
exist (so nothing has been collected yet, but the v1 submit path IS live and reachable).

Run everything as root on the app host. `deploy.sh` needs sudo. Do step 0 first and keep its output.

## 0. Pre-flight — record the rollback point

    git -C /srv/shiny-server/osmose-src rev-parse HEAD
    systemctl show osmose-shiny.service -p Environment -p EnvironmentFiles -p StateDirectory
    curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8838/

Expect c21349d1..., Environment without any feedback vars, and 200.
**Write the SHA down** — it is the rollback ref in step 6.

## ORDERING — do steps 1-3 in one sitting, and do not invite submissions between them

Steps 1-2 take effect on the **still-running v1**, because the service restarts before the code is
replaced. Two consequences, neither fatal but both worth closing quickly:

- **v1 honours `OSMOSE_FEEDBACK_FILE`** (`_FILE_ENV`, `osmose/feedback.py:23` on the deployed tree),
  so from step 2's restart until step 3 completes, v1 writes to the new `/var/lib/osmose/feedback/`
  path — in v1's shape, i.e. with the reporter's address **inline**.
- **v1's `read_feedback` does no scrubbing**: it returns each record verbatim, `contact` key and all
  (the normalisation that drops it exists only in v2). So the moment the token from step 1 is live,
  v1's `GET /api/feedback` will serve unscrubbed addresses to anyone holding it.

The store is empty at the time of writing (`data/feedback/` does not exist), so there is nothing to
expose *today* — the risk is only for a submission that arrives inside the window. Keep the window to
minutes and it is a non-issue.

**If you must stop between step 2 and step 3**, remove the `OSMOSE_FEEDBACK_TOKEN` line from
`/etc/osmose-shiny.env` and `systemctl restart osmose-shiny` first: with no token,
`check_feedback_token` returns False for every caller and the endpoint is closed again. Put the line
back before step 3.

Setting it to an EMPTY STRING also disables it — `check_feedback_token` (`osmose/feedback.py`) reads
`tok = os.environ.get(...)` then `if not tok ... return False`, so unset and `""` behave identically
and the endpoint is closed either way. **This is the opposite of `OSMOSE_TRUSTED_PROXY` in the same
file**, which is a bare `os.environ.get(...)` truthiness test on the string, so `0` and `false`
ENABLE it and only removing the line disables it. The two variables genuinely do not share a
convention — check the guard, do not pattern-match from the neighbouring line.

## 1. Secrets file (root-owned, 0600)

NOT `Environment=` in the unit: `systemctl show` prints those to ANY local user (demonstrated this
session as an unprivileged account). An EnvironmentFile's contents never appear there.

    install -m 0600 -o root -g root /dev/null /etc/osmose-shiny.env
    printf 'OSMOSE_FEEDBACK_TOKEN=%s\n' "$(openssl rand -hex 32)" >> /etc/osmose-shiny.env
    printf 'OSMOSE_TRUSTED_PROXY=1\n' >> /etc/osmose-shiny.env
    printf 'OSMOSE_FEEDBACK_FILE=/var/lib/osmose/feedback/feedback.jsonl\n' >> /etc/osmose-shiny.env
    printf 'OSMOSE_CONTACTS_FILE=/var/lib/osmose/feedback/contacts.jsonl\n' >> /etc/osmose-shiny.env
    cat /etc/osmose-shiny.env    # note the token; you need it in step 5

TRAP: `OSMOSE_TRUSTED_PROXY=0` ENABLES it — the code does a plain truthiness test on the STRING.
To disable it you must DELETE the line, not set it to 0/false/no.

Why relocate the store: it defaults to `<repo>/data/feedback/` INSIDE the prod clone. The unit's own
comment already establishes the rule ("keeping state out of the source tree survives a redeploy").

## 2. systemd drop-in

`deploy.sh:223` does `cat > "$SERVICE_FILE"` — it rewrites the unit wholesale on EVERY deploy, so a
hand-added `Environment=` line is destroyed each time. It never touches the `.d/` directory, and
`daemon-reload` (which deploy.sh runs) picks drop-ins up. So the drop-in survives; a unit edit does not.

    mkdir -p /etc/systemd/system/osmose-shiny.service.d
    cat > /etc/systemd/system/osmose-shiny.service.d/override.conf <<'EOF'
    [Service]
    EnvironmentFile=/etc/osmose-shiny.env
    StateDirectory=osmose/calibration_results osmose/feedback
    EOF
    systemctl daemon-reload
    systemd-analyze cat-config systemd/system/osmose-shiny.service | tail -20

The StateDirectory line repeats calibration_results deliberately: systemd is expected to APPEND
list-valued settings across drop-ins, but listing both is correct whether it appends or overrides.
It matters because /var/lib/osmose is root:root drwxr-xr-x — `shiny` CANNOT create a subdir there,
so `append_feedback`'s `mkdir(parents=True)` would fail. systemd creates StateDirectory paths owned
by the service user on every start; that is what makes the feedback path writable.

VERIFY before deploying:
    systemctl restart osmose-shiny && ls -ld /var/lib/osmose/feedback
Expect drwx------ (or similar) owned by shiny. If it is absent, STOP — the append will fail silently
at the first submission and you will have deployed a feature that cannot store anything.

## 3. Deploy

    sudo bash /home/razinka/osmopy/deploy.sh

What it does: fetch + `checkout --force --detach origin/master` in the prod clone (a separate clone,
clean working tree — the 481-commit distance is irrelevant to git), copies the Java JAR from the dev
tree, fixes the symlink, pip work, rewrites the unit, daemon-reload, restart, polls HTTP for ~40s.

BLAST RADIUS, know this before running: the pip upgrades go into the SHARED env
`/opt/micromamba/envs/shiny`, which also serves bowtie_app, EcoNeTool and marinesabres (per
/etc/nginx/sites-enabled/nid4ocean). `cma`, `shinyswatch`, `shinywidgets`, `pyarrow` and
`shiny_deckgl` are upgraded UNCONDITIONALLY. A deploy of this app can therefore move other apps'
dependencies. Consider checking those three still load afterwards.

## 4. Post-deploy verification — the four handoff items

(a) pulled:
    git -C /srv/shiny-server/osmose-src rev-parse --short HEAD     # expect 55cadb6
    ls /srv/shiny-server/osmose-src/osmose/feedback_review.py      # v2 marker; absent on v1

(b) token present but NOT leaking:
    systemctl show osmose-shiny.service -p Environment             # must NOT contain the token
    systemctl show osmose-shiny.service -p EnvironmentFiles        # must list /etc/osmose-shiny.env
    curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8838/feedback/review          # 403
    curl -s -o /dev/null -w '%{http_code}\n' -H "x-feedback-token: <TOKEN>" \
         http://127.0.0.1:8838/feedback/review                                              # 200

(c) proxy trust — submit from two different client IPs and confirm they get separate rate buckets
    (both should be accepted; with XFF ignored they would share one 5/hour bucket).

(d) the one act that settles writability — MAKE A REAL SUBMISSION through the UI at /osmose/, with
    an email address, then:
    ls -l /var/lib/osmose/feedback/
    python3 -c "import json;[print(json.loads(l).keys()) for l in open('/var/lib/osmose/feedback/feedback.jsonl')]"

    The record MUST show `has_contact` and MUST NOT contain a `contact` key. If it shows `contact`,
    v1 is still being imported — stop and recheck (a).

## 5. Known, pre-existing, NOT caused by this deploy

- **`pyvis` — RESOLVED 2026-09-15, no action needed.** It was missing from the shared env because
  `deploy.sh` did not install it; the import at `osmose/trophic_network.py:204` is lazy and
  UNGUARDED, so the app started fine and the trophic-network feature raised `ModuleNotFoundError`
  only when used. `deploy.sh` now installs it (PR #152, `bee2996`), and the deploy carried it into
  the running env — verified there: `pyvis 4.2`.
  **The trap, if this ever recurs:** `pyvis` is a FORK (`razinkele/pyvis`, "Optimized Edition",
  versioned 4.x); PyPI's `pyvis` is a different lineage that stops at **0.3.2**, and `pip show pyvis`
  succeeds for BOTH. So it can never be a presence check like `pymoo`/`SALib` — a presence check
  passes while leaving the WRONG package installed. The floor check (`pyvis: 4.2`) is what
  distinguishes them, because `0.3.2 < 4.2` fails it. If you ever see `pyvis 0.3.2`, pip resolved
  PyPI instead of the fork; reinstall with `--force-reinstall` from the git URL.
- The env has `shiny==1.7.0` against a declared `shiny>=1.6.3,<1.7`. deploy.sh's floor check only
  tests the LOWER bound, so it passes a check meant to catch exactly this. Pre-dates the deployed
  commit (pin added 2026-06-16, prod deployed 2026-07-19) and prod currently serves 200, so it is
  tolerated in practice — but it is an unvalidated combination.

## 6. Rollback

    OSMOSE_DEPLOY_REF=<SHA from step 0> sudo bash /home/razinka/osmopy/deploy.sh

deploy.sh honours OSMOSE_DEPLOY_REF, so rollback uses the same path as deploy. The drop-in and the
secrets file are independent of the code version and can stay. To also revert those:
    rm /etc/systemd/system/osmose-shiny.service.d/override.conf /etc/osmose-shiny.env
    systemctl daemon-reload && systemctl restart osmose-shiny

## What this sequence does NOT de-risk

481 commits of engine/schema/UI change land at once on a service that has served July's code since
2026-07-19. Nothing static can clear that; step 4's HTTP 200 and a real submission are the evidence,
and step 6 is the way back. Consider deploying during low usage.
