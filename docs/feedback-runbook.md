# Feedback system — operator runbook

In-app "Send feedback" (bug report / suggestion) plus a token-gated maintainer review page.
This document is for whoever **runs** the deployment. Every claim below was checked against the
code at the cited `file:line`; where something could not be checked from the source tree it says
so explicitly.

- Submission: `ui/components/feedback_modal.py` (a Shiny reactive effect — there is no HTTP POST)
- Store + token check: `osmose/feedback.py`
- Rate limiter: `osmose/feedback_limits.py`
- Review page HTML: `osmose/feedback_review.py`
- Routes: `app.py:709-741`

## 1. The two stores — one is PII

| File | Default path | Contents | Safe to share? |
|---|---|---|---|
| Feedback records | `data/feedback/feedback.jsonl` | type, message, timestamp, app version, nav tab, id, `has_contact` **boolean** | See caveats below |
| Contacts | `data/feedback/contacts.jsonl` | `{"id": ..., "email": ...}` — **reporter email addresses** | **No. Never.** |

The split is structural, not conventional: `build_feedback_record` (`osmose/feedback.py:56-88`)
takes `contact` only to set the boolean `has_contact` flag and **never puts the address in the
record**. Storing the address is a separate call to `save_contact`
(`osmose/feedback.py:108-121`), which writes the side-store.

Both default paths are gitignored by the single rule `data/feedback/` at `.gitignore:14`
(verified with `git check-ignore`). **That protection is tied to the default location** — if you
point `OSMOSE_FEEDBACK_FILE` or `OSMOSE_CONTACTS_FILE` somewhere else inside the repo, nothing
ignores it.

Two caveats on "the feedback file is safe to paste":

1. **It is `read_feedback()`'s *output* that is scrubbed, not the file.** `read_feedback`
   (`osmose/feedback.py:165`) normalises a legacy v1 line that still carries a literal `contact`
   key: the address is dropped and folded into the `has_contact` boolean. Measured — feeding it a
   v1 line containing `secret@example.org` returns `{"has_contact": true, ...}` with no `contact`
   key and no address, while **the address is still sitting in the raw file on disk**. So export
   through `read_feedback()`; never `cat` the raw store into an issue unless you know every line
   was written by current code.
2. A reporter can type anything into the message box, including their own email, a colleague's
   name, or a file path. The structured contact channel is protected; **free text is not**.

A hybrid line carrying both `contact` and an explicit `has_contact: false` drops the address (the
`pop` is unconditional) but keeps the flag `false` — `setdefault` will not overwrite it. The
address never leaks; the flag is just unreliable on such a line. Use `lookup_contact` (§8) rather
than trusting the flag if it matters.

## 2. Environment variables

All four are read at call time, so a `systemd` `Environment=` line takes effect on restart.

| Variable | Read at | Effect when unset |
|---|---|---|
| `OSMOSE_FEEDBACK_TOKEN` | `osmose/feedback.py:178` (`check_feedback_token`) | **Both read endpoints are disabled** — every request gets `403`. |
| `OSMOSE_FEEDBACK_FILE` | `osmose/feedback.py:39` (`_resolve`) | Defaults to `<repo>/data/feedback/feedback.jsonl`. |
| `OSMOSE_CONTACTS_FILE` | `osmose/feedback.py:47` (`_resolve_contacts`) | Defaults to `<repo>/data/feedback/contacts.jsonl`. |
| `OSMOSE_TRUSTED_PROXY` | `ui/components/feedback_modal.py:173` (`_client_key`) | `X-Forwarded-For` is ignored — see §5. |

`OSMOSE_TRUSTED_PROXY` is a **truthiness check on the value**, so `OSMOSE_TRUSTED_PROXY=0` counts
as *set*. Use `1` and delete the line to disable, rather than setting it to a falsey-looking
string and expecting that to turn it off.

### Set the store paths on the production deployment

`DEPLOY.md:11` records that the production service runs as user `shiny` and that **the source tree
is read-only to that user** — which is why calibration checkpoints already use
`OSMOSE_RESULTS_DIR=/var/lib/osmose/calibration_results`.

The feedback store has exactly the same problem and no equivalent override configured. The
defaults point inside the read-only source tree, and `append_feedback`
(`osmose/feedback.py:96-105`) does `mkdir(parents=True)` then `open(p, "a")`. On a read-only tree
that raises `PermissionError`, which is caught at `ui/components/feedback_modal.py:312`, logged as
`feedback save failed`, and shown to the user as `Couldn't save feedback — try again.` — on
**every** submission.

So set both, alongside the existing `OSMOSE_RESULTS_DIR`:

```
Environment=OSMOSE_FEEDBACK_FILE=/var/lib/osmose/feedback/feedback.jsonl
Environment=OSMOSE_CONTACTS_FILE=/var/lib/osmose/feedback/contacts.jsonl
```

and make sure the directory exists and is owned by `shiny` (a `StateDirectory=` entry, as the unit
already does for calibration results, is the tidiest way).

> This one is inferred from `DEPLOY.md:11` plus the code path, not observed on the live service —
> the deployment is not reachable from the source tree. **Verify it by submitting one piece of
> feedback after a restart and confirming a record lands**, rather than assuming either way.

## 3. Reaching the review page

Two routes, both registered ahead of Shiny's catch-all mount (`app.py:739,741`):

| Path | Response | Handler |
|---|---|---|
| `/feedback/review` | HTML page, `Cache-Control: no-store` | `app.py:719-733` |
| `/api/feedback` | JSON array of records | `app.py:709-715` |

Both require the header **`x-feedback-token`** (`app.py:711,721`) matching
`OSMOSE_FEEDBACK_TOKEN`. Starlette's header lookup is case-insensitive, so `X-Feedback-Token`
works identically.

```bash
curl -sS -H "X-Feedback-Token: $OSMOSE_FEEDBACK_TOKEN" \
     http://127.0.0.1:8838/feedback/review
```

**On the public URL, mind the prefix.** The routes are registered at `/feedback/review` and
`/api/feedback`. `DEPLOY.md:6,9` shows the service running with `--root-path /osmose` behind nginx
at `/osmose/`, so the externally reachable URL is very likely
`https://<host>/osmose/feedback/review`. The nginx configuration is not in this repo and the exact
path-stripping behaviour was not measured — **confirm with `curl` before relying on either form.**

A `403` means "token missing, unset, or wrong" and **does not distinguish those cases**
(`check_feedback_token`, `osmose/feedback.py:171-181`, returns `False` for all of them). If you
get a `403` you did not expect, check that the variable is actually set in the service
environment, not just in your shell.

The page renders **every** record with no pagination (`render_review_html`,
`osmose/feedback_review.py:215-238`) and `read_feedback` loads the whole file into memory. That is
the practical reason to rotate (§7) well before the 50 MiB cap.

## 4. What a user sees when a submission is refused

Checks run in this order (`_classify_and_consume`, `ui/components/feedback_modal.py:235-282`); the
order is security-relevant and documented as such in that function.

| Message the user sees | Cause |
|---|---|
| `Enter a message before sending.` | Empty message. No rate-limit slot consumed. |
| `That email address doesn't look right — correct it or leave it blank.` | Email fails the shape check. No slot consumed. |
| `You've sent several already — please wait a little before sending more.` | Rate limited, and the limiter can prove it is this client's own cap. |
| `We couldn't accept this right now — please try again shortly.` | Rate limited, cause **ambiguous** — see below. |
| `Couldn't save feedback — try again.` | The store write failed in a way that **might** be transient. Retrying is reasonable. |
| `The server can't store feedback right now — this has been logged.` | The store write failed **terminally** — full store, or an unwritable path. Retrying can never work; an operator must act. See §6 and §7. |
| `Thanks — feedback saved.` | Success — **and also the honeypot path.** |

The two save-failure messages are chosen by `_is_terminal_save_failure`
(`ui/components/feedback_modal.py`): a `RuntimeError` (the store-full guard) or an `OSError` with
`EACCES`/`EPERM`/`EROFS`/`ENOSPC`/`EDQUOT` is terminal; anything else keeps the retry invitation,
because "try again" is the safer answer when the cause is genuinely unknown. **Neither message
says which condition occurred** — both terminal causes share one string. That is the same rule that
governs the HTTP routes (§6): detail goes to the log, never to the caller. A user quoting the
terminal message is telling you the server needs attention, not that they did anything wrong.

### The vague one is deliberate

`at_capacity` (`osmose/feedback_limits.py:44-65`) narrows the cause in **one direction only**:

- `at_capacity` **False** proves the refusal was this client's own per-client cap. The UI may
  therefore say "you've sent several already", and does.
- `at_capacity` **True** is **ambiguous**. An established client that is over its own cap *while
  the table happens to be full* reads exactly the same. The limiter cannot tell that apart from
  "the table is full and you are a new key", so the UI asserts **no cause at all**.

**If a support ticket quotes `We couldn't accept this right now`, that message is not evidence the
server was full.** It means the submission was rate-limited and the code declined to guess why.
Check the log (§5) for a saturation warning to distinguish them. Telling a first-time submitter who
has sent nothing that they "have already sent several" is the failure this wording avoids.

### Successful-looking non-submissions

The honeypot hit (`DROP`) shows the **same** success notification, clears the same fields and
dismisses the modal — a bot must not be able to detect it. Nothing is stored. The only trace is
an INFO line, `feedback honeypot filled` (`ui/components/feedback_modal.py:410`). If a user insists
they submitted something that is not in the store, check for that line — a browser extension or
autofill filling the offscreen "Website" field would do it.

## 5. Rate limiting and its degraded modes

The production limiter is **5 submissions per client per hour**
(`ui/components/feedback_modal.py:40`), in-memory, with a **10 000-key** table
(`MAX_KEYS`, `osmose/feedback_limits.py:19`).

It is per-process. `osmose/feedback_limits.py:1-13` states the target is a single-worker deploy; if
you ever run multiple workers the cap becomes per-worker and must be divided accordingly.

### How the client is identified

`_client_key` (`ui/components/feedback_modal.py:144-211`), in order:

1. **`X-Forwarded-For`, but only when `OSMOSE_TRUSTED_PROXY` is set** — and then the **rightmost**
   comma-separated entry. With one trusted proxy in front, that is the address the proxy itself
   observed. The **leftmost entry is attacker-supplied** and taking it is the classic XFF spoof:
   it would hand every request a fresh bucket and switch rate limiting off entirely.
2. Otherwise `session.http_conn.client.host`.
3. Otherwise one shared constant key.

The rightmost-entry rule assumes **exactly one trusted hop**. If you put a CDN or a second proxy
in front of nginx, the rightmost entry becomes that extra hop's view and the effective key changes
— re-check this if the topology changes.

### The four ways this degrades to one global bucket

Three warn. **The fourth is silent** — this matters most on this deployment, which sits behind
nginx.

| # | Condition | Key used | Logged? |
|---|---|---|---|
| 1 | `X-Forwarded-For` present, `OSMOSE_TRUSTED_PROXY` **unset** | proxy's address | warns once |
| 2 | `OSMOSE_TRUSTED_PROXY` set, but XFF has no usable rightmost entry (trailing comma) | proxy's address | warns once |
| 3 | No client address available at all | `_no-client-address_` | warns once |
| 4 | **Proxy does not send `X-Forwarded-For` at all** | proxy's address | warns once (**new, 2026-09-14**) |

Case 4 was **completely silent until 2026-09-14**, and it is the one that matches a default nginx
deployment. Measured before the fix: no XFF header, `client.host = 127.0.0.1`, `_client_key`
returned `'127.0.0.1'` for every session and logged nothing at all — the whole `if xff:` block is
skipped, so none of the other three warnings can fire.

It now warns once, on a heuristic: no XFF **and** `client.host` is loopback or RFC1918. The key is
deliberately unchanged — a shared bucket is the correct behaviour when no client address is
available; the silence was the defect. Because it is a heuristic it can be a false positive on a
genuine direct connection from localhost, which the warning says in its own text.

> The heuristic is loopback + RFC1918 (`10/8`, `172.16/12`, `192.168/16`) only — **not**
> `ipaddress.is_private`, which is also True for the RFC5737 documentation ranges
> (`198.51.100.0/24`, `203.0.113.0/24`) and would warn about healthy direct deployments.

**You need the header *and* the variable.** Setting `OSMOSE_TRUSTED_PROXY` without nginx sending
`X-Forwarded-For` leaves you in case 4; sending the header without the variable is case 1. If any
of these warnings appears, per-client rate limiting is not in effect.

The three `_warned_*` flags (`ui/components/feedback_modal.py:55-57`) are **process-lifetime and
never re-armed**. Each fires at most once per process. After a restart, look at the log around the
*first* submission; later ones say nothing regardless of how bad the configuration is.

### Table saturation — do NOT alert on the warning's rate

When the key table is full, new keys are refused (established clients continue) and this is logged
once:

```
Rate limiter table at capacity (10000 keys); refusing new keys
```

Eviction-by-oldest is deliberately not implemented (`_sweep_stale`,
`osmose/feedback_limits.py:90-106`): it would let an attacker who has exhausted their own limit
flood the table to evict their own record and reset it.

The warning is one-shot, and it is **re-armed only inside a sweep** — and sweeps happen at most
once per `_sweep_interval`, which for the production limiter is `max(1.0, 3600/10)` = **360 s**.

Measured on a limiter configured exactly as production's:

- 10 000 keys loaded, then 51 refusals in one episode → **1** warning.
- A second burst of refusals **359 s later** → still **1** warning. No sweep ran, so the flag was
  never re-armed.
- After a sweep that drained the table, the flag re-armed.

The operational consequence, stated plainly: **one warning does not mean one saturation episode,
and sustained saturation produces exactly one warning at onset no matter how long it lasts.** Alert
on the *presence* of this line, never on its rate or count — a rate-based alert will under-report
by an unbounded factor. To judge severity, look at whether it recurs across days and at user
reports of the ambiguous refusal message (§4), not at how many times it was logged.

## 6. Other failure modes worth knowing

**A bad `repo_url` costs the whole review page, not one card.** `github_issue_url`
(`osmose/feedback_review.py:150-155`) validates the scheme — only `http`/`https` — and raises
`ValueError` otherwise, because `html.escape` cannot neutralise a `javascript:` or `data:` scheme
in an `href`. That raise propagates through `render_review_html` and is caught by the route, so
**every card is lost and the caller gets a bare `500 internal`**. `_REPO_URL` is an
operator-controlled constant at `app.py:49` (`https://github.com/razinkele/osmopy`), so this is a
deployment bug, not something a reporter can trigger. The server-side traceback
(`app.py:733`) is the only way to see it.

**The review route returns a bare `internal` on purpose — do not "improve" it.** Both routes
answer unauthenticated callers, so neither may disclose why it failed. `app.py:702-706` records
that a mutation which let exception text through was measured to put **the token itself** in the
response. The detail goes to the log via `_log.exception`; the response body stays bare. A
well-meaning future change to return the error "to help debugging" reintroduces a token leak. If
you need detail, read the log.

**A lost contact address does not fail the submission.** `_store_submission`
(`ui/components/feedback_modal.py:316-325`) guards `save_contact` separately: once
`append_feedback` returns, the feedback *is* stored, and reporting failure would make the user
resubmit — producing a duplicate record plus an orphan `has_contact: true` with no contacts row.
Instead it logs at ERROR. Grep for `contact address was LOST`; the line names the feedback id.

**Both stores share one write path.** `append_feedback` and `save_contact` both go through
`_append_json_line` (`osmose/feedback.py`), so both carry the same `MAX_STORE_BYTES` cap and the
same exclusive `flock`. Until 2026-09-14 those protections lived only in `append_feedback`, which
meant the public file had them and `contacts.jsonl` — the one holding email addresses — had
neither; the asymmetry ran backwards. A full contacts store raises
`contacts store is full (<n> bytes) — rotate <path>`, which `_store_submission` catches and logs
as `contact address was LOST` **without failing the submission**, since the feedback record itself
is already stored by then.

**If every post-write step fails**, the user sees no confirmation at all and will probably
resubmit. That case escalates to ERROR: `record written but EVERY post-write step failed`
(`ui/components/feedback_modal.py:382`).

## 7. Rotating the store at `MAX_STORE_BYTES`

`MAX_STORE_BYTES` is **50 MiB** (`osmose/feedback.py:31`). `append_feedback`
(`osmose/feedback.py:94-95`) checks the size **before** writing and raises `RuntimeError` once the
file is at or over it:

```
feedback store is full (<n> bytes) — rotate <path>
```

That exception is caught by `_store_submission`, which classifies it as **terminal**, so the user
is shown `The server can't store feedback right now — this has been logged.` and is **not** invited
to retry — retrying would fail forever until an operator rotates. (Before 2026-09-14 this case
showed `Couldn't save feedback — try again.`, advice that could not work.) The
`feedback store is full` text itself appears only in the server-side traceback under the
`feedback save failed` ERROR, so the log is still where you find out which failure it was.

Rotate with a plain rename. `append_feedback` opens in append mode and creates parent directories,
so it recreates the file on the next submission; no restart is needed and there is no window where
a write is lost to a missing file:

```bash
cd /var/lib/osmose/feedback        # or wherever OSMOSE_FEEDBACK_FILE points
mv feedback.jsonl "feedback.jsonl.$(date +%Y%m%d)"
```

Then archive `feedback.jsonl.<date>` somewhere private — remember it is a store that may contain
self-disclosed PII in message bodies, and, if it is old enough to contain v1 lines, literal
addresses (§1).

**Do not rotate `contacts.jsonl` at the same time.** It is uncapped, small, and `lookup_contact`
on any *older* feedback id needs it. Rotating it orphans every historical id.

After rotation the review page shows only the records in the live file. Rotate at a size that keeps
the page usable rather than waiting for the hard cap.

## 8. Looking up a reporter's email

`lookup_contact` (`osmose/feedback.py:124-136`) maps a feedback id to the stored address.

**It has no production call site** — verified: the only references in the tree are its own
definition, a docstring mention in `osmose/feedback_review.py:10`, and tests. Nothing in `app.py`,
the modal, or the review page calls it. The review page renders the `has_contact` **flag** and
never resolves it, which is what makes that page safe to screenshot.

So this is an **operator tool, used deliberately**: reading an address is an explicit act, not
something that happens as a side effect of viewing feedback. Keep it that way — if you add an
export or a UI that resolves addresses, you remove the property the whole split exists to provide.

Take the id from the review card (`id <hex>`), then:

```bash
cd /path/to/osmopy        # the repo root — NOT the store directory
OSMOSE_CONTACTS_FILE=/var/lib/osmose/feedback/contacts.jsonl \
  .venv/bin/python -c \
  "from osmose.feedback import lookup_contact; print(lookup_contact('<feedback-id>'))"
```

**Run it from the repo root, with the interpreter that deployment actually uses.** Both halves
matter and both fail quietly:

- *Working directory.* `python -c` puts the **current directory** on `sys.path`, not the repo. Run
  from `/var/lib/osmose/feedback` (where §7 just sent you) and `import osmose` falls through to
  whatever is pip-installed. If that is an editable install pointing at a different checkout, you
  will silently query **a different version of this code** — this exact mistake was made while
  writing this runbook, and it produced a confidently wrong answer. `cd` to the repo root first, or
  set `PYTHONPATH` to it.
- *Interpreter.* `DEPLOY.md:7-8` records that the production service runs from
  `/opt/micromamba/envs/shiny` and imports `osmose` from the working copy — it is **not**
  pip-installed there and there is **no `.venv`**. On that host substitute
  `/opt/micromamba/envs/shiny/bin/python`. The `.venv/bin/python` form above is the development one.

Prints the address, or `None` if the reporter left none, the file does not exist, or the line is
corrupt. Run it as a user that can read the contacts file, and do not paste the output anywhere the
feedback record itself is going.

## 9. The GitHub promotion link — the privacy boundary

Each card carries a "promote to a GitHub issue" link: a prefilled `issues/new` URL built by
`github_issue_url` (`osmose/feedback_review.py:110-182`). It is a prefilled link rather than a REST
API call by design — no token, no bot account, nothing to rotate or leak — **and the maintainer
reviews the issue before filing it.**

**What is protected.** The body is assembled from an explicit allowlist: `message`, `version`,
`nav_tab`, `id`, and the `has_contact` *flag*. The record is never serialised wholesale, so even a
legacy v1 line still carrying a literal `contact` key cannot carry an address into a public issue.
An address a reporter typed into the email field **never reaches the issue.**

**What is not.** The body carries reporter-controlled free text in **two** fields, not one:

- `message` — the obvious one.
- **`nav_tab`** — less obvious. It comes from `_safe_nav` (`ui/components/feedback_modal.py:128`),
  which reads `input.main_nav()`: a **client-settable Shiny input with no whitelist**. A crafted
  client can put arbitrary text in it, and that text lands in the public issue body. It is not
  machine-generated and it is not safe to paste unread.

Neither is a leak of a third party's data — it is self-disclosure by the submitter — but it is
still content you are about to publish under the project's name, and a reporter may have pasted a
path, a hostname, an internal URL or a colleague's address into it.

> **The maintainer's review before filing is the ONLY gate on that text.** The feature strips the
> structured contact channel; it does not and cannot sanitise free text. Read the prefilled body
> before you press "Submit new issue" — including the `tab:` line. A maintainer who believes "the
> feature strips PII" and files unread is the failure mode this section exists to prevent.

The link opens the prefilled form; nothing is filed until you submit it. If the body is
unacceptable, edit it in GitHub's form or close the tab.

## 10. Log line quick reference

All logging goes to **stderr** via `osmose/logging.py`, formatted
`<time> [<logger>] <LEVEL>: <message>`. On the production systemd deployment that means
`journalctl -u osmose-shiny.service`.

Several messages contain em dashes, so these grep substrings deliberately stop before them.

| Level | Grep for | Means |
|---|---|---|
| WARNING | `X-Forwarded-For present but` | Degraded mode 1 — set `OSMOSE_TRUSTED_PROXY`. |
| WARNING | `has no usable rightmost entry` | Degraded mode 2 — malformed XFF; all clients share one bucket. |
| WARNING | `No client address available` | Degraded mode 3 — throttling is global. |
| WARNING | `No X-Forwarded-For header and the connecting address` | Degraded mode 4 — the proxy is not forwarding XFF; throttling is global. Heuristic, so a genuine localhost connection can trip it. |
| WARNING | `Rate limiter table at capacity` | Saturation. **Alert on presence, never on rate** (§5). |
| ERROR | `feedback save failed` | Store write failed — permissions, or the 50 MiB cap (§7). Traceback names which. |
| ERROR | `contact address was LOST` | Record stored, address not. Names the feedback id. Includes the full-contacts-store case (`contacts store is full` in the traceback). |
| ERROR | `EVERY post-write step failed` | User saw no confirmation; expect a duplicate submission. |
| ERROR | `feedback review page failed` | Review page returned `500` — usually a bad `_REPO_URL` (§6). |
| ERROR | `feedback API failed` | `/api/feedback` returned `500`. |
| WARNING | `Skipping corrupt feedback line` | A malformed JSON line in the store; the read continues. |
| WARNING | `Skipping non-object feedback line` | Valid JSON that is not an object; skipped so one bad line cannot hide every record. |
| INFO | `feedback honeypot filled` | A submission was silently dropped as automated (§4). |

The degraded-mode warnings fire **once per process**. Their absence long after a restart proves
nothing.
