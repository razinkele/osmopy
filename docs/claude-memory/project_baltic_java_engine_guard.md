---
name: project-baltic-java-engine-guard
description: "DO NOT run Baltic (or any nbackground>0 config) on the Java engine — it's Python-engine-only by design; guard shipped 2026-06-16 (9f45cff)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3c92cf51-9a04-490e-b26d-2e39302eb27f
---

**▶▶ DO NOT try to run Baltic (or any background-species config) on the JAVA engine — it's PYTHON-ENGINE-ONLY by design.** Guard SHIPPED 2026-06-16 (master `9f45cff`, pushed direct).

The Java reference engine (`osmose_4.3.3` JAR) crashes at year 0 on Baltic: `No catchability found for prey GreySeal`. TWO independent root causes (diagnosed by reproducing + decompiling the JAR's `fr.ird.osmose.util.Matrix`): (1) **separator conflict** — OSMOSE's Java `Matrix` reader splits matrices on **`;`** (the working `predation-accessibility.csv` uses `;`), but the Python-side fishery matrices `fishery-catchability.csv`/`fishery-discards.csv` use **commas** because the Python engine reads them via `pd.read_csv(index_col=0)` (comma default, `osmose/engine/config.py:287`) → a single file can't satisfy both engines; (2) **background species omitted** — GreySeal(sp14)/Cormorant(sp15) are absent from the catchability, discards, AND predation-accessibility matrices; Java needs them everywhere (in predation-accessibility as PREDATOR columns with REAL ecological values, not zeros — seals/cormorants are top predators). Parity suite is EEC/BoB (`nbackground=0`), so never caught.

**Fix = `osmose.runner.java_engine_block_reason(config)`** (returns a reason when `simulation.nbackground>0`); `handle_run` (`ui/pages/run.py`) blocks a Java run early with a clear notification instead of launching a doomed subprocess.

**Making Baltic actually Java-runnable is a deferred design effort** (separator-agnostic readers + complete `;` background-species matrices incl. calibrated predation accessibility) — NOT worth it unless Java-parity for Baltic is specifically wanted; the Python engine runs Baltic natively (verified e2e).
