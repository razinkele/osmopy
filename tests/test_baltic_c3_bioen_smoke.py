"""C3 spec Sec.7: the realistic-config bioen regression.

Runs PRODUCTION `data/baltic` (unmodified on disk) merged in-memory with the C3 overlay
(`data/baltic/scenarios/c3_bioen/c3_bioen_arm.json`, Task 11's fit), with
`population.seeding.year.max = 1` so every species leaves OSMOSE's synthetic seeding
bootstrap in year 1 -- gonad-derived spawning has to carry every population from year 2
onward, or it goes extinct. That is the entire point of the test: it is a regression guard
against the bioen arm being 100% bootstrap-sustained (Task 8's `data/examples_bioen` failure
mode), not a measurement of OSMOSE's seeding machinery itself.

WHICH POPULATION EACH ASSERTION COUNTS (CLAUDE.md gotcha -- do not conflate these):
  - `biomass()` applies `output.cutoff.age` (0.5 yr for every Baltic species), so the tail
    assertions below count fish >= 6 months old, NOT eggs/YOY. A population that looks
    "collapsed" here could in principle still have a live YOY pool -- these assertions
    do not see that pool, by design (it is not old enough to indicate the population is
    self-sustaining past the bootstrap).
  - `res.ssb()` (mature-schools-only spawning-stock biomass) is a STRONGER check than the
    biomass tail: a long-lived species' original year-1 seeded cohort could still be alive
    and growing at year 8 without ever having reproduced. SSB > 0 in the final year proves
    an individual actually reached reproductive maturity and spawned -- i.e. that gonad-
    derived spawning, not bootstrap carryover, is what's being observed. `output.ssb.enabled`
    is off by default in production `data/baltic` (confirmed 2026-09-05: without it,
    `OsmoseResults.ssb()` raises `FileNotFoundError` in in-memory mode, same as
    `scripts/baltic_c3_bioen_ab.py`'s harness hit and chose to skip around -- this test
    opts the output ON instead, in-memory only, because the SSB check is the whole point).

CURRENT STATUS (measured seed=42, confirmed on seeds 1 and 7, 8-yr window, this branch head):
FAILS, and this is a diagnosed REAL FINDING, not a broken test -- see
`.superpowers/sdd/2026-08-30-baltic-c3-bioen-stage1/task-13-report.md` for full numbers.
Under the seeding cutoff: cod_west and cod_east go EXTINCT (biomass, abundance-by-age, AND
ssb all hit exactly 0.0 by year 4 in every seed tested -- ssb stays exactly 0.0 for their
ENTIRE lives, i.e. no individual of either species ever reaches reproductive maturity before
the cohort dies out); flounder also goes extinct (abundance-by-age hits exactly 0.0 by year
7) but DOES briefly reach positive ssb in years 3-4 (~3.9, ~3.1) before the population is
lost anyway -- reproduction happened but could not outpace losses; herring is not extinct by
year 8 but collapses >99.99% (abundance-by-age 5.19e12 -> 2.82e5, year 1 to year 8, ssb
falling from ~4.0e4 to ~1.8e-8, still declining). Only sprat sustains, with real, growing
ssb throughout. A same-seed, same-window CONTROL run of the IDENTICAL production config with
the C3 overlay removed (classic growth, bioen off) SUSTAINS AND GROWS all five species under
the identical seeding cutoff -- ruling out a generic seeding-bootstrap artifact (contrast
Task 8's `data/examples_bioen` collapse, which was an orthogonal accessibility-matrix
orientation bug that broke classic growth too, so it implicated no bioen-specific mechanism).
This isolates the cause to the bioen arm.

Mechanism, from `mortality()`'s per-cause rates (`osmose/engine/output.py`'s Java-matching
convention, NOT `meanEnetFaced`, which averages only over schools present at each output
step and is therefore survivor-biased -- it cannot see schools that starved to death within
a step, so a positive mean would be consistent with heavy starvation mortality and is not
usable as counter-evidence): PREDATION, not starvation, is the dominant and ultimately
complete proximate cause for every collapsing species. Juvenile predation-mortality rate for
cod_west climbs 0.13 (yr1) -> 2.32 (yr2) -> inf (yr3, i.e. every juvenile at risk that step
was eaten -- N_end=0 with nonzero deaths, a literal complete-cohort wipeout, not a rate sum
artifact); cod_east and flounder show the same inf-rate wipeout pattern at their own
extinction years; herring's ADULT predation rate reaches inf at both year 7 and year 8. Over
the same span, STARVATION rates for the collapsing juveniles stay an order of magnitude
smaller and, for cod_west, DECLINE (0.39 -> 0.13 -> 0.01) even as predation explodes -- the
opposite of a starvation-driven signature. This does not by itself prove a root cause (a
plausible but UNTESTED hypothesis: slower bioen-driven growth keeps individuals in a
predation-vulnerable size window longer, raising cumulative exposure -- Task 14's to chase,
not this test's), and Gate B's PASS (Task 9) bounds the bioen engine mechanics themselves, so
the likelier locus is Task 11's fitted growth/maturity parameters under this stress
condition. Per the plan's explicit instruction, this assertion is NOT weakened, the seeding
window is NOT extended, and the overlay is NOT retuned to force a pass -- the failure is the
deliverable, kept executing (`xfail(strict=True)`, not skipped) so it flips loudly the moment
someone changes the overlay or the growth/maturity/reproduction path.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from osmose.config import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine import PythonEngine

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "data" / "baltic" / "scenarios" / "c3_bioen" / "c3_bioen_arm.json"


@pytest.mark.integration
@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN GAP, owner: C3 Stage-1 Task 14 (or whoever refits Task 11's overlay). "
        "Measured (seed=42, confirmed on seeds 1 and 7): production Baltic + the C3 bioen "
        "overlay collapses cod_west and cod_east to total extinction by year 4 (ssb=0.0 for "
        "their entire lives -- no individual ever reaches maturity), flounder to extinction "
        "by year 7 (despite briefly positive ssb in years 3-4), and herring >99.99% by year "
        "8, once population.seeding.year.max=1 forces real reproduction to carry the "
        "population past OSMOSE's synthetic seeding bootstrap. Only sprat sustains. A "
        "same-seed control run of the identical config with the C3 overlay removed (classic "
        "growth, bioen off) sustains and GROWS all five species under the identical seeding "
        "cutoff, isolating the cause to the bioen arm -- not a generic seeding-bootstrap "
        "artifact. Mechanism (see module docstring): PREDATION mortality, not starvation, is "
        "the dominant and ultimately complete (rate=inf, literal cohort wipeout) proximate "
        "cause; starvation rates stay small and declining over the same window. Full "
        "numbers: task-13-report.md. strict=True so this flips loudly (XPASS -> failure) the "
        "moment the overlay or the growth/maturity/reproduction path changes enough to fix "
        "it -- do not weaken this test's assertions or extend the seeding window to force a "
        "pass; the collapse is the finding, not a test bug."
    ),
)
def test_bioen_arm_sustains_populations_past_the_seeding_window():
    demo_dir = Path(tempfile.mkdtemp())
    config_file = osmose_demo("baltic", demo_dir)["config_file"]
    cfg = dict(OsmoseConfigReader().read(str(config_file)))
    ov = {k: v for k, v in json.loads(OVERLAY.read_text()).items() if not k.startswith("_")}
    cfg.update(ov)
    cfg["simulation.time.nyear"] = "8"
    cfg["population.seeding.year.max"] = "1"
    # Off by default in production data/baltic -- see docstring. In-memory-only override;
    # data/baltic/*.csv on disk is untouched.
    cfg["output.ssb.enabled"] = "true"
    res = PythonEngine().run_in_memory(cfg, seed=42)

    bio = res.biomass()  # >= 6-month-old population only -- see docstring.
    for sp in ("cod_west", "cod_east", "herring", "sprat", "flounder"):
        s = bio[sp].to_numpy(dtype=float)
        assert np.isfinite(s).all() and s[-1] > 0.0, sp
        tail = s[-3:]
        assert not (tail[2] < tail[1] < tail[0] and tail[2] < 0.5 * tail[0]), (
            f"{sp}: monotone collapse in years 6-8"
        )

    # Gonad-derived spawning happened after the window closed: SSB > 0 in the last year for
    # every assessed stock (mature-schools-only population -- see docstring for why this is
    # a stronger check than the biomass tail alone).
    ssb = res.ssb()
    for sp in ("cod_west", "cod_east", "herring", "sprat", "flounder"):
        assert ssb[sp].to_numpy(dtype=float)[-1] > 0.0, sp
