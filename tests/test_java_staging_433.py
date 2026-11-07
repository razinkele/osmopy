"""4.3.3 staging must give Java the resource step count it requires.

Java 4.3.3's ``ResourceForcing.init()`` reads ``species.biomass.nsteps.year.spN``, falling back to
the global ``species.biomass.nsteps.year``, for every resource species forced from a NetCDF file
(``species.file.spN``), and exits when neither is set ("Could not find parameter
species.biomass.nsteps.year"). The bundled Bay of Biscay config has never carried the key: the
Python engine reads the step count off the NetCDF itself. The writer's 4.3.x branch does not
synthesize it either, so choosing the 4.3.3 jar in the Run tab failed on that config.

The unit tests pin the staged OUTPUT, need no jar, and run in CI. The last test runs the real
4.3.3 jar through the Run tab's own runner and skips where the jar or ``java`` is missing.
"""

from __future__ import annotations

import queue
import shutil
from pathlib import Path

import pytest

from osmose.config.reader import OsmoseConfigReader
from ui.pages.run import stage_config_for_java

ROOT = Path(__file__).resolve().parents[1]
BOB = ROOT / "data" / "examples" / "osm_all-parameters.csv"
JAR_433 = ROOT / "osmose-java" / "osmose_4.3.3-jar-with-dependencies.jar"
_KEY = "species.biomass.nsteps.year"


def _master_entries(master: Path) -> list[tuple[str, str]]:
    out = []
    for raw in master.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ";" not in line:
            continue
        key, _, value = line.partition(";")
        out.append((key.strip(), value.strip()))
    return out


def _bob() -> dict[str, str]:
    raw = dict(OsmoseConfigReader().read(str(BOB)))
    file_forced = [k for k in raw if k.startswith("species.file.sp")]
    assert file_forced, "fixture premise broke: BoB no longer forces resources from a NetCDF"
    assert not any(k.startswith(_KEY) for k in raw), f"fixture premise broke: BoB now sets {_KEY}"
    return raw


def test_433_staging_adds_the_resource_step_count(tmp_path):
    raw = _bob()
    master, _ = stage_config_for_java(raw, tmp_path / "stage", BOB.parent, target_version="4.3.3")
    values = [v for k, v in _master_entries(master) if k == _KEY]
    assert values == [raw["simulation.time.ndtperyear"]], (
        f"4.3.3 staging wrote {_KEY} = {values}; Java 4.3.3 exits without it"
    )


def test_433_staging_keeps_a_value_the_config_already_sets(tmp_path):
    raw = {**_bob(), _KEY: "12"}
    master, _ = stage_config_for_java(raw, tmp_path / "stage", BOB.parent, target_version="4.3.3")
    assert [v for k, v in _master_entries(master) if k == _KEY] == ["12"]


def test_433_staging_adds_nothing_when_every_resource_has_its_own_count(tmp_path):
    raw = _bob()
    for key in [k for k in raw if k.startswith("species.file.sp")]:
        raw[f"{_KEY}.sp{key.rsplit('.sp', 1)[1]}"] = "24"
    master, _ = stage_config_for_java(raw, tmp_path / "stage", BOB.parent, target_version="4.3.3")
    assert _KEY not in dict(_master_entries(master))


def test_44x_staging_is_untouched_by_the_433_rule(tmp_path):
    """4.4.x gets per-species keys from the writer; the 4.3.3 global fallback must not leak in."""
    master, _ = stage_config_for_java(
        _bob(), tmp_path / "stage", BOB.parent, target_version="4.4.1"
    )
    entries = dict(_master_entries(master))
    assert _KEY not in entries
    assert any(k.startswith(f"{_KEY}.sp") for k in entries), "writer's 4.4.x keys went missing"


_HAVE_433 = shutil.which("java") is not None and JAR_433.exists()


@pytest.mark.skipif(not _HAVE_433, reason="needs `java` and osmose-java/ 4.3.3 jar")
def test_bob_runs_on_the_433_jar_through_the_run_tab_runner(tmp_path):
    """The Run tab's own staging + runner, with the 4.3.3 jar on BoB, for one year."""
    from osmose.config.aliases import target_version_for_jar
    from osmose.runner import OsmoseRunner
    from ui.pages.run import _java_engine_thread

    master, overrides = stage_config_for_java(
        _bob(), tmp_path / "work", BOB.parent, target_version=target_version_for_jar(JAR_433)
    )
    # BoB records from year 20; start at 0 so a one-year run writes rows to check.
    overrides = {**overrides, "simulation.time.nyear": "1", "output.start.year": "0"}
    out = tmp_path / "work" / "output"
    log_q: queue.Queue = queue.Queue()
    done_q: queue.Queue = queue.Queue()
    _java_engine_thread(
        OsmoseRunner(jar_path=JAR_433), master, out, None, overrides, 600, log_q, done_q
    )

    kind, result, _msg = done_q.get_nowait()
    assert kind == "done"
    # Java prints its exception to STDOUT; stderr holds only SLF4J noise.
    assert result.returncode == 0, (
        f"4.3.3 jar failed on staged BoB.\nSTDOUT tail:\n{result.stdout[-1500:]}\n"
        f"STDERR tail:\n{result.stderr[-500:]}"
    )
    biomass = out / "biscay_biomass_Simu0.csv"
    assert biomass.exists(), f"no biomass output; wrote: {sorted(p.name for p in out.glob('*'))}"
    assert len(biomass.read_text().splitlines()) > 2, "biomass output has no data rows"
