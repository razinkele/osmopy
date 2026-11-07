"""``osmose.java_staging``: copying the config directory must not recurse into the stage itself.

``write_temp_config`` copies the whole source directory into the stage. When the stage sits inside
that source, as it does when calibration stages under a ``work_dir`` kept beside its config, the
copy reached the stage directory it had just created and copied it into itself until the path
grew too long.
"""

from __future__ import annotations

import pytest

from osmose.java_staging import write_temp_config


@pytest.mark.parametrize("inside", ["stage", "work/stage"])
def test_a_stage_inside_the_source_dir_is_not_copied_into_itself(tmp_path, inside):
    src = tmp_path / "cfg"
    (src / "maps").mkdir(parents=True)
    (src / "maps" / "m.csv").write_text("x\n")
    stage = src / inside

    master = write_temp_config({"simulation.nspecies": "0"}, stage, source_dir=src)

    assert master == stage / "osm_all-parameters.csv" and master.exists()
    assert (stage / "maps" / "m.csv").read_text() == "x\n", "config files must still be copied"
    nested = stage / inside.split("/")[0]
    assert not nested.exists(), f"the stage was copied into itself ({nested} exists)"
