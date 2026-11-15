"""The resource table's sync must skip per-stage array cells, decided from the config.

``render_species_table`` (shared with the species table) shows an array cell read-only with no
input. Shiny keeps the last value of an input removed from the page, so without a config-based
skip a stale input would write a scalar over the array.
"""

from __future__ import annotations

from ui.pages.forcing import collect_resource_table_updates


def test_resource_table_sync_skips_an_array_cell_with_a_stale_input():
    n_focal = 2
    cfg = {"species.size.min.sp2": "0.1;0.2", "species.size.min.sp3": "0.5"}
    stale = {"spt_species_size_min_2": 0.002, "spt_species_size_min_3": 0.7}

    def read(input_id):
        if input_id not in stale:
            raise AttributeError(input_id)
        return stale[input_id]

    updates = collect_resource_table_updates(2, n_focal, cfg, read)
    assert "species.size.min.sp2" not in updates  # the array is left alone
    assert updates["species.size.min.sp3"] == "0.7"  # a plain cell still syncs
