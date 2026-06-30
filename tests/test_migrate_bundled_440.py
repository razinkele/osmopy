from pathlib import Path

from scripts.migrate_bundled_to_440 import _collect_param_files, _convert_line, _rename_forward


def test_rename_forward_prefix():
    assert _rename_forward("fisheries.enabled") == "module.multispecies.fisheries.enabled"
    # version-stable keys pass through unchanged
    assert (
        _rename_forward("mortality.additional.larva.rate.sp0")
        == "mortality.additional.larva.rate.sp0"
    )
    assert _rename_forward("species.linf.sp0") == "species.linf.sp0"


def test_convert_line_rate_scale_and_version():
    ok: set[str] = set()  # no pre-existing keys -> no skip-if-exists drops
    assert _convert_line("mortality.additional.larva.rate.sp0;2.145\n", 24.0, ok).strip() == (
        f"mortality.additional.larva.rate.sp0;{2.145 * 24.0!r}"
    )
    assert _convert_line("osmose.version;4.3.3\n", 24.0, ok).strip() == "osmose.version;4.4.1"
    # comments + blanks + non-rate keys untouched (key only renamed)
    assert _convert_line("# a comment\n", 24.0, ok) == "# a comment\n"
    assert _convert_line("species.linf.sp0;100\n", 24.0, ok).strip() == "species.linf.sp0;100"


def test_convert_line_skip_if_exists_drops_colliding_rename():
    # fisheries.enabled -> module.multispecies.fisheries.enabled; if the NEW key already exists in
    # the source, the OLD line is dropped (no duplicate).
    assert (
        _convert_line("fisheries.enabled;true\n", 24.0, {"module.multispecies.fisheries.enabled"})
        is None
    )


def test_param_file_guard_excludes_matrices():
    master = next(Path("data/eec_full").glob("*all-parameters*.csv"))
    params = {p.name for p in _collect_param_files(master)}
    assert "eec_param-species.csv" in params  # a key-value param file IS included
    assert "predation-accessibility.csv" not in params  # a matrix CSV is NOT
    assert "grid-mask.csv" not in params
