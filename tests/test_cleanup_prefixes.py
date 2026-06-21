"""Item 2: wizard/maps/validation tempdirs must be swept by cleanup_old_temp_dirs.

These dirs are session-lifetime (osmose_wizard_/osmose_maps_ back state.config_dir)
or transient (osmose_val_), so they are cleaned by the age-gated sweep, never by a
prompt rmtree in their create handler.
"""

import tempfile

from osmose.cleanup import _OSMOSE_PREFIXES, cleanup_old_temp_dirs


def test_new_prefixes_registered():
    for prefix in ("osmose_wizard_", "osmose_maps_", "osmose_val_"):
        assert prefix in _OSMOSE_PREFIXES


def test_sweep_removes_new_prefix_dirs(tmp_path, monkeypatch):
    # Isolate the sweep to a private temp root -- cleanup_old_temp_dirs(0) rmtrees
    # EVERY osmose-prefixed dir under gettempdir(); without this it would delete
    # real /tmp osmose dirs that concurrent xdist tests are using. Mirrors the
    # pattern in tests/test_cleanup.py.
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    made = [tmp_path / f"{p}x" for p in ("osmose_wizard_", "osmose_maps_", "osmose_val_")]
    for d in made:
        d.mkdir()
    cleanup_old_temp_dirs(max_age_hours=0)  # 0 == remove all osmose temp dirs
    for d in made:
        assert not d.exists(), f"{d} was not swept"
