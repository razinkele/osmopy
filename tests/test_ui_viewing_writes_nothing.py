"""Viewing a tab must not change the config, and so cannot change a run.

Every page writes its inputs back into the config. An input that still shows what the form renders
for a key the config leaves unset (the schema default, or a placeholder such as 0) has not been
touched, and must not be written. Until 2026-11-15 it was: opening the Setup tab wrote
``species.lmax = 120`` into every demo (unset means no length cap in both engines) and, on
baltic_a2, mortality rates of 0.2 and 2.0/yr over keys the engine reads as 0; a placeholder
``movement.distribution.ncell = 0`` is a one-cell patch in Java (unset: the whole domain).

These tests render each demo with the app's own render functions, read back the value each input
would send from the generated HTML, feed those values to the page write paths (``sync_inputs`` and
the species and resource table collectors), and require that nothing would be written. A real
browser run of the same claim is ``tests/test_e2e_config_reload.py``.
"""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

import pytest
from shiny import reactive

from osmose.config.reader import OsmoseConfigReader
from osmose.demo import list_demos, osmose_demo
from osmose.schema import build_registry
from osmose.schema.base import ParamType
from osmose.schema.ltl import LTL_FIELDS
from osmose.schema.species import SPECIES_FIELDS
from ui.components.param_form import (
    collect_species_table_updates,
    render_field,
    render_species_table,
    seed_new_species,
)
from ui.pages._helpers import count_map_entries, parse_nspecies
from ui.pages.forcing import collect_resource_table_updates
from ui.state import AppState, same_config_value, sync_inputs

_REGISTRY = build_registry()


class _InputValues(HTMLParser):
    """The value each input in rendered HTML sends to the server, keyed by input id."""

    def __init__(self) -> None:
        super().__init__()
        self.values: dict[str, object] = {}
        self._select: str | None = None
        self._first_option: str | None = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "input" and a.get("id"):
            kind = a.get("type")
            if kind == "number":
                raw = a.get("value")
                if raw in (None, ""):
                    self.values[a["id"]] = None
                else:
                    # The browser sends a JS number, and JSON brings an integral one back as int.
                    num = float(raw)
                    self.values[a["id"]] = int(num) if num.is_integer() else num
            elif kind == "checkbox":
                self.values[a["id"]] = "checked" in a
            elif kind == "text":
                self.values[a["id"]] = a.get("value", "")
            elif kind == "file":
                self.values[a["id"]] = None  # no upload yet
        elif tag == "select" and a.get("id"):
            self._select, self._first_option = a["id"], None
        elif tag == "option" and self._select:
            if self._first_option is None:
                self._first_option = a.get("value")
            if "selected" in a:
                self.values[self._select] = a.get("value")

    def handle_endtag(self, tag):
        if tag == "select" and self._select:
            self.values.setdefault(self._select, self._first_option)  # nothing selected: first
            self._select = None


def _values(tag) -> dict[str, object]:
    parser = _InputValues()
    parser.feed(str(tag))
    return parser.values


class _Inputs:
    """Fake Shiny ``input``: an id answers with its rendered value; an unknown id does not exist."""

    def __init__(self, values: dict[str, object]) -> None:
        self._v = values

    def __getattr__(self, name):
        if name.startswith("_") or name not in self._v:
            raise AttributeError(name)
        return lambda: self._v[name]


def _index_range(pattern: str, cfg: dict[str, str]) -> range:
    if ".fsh{idx}" in pattern:
        return range(int(float(cfg.get("simulation.nfisheries", "1") or 1)))
    if ".map{idx}" in pattern:
        return range(max(count_map_entries(cfg), 1))
    if ".mpa{idx}" in pattern:
        return range(1)
    return range(parse_nspecies(cfg, default=3))


@pytest.fixture(scope="module", params=list_demos())
def demo(request, tmp_path_factory) -> tuple[str, dict[str, str]]:
    info = osmose_demo(request.param, tmp_path_factory.mktemp(request.param))
    return request.param, dict(OsmoseConfigReader().read(Path(info["config_file"])))


def test_the_form_inputs_write_nothing(demo):
    """Every schema field the forms render through ``render_field`` (Setup globals, Movement,
    Fishing, Forcing, Grid, ...), for every index the demo uses."""
    name, cfg = demo
    keys: list[str] = []
    values: dict[str, object] = {}
    for field in _REGISTRY.all_fields():
        if field.param_type in (ParamType.FILE_PATH, ParamType.MATRIX):
            continue
        for idx in _index_range(field.key_pattern, cfg) if field.indexed else [None]:
            keys.append(field.resolve_key(idx) if idx is not None else field.key_pattern)
            values.update(_values(render_field(field, species_idx=idx, config=cfg)))
    assert len(keys) > 200
    state = AppState()
    with reactive.isolate():
        state.config.set(dict(cfg))
        written = sync_inputs(_Inputs(values), state, keys)
        after = state.config.get()
    changed = {
        k: (cfg.get(k), after.get(k)) for k in set(cfg) | set(after) if cfg.get(k) != after.get(k)
    }
    assert not changed, f"{name}: viewing the forms would write {sorted(changed.items())[:8]}"
    assert len(written) == 0 or all(k in cfg for k in written)


def test_the_species_table_writes_nothing(demo):
    name, cfg = demo
    n = parse_nspecies(cfg, default=3)
    names = [cfg.get(f"species.name.sp{i}", f"Species {i}") for i in range(n)]
    table = render_species_table(SPECIES_FIELDS, n, names, show_advanced=True, config=cfg)
    values = _values(table)
    assert values, "the parser found no inputs in the species table"
    updates = collect_species_table_updates(
        SPECIES_FIELDS, n, cfg, lambda i: getattr(_Inputs(values), i)(), show_advanced=True
    )
    changed = {
        k: (cfg.get(k), v)
        for k, v in updates.items()
        if not same_config_value(cfg.get(k), v, _REGISTRY.match_field(k))
    }
    assert not changed, (
        f"{name}: viewing the species table would write {sorted(changed.items())[:8]}"
    )


def test_the_resource_table_writes_nothing(demo):
    name, cfg = demo
    n_focal = parse_nspecies(cfg)
    n_res = int(float(cfg.get("simulation.nresource", "3") or 3)) or 3
    fields = [f for f in LTL_FIELDS if f.indexed]
    names = [cfg.get(f"species.name.sp{n_focal + i}", f"Resource {i}") for i in range(n_res)]
    table = render_species_table(
        fields, n_res, names, start_idx=n_focal, config=cfg, copy_event="copy_rsc0_to_all"
    )
    values = _values(table)
    updates = collect_resource_table_updates(
        n_res, n_focal, cfg, lambda i: getattr(_Inputs(values), i)()
    )
    changed = {
        k: (cfg.get(k), v)
        for k, v in updates.items()
        if not same_config_value(cfg.get(k), v, _REGISTRY.match_field(k))
    }
    assert not changed, f"{name}: viewing the resource table would write {sorted(changed.items())}"


def test_an_edit_in_a_table_is_still_written():
    """The untouched rule must not swallow edits: a changed value, or any value for a key the
    config sets, is written."""
    cfg = {"simulation.nspecies": "1", "species.linf.sp0": "50", "species.name.sp0": "cod"}
    values: dict[str, object] = {"spt_species_lmax_0": 120, "spt_species_linf_0": 60}
    reader = _Inputs(values)

    def collect() -> dict[str, str]:
        return collect_species_table_updates(
            SPECIES_FIELDS, 1, cfg, lambda input_id: getattr(reader, input_id)()
        )

    assert "species.lmax.sp0" not in collect()  # unset and showing its default: not written
    assert collect()["species.linf.sp0"] == "60"  # a key the config sets: written
    values["spt_species_lmax_0"] = 95
    assert collect()["species.lmax.sp0"] == "95"  # an edit of an unset key: written


def test_raising_the_species_count_seeds_only_the_new_species():
    cfg = {"simulation.nspecies": "1", "species.linf.sp0": "50"}
    seeded = seed_new_species(SPECIES_FIELDS, 1, 3, cfg)
    assert not any(k.endswith(".sp0") for k in seeded)  # an existing species is never touched
    for i in (1, 2):
        assert seeded[f"species.linf.sp{i}"] == "100.0"  # a default the engine needs to run
        assert f"growth.gompertz.linf.sp{i}" not in seeded  # no default: not seeded
    assert seed_new_species(SPECIES_FIELDS, 3, 3, cfg) == {}
    assert seed_new_species(SPECIES_FIELDS, 3, 2, cfg) == {}  # lowering seeds nothing


def test_seeding_keeps_a_re_added_age_based_species_age_based():
    """Lowering the count keeps the dropped species' keys. Raising it again must not seed a
    maturity size next to a kept maturity age: Java lets a size win, so the species would stop
    maturing by age (CodeRabbit, PR #274)."""
    cfg = {"simulation.nspecies": "1", "species.maturity.age.sp1": "2.0"}
    seeded = seed_new_species(SPECIES_FIELDS, 1, 2, cfg)
    assert "species.maturity.size.sp1" not in seeded
    assert "species.linf.sp1" in seeded  # the rest of the species is still seeded


def test_viewing_setup_on_a_config_without_a_species_count_writes_nothing():
    """A fresh session's config is empty, so it parses as 0 species while the count input starts
    at 3. Viewing the Species tab then must not "raise" the count and seed three species, nor
    write the count itself (CodeRabbit, PR #274)."""
    from ui.pages.setup import collect_setup_species_updates

    table = render_species_table(SPECIES_FIELDS, 3, ["a", "b", "c"], config={})
    values = _values(table)
    reader = _Inputs(values)
    updates = collect_setup_species_updates(3, {}, lambda i: getattr(reader, i)())
    assert updates == {}


def test_raising_a_configured_species_count_seeds_through_the_setup_sync():
    from ui.pages.setup import collect_setup_species_updates

    cfg = {"simulation.nspecies": "2", "species.linf.sp0": "50", "species.linf.sp1": "40"}
    reader = _Inputs({})  # the new species' cells are not rendered yet
    updates = collect_setup_species_updates(3, cfg, lambda i: getattr(reader, i)())
    assert updates["simulation.nspecies"] == "3"
    assert updates["species.linf.sp2"] == "100.0"
    assert not any(k.endswith((".sp0", ".sp1")) for k in updates)
