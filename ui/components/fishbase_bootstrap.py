"""'Bootstrap from FishBase' inline panel: resolve -> (disambiguate) -> review -> apply.

Rendered INLINE in the Species Configuration card (not a modal): the output_ui
placeholders live in the static page body — the proven pattern in this repo. Pure helpers
(candidate_label, review_rows, apply_traits) are unit-tested; the server wires them to
render.ui outputs. Data is CC-BY-NC (FishBase via rOpenSci / Source Cooperative).
"""

from __future__ import annotations

from shiny import reactive, render, ui
from shiny.types import SilentException

from osmose import fishbase
from osmose.logging import setup_logging
from osmose.schema.species import SPECIES_FIELDS

_log = setup_logging("osmose.fishbase_ui")
_ATTRIB = "Data: FishBase/SeaLifeBase via rOpenSci / Source Cooperative (CC-BY-NC)."

# key stem -> OsmoseField (for resolve_key + description/unit). OsmoseField has
# `description` (the UI display string) — there is NO `.label` attribute.
_FIELD_BY_STEM = {f.key_pattern.replace(".sp{idx}", ""): f for f in SPECIES_FIELDS}


def _pick_id(species_index: int, key: str) -> str:
    """Shiny-safe checkbox id (dots illegal in input ids; namespace by species)."""
    return f"fb_pick_{species_index}_" + key.replace(".", "_")


def candidate_label(m) -> str:
    """Human label for a SpecMatch in the disambiguation dropdown."""
    common = f" — {m.common_name}" if m.common_name else ""
    return f"{m.scientific_name}{common} [{m.db.upper()}]"


def review_rows(cfg: dict, species_index: int, traits: dict) -> list[dict]:
    """Build review-table rows pairing current config value with the fetched median."""
    rows = []
    for key, est in traits.items():
        field = _FIELD_BY_STEM.get(key)
        rows.append(
            {
                "key": key,
                "label": field.description if field else key,
                "current": cfg.get(field.resolve_key(species_index)) if field else None,
                "fetched": est.value,
                "n": est.n,
                "range": (est.min, est.max),
                "unit": est.unit,
            }
        )
    return rows


def apply_traits(cfg: dict, species_index: int, traits: dict, selected: set[str]) -> dict:
    """Return a NEW config with only the selected traits written to sp{index} keys."""
    out = dict(cfg)
    for key in selected:
        field = _FIELD_BY_STEM.get(key)
        est = traits.get(key)
        if field is None or est is None:
            continue
        out[field.resolve_key(species_index)] = str(est.value)
    return out


def fishbase_bootstrap_ui():
    """Inline panel embedded in the Species Configuration card (static output_ui slots)."""
    return ui.div(
        ui.hr(),
        ui.h6("Bootstrap from FishBase"),
        ui.output_ui("fb_species_select"),
        ui.input_text("fb_name", "Scientific or common name"),
        ui.input_action_button("fb_fetch", "Fetch traits", class_="btn-primary btn-sm"),
        ui.output_ui("fb_candidates"),
        ui.output_ui("fb_review"),
        ui.tags.small(_ATTRIB, class_="text-muted d-block mt-2"),
        class_="mt-2",
    )


def fishbase_bootstrap_server(input, output, session, state):
    _matches: reactive.Value = reactive.Value([])  # candidates from the last resolve
    _traits: reactive.Value = reactive.Value({})
    _match: reactive.Value = reactive.Value(None)  # the chosen SpecMatch

    def _n_species() -> int:
        with reactive.isolate():
            cfg = state.config.get()
            names = state.species_names.get() or []
        try:
            return int(float(cfg.get("simulation.nspecies", len(names)) or 0))
        except (TypeError, ValueError):
            return len(names)

    @render.ui
    def fb_species_select():
        state.load_trigger.get()  # refresh slots when species count/names change
        with reactive.isolate():
            names = state.species_names.get() or []
        n = _n_species()
        choices = {
            str(i): (names[i] if i < len(names) and names[i] else f"Species {i}")
            for i in range(max(n, 0))
        }
        return ui.input_select("fb_species", "Species (config slot)", choices=choices)

    def _do_fetch(m):
        """Fetch + store traits for a chosen match (busy-wrapped)."""
        ui.notification_show("Fetching from FishBase…", id="fb_busy", duration=None)
        try:
            traits = fishbase.fetch_traits(m.spec_code, m.db)
        except fishbase.FishBaseUnavailable:
            _traits.set({})
            _match.set(None)
            ui.notification_show(
                "FishBase unavailable — try again later.", type="error", duration=8
            )
            return
        finally:
            ui.notification_remove("fb_busy")
        _match.set(m)
        _traits.set(traits)

    @reactive.effect
    @reactive.event(input.fb_fetch)
    def _fetch():
        name = (input.fb_name() or "").strip()
        if not name:
            ui.notification_show("Enter a species name.", type="warning", duration=5)
            return
        _matches.set([])
        _traits.set({})
        _match.set(None)
        ui.notification_show("Looking up species…", id="fb_busy", duration=None)
        try:
            matches = fishbase.resolve_species(name)
        except fishbase.FishBaseNoMatch:
            ui.notification_show(
                f"No FishBase/SeaLifeBase record for '{name}'.", type="error", duration=8
            )
            return
        except fishbase.FishBaseUnavailable:
            ui.notification_show(
                "FishBase unavailable — try again later.", type="error", duration=8
            )
            return
        finally:
            ui.notification_remove("fb_busy")
        _matches.set(matches)
        if len(matches) == 1:
            _do_fetch(matches[0])

    @render.ui
    def fb_candidates():
        matches = _matches.get()
        if len(matches) <= 1:
            return ui.div()
        choices = {str(i): candidate_label(m) for i, m in enumerate(matches)}
        return ui.div(
            ui.input_select("fb_candidate", f"{len(matches)} matches — pick one", choices=choices),
            ui.input_action_button(
                "fb_use_candidate", "Use this match", class_="btn-secondary btn-sm"
            ),
            class_="mt-2",
        )

    @reactive.effect
    @reactive.event(input.fb_use_candidate)
    def _pick():
        matches = _matches.get()
        try:
            i = int(input.fb_candidate())
        except (TypeError, ValueError, SilentException):
            return
        if 0 <= i < len(matches):
            _do_fetch(matches[i])

    @render.ui
    def fb_review():
        traits = _traits.get()
        m = _match.get()
        if not traits or m is None:
            return ui.div()
        with reactive.isolate():
            cfg = state.config.get()
        idx = int(input.fb_species())
        header = ui.tags.div(
            f"{m.scientific_name} ({m.common_name}) — {m.db.upper()}", class_="fw-bold mb-1"
        )
        rows = [
            ui.tags.tr(
                ui.tags.td(ui.input_checkbox(_pick_id(idx, r["key"]), "", value=True)),
                ui.tags.td(r["label"]),
                ui.tags.td("" if r["current"] is None else str(r["current"])),
                ui.tags.td(f"{r['fetched']:.4g} {r['unit']}"),
                ui.tags.td(str(r["n"])),
                ui.tags.td(f"{r['range'][0]:.4g}–{r['range'][1]:.4g}"),
            )
            for r in review_rows(cfg, idx, traits)
        ]
        return ui.div(
            header,
            ui.tags.table(
                ui.tags.thead(
                    ui.tags.tr(
                        *[
                            ui.tags.th(h)
                            for h in ("✓", "Trait", "Current", "FishBase", "n", "Range")
                        ]
                    )
                ),
                ui.tags.tbody(*rows),
                class_="table table-sm",
            ),
            ui.input_action_button("fb_apply", "Apply selected", class_="btn-success btn-sm"),
        )

    @reactive.effect
    @reactive.event(input.fb_apply)
    def _apply():
        traits = _traits.get()
        if not traits:
            return
        idx = int(input.fb_species())
        selected = {k for k in traits if _checkbox(input, _pick_id(idx, k))}
        with reactive.isolate():
            cfg = dict(state.config.get())
        new_cfg = apply_traits(cfg, idx, traits, selected)
        if new_cfg != cfg:
            state.load_config(new_cfg)
            state.dirty.set(True)
            with reactive.isolate():
                state.load_trigger.set(state.load_trigger.get() + 1)
        ui.notification_show(
            f"Applied {len(selected)} trait(s) to species {idx}.", type="message", duration=4
        )


def _checkbox(input, input_id: str) -> bool:
    try:
        return bool(getattr(input, input_id)())
    except (AttributeError, SilentException):
        return False
