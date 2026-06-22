"""Shared config-diff presentation: classify + render a Key/A/B/Change table.

Single source of truth for what a config diff looks like, consumed by both the
Scenario Diff run-comparison page and the Scenarios page's compare modal. Pure:
takes [{key, value_a, value_b}] dicts, returns classified rows / a Shiny UI
element. No reactive, no I/O. Callers own the empty/edge cases (the wording
differs per surface) and pass only NON-empty diff lists to the renderer.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict, cast

from shiny import ui

from ui.styles import STYLE_MONO_KEY, STYLE_SCROLL_TABLE

_CHANGE_ORDER = {"changed": 0, "added": 1, "removed": 2}


class DiffRow(TypedDict):
    """A row in a config diff table with classification."""

    key: str
    value_a: str | None
    value_b: str | None
    change: Literal["changed", "added", "removed"]


def classify_config_diffs(
    diffs: list[dict[str, str | None]],
) -> list[DiffRow]:
    """Tag each {key, value_a, value_b} row with a change type and sort.

    change is "added"   when value_a is None (key only in B),
              "removed" when value_b is None (key only in A),
              "changed" otherwise (both present, differ — incl. an empty-string
              value, since only None means a missing key).
    Sorted changed-group-first, then added, then removed; alphabetical by key
    within each group. Deterministic and independent of input order. Pure.
    """
    rows: list[DiffRow] = []
    for d in diffs:
        va = d.get("value_a")
        vb = d.get("value_b")
        if va is None:
            change: Literal["changed", "added", "removed"] = "added"
        elif vb is None:
            change = "removed"
        else:
            change = "changed"
        row: DiffRow = cast(
            DiffRow,
            {
                "key": d["key"],
                "value_a": va,
                "value_b": vb,
                "change": change,
            },
        )
        rows.append(row)
    rows.sort(key=lambda r: (_CHANGE_ORDER[r["change"]], r["key"]))
    return rows


def render_config_diff_table(diffs: list[dict[str, str | None]]):
    """Classify raw diff dicts and return a count line + badged, sorted,
    scrollable Key/A/B/Change table. Pass only NON-empty diff lists."""
    rows = classify_config_diffs(diffs)
    n = len(rows)
    badge_cls = {"changed": "bg-secondary", "added": "bg-success", "removed": "bg-danger"}

    def _val_cell(v):
        return ui.tags.td("—" if v is None else v)

    body = [
        ui.tags.tr(
            ui.tags.td(r["key"], style=STYLE_MONO_KEY),
            _val_cell(r["value_a"]),
            _val_cell(r["value_b"]),
            ui.tags.td(ui.tags.span(r["change"], class_=f"badge {badge_cls[r['change']]}")),
        )
        for r in rows
    ]
    table = ui.tags.table(
        ui.tags.thead(
            ui.tags.tr(
                ui.tags.th("Key"),
                ui.tags.th("A"),
                ui.tags.th("B"),
                ui.tags.th("Change"),
            )
        ),
        ui.tags.tbody(*body),
        class_="table table-sm table-striped",
        style="font-size: 13px;",
    )
    return ui.div(
        ui.p(f"{n} differing config key{'s' if n != 1 else ''}", class_="text-muted"),
        ui.div(table, style=STYLE_SCROLL_TABLE),
    )
