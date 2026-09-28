"""Light-mode text contrast on Bootstrap tables (regression gate for issue #160).

ADVISORY: CI does not run ``-m e2e`` (``addopts`` excludes it); run locally with
``pytest -m e2e tests/test_e2e_theme_contrast.py``. The CI-side gate for this bug is the
``visual-gate`` job's ``advanced.png`` baseline.

Mechanism under test: Bootstrap 5.3 paints every table cell with
``background-color: var(--bs-table-bg)`` (which resolves to the superhero theme's dark
``--bs-body-bg``) and draws striping as an inset ``box-shadow``. ``www/osmose.css`` overrides
the ``td`` background only on odd rows, so in light mode the EVEN rows -- and every cell of a
non-striped table -- kept a dark background under dark light-theme text (1.08:1 measured).
The fix clears ``--bs-table-bg`` under ``[data-theme="light"] .table``.

Mutation-checked: reverting that one declaration drops the even-row ratio to ~1.1 and the
assertion fails; odd rows stay green either way, which is why BOTH parities are asserted.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

from tests._e2e_support import dismiss_changelog_modal

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")

_WCAG_AA_NORMAL_TEXT = 4.5

# Composites each cell's effective background (ancestor rgba layers, then Bootstrap's inset
# box-shadow) and returns the WCAG contrast ratio against the cell's text colour. The raw
# ``backgroundColor`` alone would lie: the striped rows are ``rgba(0,0,0,0.016)`` over the
# card, and the dark even rows only reach rgb(27,48,65) AFTER the 5 % white shadow.
_CONTRAST_JS = """
() => {
  const parse = (s) => {
    const m = (s.match(/[\\d.]+/g) || []).map(Number);
    return { r: m[0] || 0, g: m[1] || 0, b: m[2] || 0, a: m.length > 3 ? m[3] : 1 };
  };
  const over = (top, bot) => ({
    r: top.r * top.a + bot.r * (1 - top.a),
    g: top.g * top.a + bot.g * (1 - top.a),
    b: top.b * top.a + bot.b * (1 - top.a),
    a: 1,
  });
  const lum = (c) => {
    const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
    return 0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b);
  };
  const ratio = (a, b) => {
    const [h, l] = [lum(a), lum(b)].sort((x, y) => y - x);
    return (h + 0.05) / (l + 0.05);
  };
  const effBg = (el) => {
    const chain = [];
    for (let e = el; e; e = e.parentElement) chain.push(getComputedStyle(e).backgroundColor);
    let acc = { r: 255, g: 255, b: 255, a: 1 };
    for (const c of chain.reverse()) { const p = parse(c); if (p.a > 0) acc = over(p, acc); }
    return acc;
  };
  const rows = document.querySelectorAll('table.table-striped tbody tr');
  const pick = (tr) => {
    const td = tr.querySelector('td');
    const cs = getComputedStyle(td);
    const shadow = cs.boxShadow === 'none' ? { r: 0, g: 0, b: 0, a: 0 } : parse(cs.boxShadow.split(')')[0] + ')');
    const bg = over(shadow, effBg(td));
    return { text: td.textContent.trim().slice(0, 40), bg: `rgb(${Math.round(bg.r)},${Math.round(bg.g)},${Math.round(bg.b)})`,
             color: cs.color, contrast: ratio(parse(cs.color), bg) };
  };
  return { n: rows.length, odd: pick(rows[0]), even: pick(rows[1]) };
}
"""


def test_light_mode_advanced_table_rows_both_readable(page: Page, app: ShinyAppProc):
    page.add_init_script("try { localStorage.setItem('osmose-theme', 'light'); } catch (e) {}")
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=15000)
    dismiss_changelog_modal(page)
    assert page.evaluate("document.documentElement.getAttribute('data-theme')") == "light"

    # The Advanced table needs a loaded config; the `minimal` demo lives on the Domain page.
    page.locator(".nav-pills .nav-link[data-value='grid']").click()
    page.wait_for_selector("#load_example", timeout=15000)
    page.select_option("#load_example", "minimal")
    page.click("#btn_load_example")
    page.locator(".nav-pills .nav-link[data-value='advanced']").click()
    page.wait_for_selector("table.table-striped tbody tr", timeout=15000)

    m = page.evaluate(_CONTRAST_JS)
    assert m["n"] >= 2, f"need two rows to compare parities, got {m['n']}"
    for parity in ("odd", "even"):
        cell = m[parity]
        assert cell["contrast"] >= _WCAG_AA_NORMAL_TEXT, (
            f"{parity} row '{cell['text']}' is {cell['contrast']:.2f}:1 "
            f"({cell['color']} on {cell['bg']}); WCAG AA needs {_WCAG_AA_NORMAL_TEXT}:1"
        )
