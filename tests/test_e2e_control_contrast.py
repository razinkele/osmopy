"""Switch / checkbox / radio contrast stays >= 3:1 (WCAG 1.4.11) in both themes.

Measured 2026-11-22 before the fix: the switch knob read 1.83:1 against its track when off and
2.08:1 (white on amber) when on; in the light theme an unchecked switch was white on a white card
(1.0:1). Bootstrap paints the knob/tick as an SVG data URI, so this reads the fill colour out of
the computed ``background-image`` and composites every translucent layer before scoring.

The test measures REAL controls in place (toggling ``.checked`` without firing a change event,
then restoring it): an injected Bootstrap probe passed while Shiny's native checkboxes and radios,
which carry no ``.form-check-input`` class, were still unstyled.

Run explicitly:
    pytest tests/test_e2e_control_contrast.py -m e2e -o addopts="" -p no:cacheprovider
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

from tests._e2e_support import dismiss_changelog_modal

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")
_T = 60_000
_MIN = 3.0

_PROBE_JS = r"""
async (sel) => {
  const hex = s => { const h = s.replace('#', ''); const f = h.length === 3
      ? h.split('').map(c => parseInt(c + c, 16)) : [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16));
      return [...f, 1]; };
  const parse = s => { if (!s) return null; s = s.trim();
      if (s.startsWith('#')) return hex(s);
      const m = s.match(/rgba?\(([^)]+)\)/); if (!m) return null;
      const p = m[1].split(/[ ,\/]+/).filter(Boolean).map(Number);
      return [p[0], p[1], p[2], p.length > 3 ? p[3] : 1]; };
  const lum = c => { const f = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
      return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2]); };
  const blend = (fg, bg) => [0, 1, 2].map(i => fg[i] * fg[3] + bg[i] * (1 - fg[3])).concat(1);
  const cr = (a, b) => { const x = lum(a), y = lum(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); };
  const surface = el => { let e = el.parentElement; const st = [];
      while (e) { const c = parse(getComputedStyle(e).backgroundColor);
        if (c && c[3] > 0) st.push(c); if (c && c[3] === 1) break; e = e.parentElement; }
      let b = [0, 0, 0, 1]; for (let i = st.length - 1; i >= 0; i--) b = blend(st[i], b); return b; };
  const mark = el => { const raw = decodeURIComponent(getComputedStyle(el).backgroundImage);
      const m = raw.match(/(?:fill|stroke)='([^']+)'/g); if (!m) return null;
      const v = m.map(x => x.split("'")[1]).find(x => x !== 'none');
      return parse(v.replace(/%28/g, '(').replace(/%29/g, ')')); };
  const el = document.querySelector(sel);
  if (!el) return { missing: sel };
  const orig = el.checked; const out = [];
  for (const theme of ['dark', 'light']) {
    document.documentElement.setAttribute('data-theme', theme);
    for (const checked of [false, true]) {
      el.checked = checked; await new Promise(r => setTimeout(r, 300));
      const cs = getComputedStyle(el); const bg = surface(el);
      const fill = blend(parse(cs.backgroundColor), bg);
      const border = blend(parse(cs.borderTopColor), bg);
      const k = mark(el);
      out.push({ theme, checked,
        // the control must be findable against the card: by its border OR its fill
        boundary: Math.max(cr(border, bg), cr(fill, bg)),
        // knob / tick / dot against the fill it sits on (unchecked checkbox/radio have none)
        mark: k ? cr(blend(k, fill), fill) : null });
    }
  }
  el.checked = orig; document.documentElement.setAttribute('data-theme', 'dark');
  return out;
}
"""


# Real controls as Shiny renders them (2026-11-22: input_checkbox / input_radio_buttons emit
# plain <input>s WITHOUT .form-check-input, which an injected Bootstrap probe could not see).
_CONTROLS = {
    "switch": ("setup", "#simulation_fishing_mortality_enabled"),
    "checkbox": ("calibration", ".checkbox input[type=checkbox]"),
    "radio": ("results", "#trophic_predator_level input[type=radio]"),
}


@pytest.mark.parametrize("kind", list(_CONTROLS))
def test_control_contrast(page: Page, app: ShinyAppProc, kind: str):
    tab, selector = _CONTROLS[kind]
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_T)
    dismiss_changelog_modal(page)
    page.locator(f".nav-pills .nav-link[data-value='{tab}']").click()
    page.wait_for_selector(selector, state="attached", timeout=_T)
    rows = page.evaluate(_PROBE_JS, selector)
    assert isinstance(rows, list), rows
    failures = []
    for r in rows:
        if r["boundary"] < _MIN:
            failures.append(
                f"{kind} {r['theme']} checked={r['checked']}: boundary {r['boundary']:.2f}"
            )
        if r["mark"] is not None and r["mark"] < _MIN:
            failures.append(f"{kind} {r['theme']} checked={r['checked']}: mark {r['mark']:.2f}")
    # the switch knob is drawn in both states; every checked control draws a tick/dot
    assert all(r["mark"] is not None for r in rows if r["checked"] or kind == "switch"), rows
    assert not failures, "\n".join(failures)
