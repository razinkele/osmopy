# Loader: "Model selection" rename + per-model info modal — design

> Status: design (awaiting review) · 2026-06-29
> On the Grid/Domain page's example loader: (1) rename the select to "Model selection" and its
> placeholder to "— Select model —"; (2) add an ℹ info button next to **Load** that opens a modal
> listing all available models with their key facts. Backed by a small model-info registry in
> `osmose/demo.py` (also used to tidy the dropdown labels). UI/library only — no engine change.

## 1. Problem / goal

The loader (`ui/pages/grid.py:104-112`) labels the demo picker "Example configuration" with a
"— Select example —" placeholder and auto-titled labels ("Eec", "Bay Of Biscay"). The user wants it
reframed as **model selection**, and wants an **info button** beside Load that explains what each
available model is (region, size, engine compatibility) so a user can choose before loading.

## 2. Approach

### 2.1 Model-info registry — `osmose/demo.py`
Add a `DEMO_INFO: dict[str, dict[str, str]]` keyed by the `list_demos()` names
(`baltic`, `bay_of_biscay`, `eec`, `eec_full`, `minimal`), each with the fields:
`title`, `region`, `species`, `resources`, `engine`, `summary`. Grounded values (verified against the
bundled configs):

| key | title | region | species | resources | engine | summary |
|-----|-------|--------|---------|-----------|--------|---------|
| `bay_of_biscay` | Bay of Biscay | NE Atlantic (Bay of Biscay) | 8 focal | 6 LTL/plankton | Java + Python | The OSMOSE reference example (anchovy, sardine, hake, …); runs on both engines. |
| `eec` | Eastern English Channel | English Channel (reduced) | 6 focal | none | Java + Python | A reduced Eastern English Channel config; quick to run. |
| `eec_full` | Eastern English Channel (full) | English Channel | 14 focal | 10 LTL + 1 background | Python only | The full 14-species EEC — the cross-engine parity benchmark; uses background species (Python engine only). |
| `baltic` | Baltic Sea | Central/Eastern Baltic | 8 focal | 6 LTL + 2 background | Python only | Cod, herring, sprat, flounder, perch, pike-perch, smelt, stickleback; uses background species + LTL forcing (Python engine only). |
| `minimal` | Minimal | — (toy) | 2 focal | none | Java + Python | A 2-species toy configuration for quick tests and smoke runs. |

- Add an accessor `demo_info(name: str) -> dict | None` (returns the entry or `None`) — small, keeps
  callers from importing the dict directly if they prefer.
- "Engine only" reason: `baltic`/`eec_full` declare `simulation.nbackground > 0`, which the Java
  engine cannot load (existing project rule), hence "Python only".

### 2.2 Rename + registry-driven labels — `ui/pages/grid.py`
- `demo_choices` becomes `{"": "— Select model —", **{d: DEMO_INFO[d]["title"] for d in list_demos()}}`
  (registry titles replace the current `d.replace("_"," ").title()` — so "Eec" → "Eastern English
  Channel", etc.). Fall back to the auto-title if a key is somehow missing from `DEMO_INFO` (defensive).
- `ui.input_select("load_example", "Model selection", choices=demo_choices, selected="")` — label
  changed from "Example configuration".

### 2.3 Info button + modal — `ui/pages/grid.py`
- Layout: the current `layout_columns(select, Load, col_widths=[8, 4])` becomes
  `layout_columns(select, info_button, Load, col_widths=[7, 2, 3])` — select · ℹ · Load. The info
  button is `ui.input_action_button("btn_example_info", "ℹ Info", class_="btn-outline-secondary w-100")`,
  wrapped in the same flex/bottom-aligned `div` style as the existing Load button.
- A pure modal-builder helper `_model_info_modal()` (module-level in `grid.py`) builds the modal body
  from `DEMO_INFO`: a titled block per model — `title` (heading) + a one-line facts row
  (`region · species · resources · engine`) + the `summary`. Returns
  `ui.modal(body, title="Available models", easy_close=True, size="l", footer=ui.modal_button("Close"))`.
- Server handler: `@reactive.effect @reactive.event(input.btn_example_info)` → `ui.modal_show(_model_info_modal())`.
  (Follows the existing `ui.modal_show(ui.modal(...))` pattern at `scenarios.py:165`.)

## 3. Data flow

`list_demos()` + `DEMO_INFO` → the dropdown labels (titles) and the info modal. Clicking `btn_example_info`
fires the reactive event → `_model_info_modal()` renders all models from `DEMO_INFO` → `ui.modal_show`.
The Load flow (`handle_load_example`, `grid.py:812`) is unchanged.

## 4. Edge cases

- **`DEMO_INFO` missing a `list_demos()` key** (drift): the dropdown falls back to the auto-title for
  that key, and the modal-builder skips/keeps it gracefully. A unit test asserts the registry covers
  every `list_demos()` entry, so this stays caught.
- **Modal opened before selecting** — fine; it lists all models regardless of the dropdown state.
- **Long species lists** — `species` is a short count + a few headline names, not the full list, so
  the modal stays compact.

## 5. Testing

- **Unit (`tests/`):** `DEMO_INFO` has an entry for every `list_demos()` name, and each entry has all
  six fields non-empty (registry-completeness). `demo_info("baltic")["engine"]` == "Python only";
  `demo_info("bay_of_biscay")["engine"]` startswith "Java"; `demo_info("unknown")` is None.
- **Unit (`ui`):** `_model_info_modal()` returns a `ui.modal` whose rendered text contains every model
  title + the words "Python only" and "Java" (so all models + the engine facts are present). The
  dropdown-choice builder maps `eec` → "Eastern English Channel".
- **Wiring:** `import app`; an e2e click on `#btn_example_info` opens a modal containing "Available
  models" (extend the grid e2e if present, else a smoke assertion).
- No engine-dynamics change — UI + a static metadata dict only.

## 6. Out of scope

- Per-model thumbnails/maps in the modal (text facts only).
- Editing/adding models from the UI (the registry is code-defined).
- Changing the Load behaviour or the demo-generation functions.
