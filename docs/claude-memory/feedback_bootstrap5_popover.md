---
name: Bootstrap 5 popover initialization — head-script timing gotcha
description: Why tooltip initialization lives at end-of-body, not in <head>. Prevents future "why don't my tooltips work?" rabbit holes.
type: feedback
originSessionId: 12d091cb-241b-45e7-82e3-906f4025f88b
---
**Rule:** Bootstrap 5 popovers in this app are initialized by a single end-of-body `<script>` using `setInterval(500ms)` to poll for new `[data-bs-toggle="popover"]` elements. It uses `bootstrap.Popover.getInstance(el)` (BS5 API), NOT `el._bsPopover` (BS4 API).

**Why:**

- Inline `<head>` scripts execute before Bootstrap 5 loads — `setInterval` kicked off in `<head>` fires into undefined `bootstrap` global and does nothing.
- Shiny dynamically renders new DOM (e.g., when switching tabs), so a one-shot `DOMContentLoaded` init misses later-rendered popovers. Polling is intentional.
- BS4's `el._bsPopover` private prop does not exist on BS5 — using it returns `undefined` and silently no-ops.

**How to apply:**

- Never move the popover-init script into `<head>` to "speed it up".
- Never swap polling for `MutationObserver` without testing all tab switches — we picked the simpler, slightly wasteful polling on purpose.
- When debugging "tooltip not showing", check: (a) element has `data-bs-toggle="popover"`, (b) end-of-body script is in the page, (c) browser console for `bootstrap is not defined` timing errors.
