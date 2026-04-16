"""WebGL / CPU renderer badge for deck.gl map containers."""

from shiny import ui


def renderer_badge() -> ui.Tag:
    """Return a small overlay badge that shows 'WebGL' or 'CPU' rendering mode.

    Place inside an ``.osm-grid-map-container`` div.  A one-shot JS snippet
    (included automatically) detects WebGL support and updates all badges.
    """
    return ui.div(
        ui.tags.span(class_="osm-badge-dot"),
        ui.tags.span("detecting…", class_="osm-badge-label"),
        class_="osm-renderer-badge",
    )


# One-shot <script> that detects WebGL and updates every badge on the page.
# Safe to include multiple times — the guard prevents duplicate execution.
_DETECT_SCRIPT = ui.tags.script("""
(function() {
    if (window.__osmRendererDetected) return;
    window.__osmRendererDetected = true;
    function detect() {
        try {
            var c = document.createElement('canvas');
            var gl = c.getContext('webgl2') || c.getContext('webgl')
                  || c.getContext('experimental-webgl');
            return !!gl;
        } catch(e) { return false; }
    }
    var webgl = detect();
    function tag() {
        document.querySelectorAll('.osm-renderer-badge').forEach(function(el) {
            var label = el.querySelector('.osm-badge-label');
            var want = webgl ? 'WebGL' : 'CPU';
            var cls  = webgl ? 'webgl' : 'cpu';
            var drop = webgl ? 'cpu'   : 'webgl';
            el.classList.add(cls);
            el.classList.remove(drop);
            if (label && label.textContent !== want) label.textContent = want;
        });
    }
    // Tag existing badges and watch for dynamically added ones
    if (document.readyState === 'complete' || document.readyState === 'interactive') {
        tag();
    }
    document.addEventListener('DOMContentLoaded', tag);
    new MutationObserver(function() { tag(); }).observe(
        document.body, {childList: true, subtree: true}
    );
})();
""")


def renderer_badge_script() -> ui.Tag:
    """Return the detection script.  Include once in head or body."""
    return _DETECT_SCRIPT
