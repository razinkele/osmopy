"""In-process rate limiter for feedback submission (public deployment, D5).

Deliberately in-memory: a single-worker Shiny deploy is the documented target, and a
persistent limiter would be a second consistency problem for no benefit. If the app is ever
run multi-worker, this becomes per-worker and the cap must be divided accordingly — stated
here so the limitation is visible rather than discovered.
"""

from __future__ import annotations

from collections import defaultdict


class RateLimiter:
    def __init__(self, max_per_window: int, window_s: int) -> None:
        self.max_per_window = max_per_window
        self.window_s = window_s
        self._hits: dict[str, list[float]] = defaultdict(list)

    def allow(self, key: str, now: float) -> bool:
        cutoff = now - self.window_s
        if len(self._hits) > 1000:  # prune globally before it can grow without bound
            for k in [k for k, v in self._hits.items() if not v or v[-1] < cutoff]:
                del self._hits[k]
        hits = [t for t in self._hits[key] if t >= cutoff]
        if len(hits) >= self.max_per_window:
            self._hits[key] = hits
            return False
        hits.append(now)
        self._hits[key] = hits
        return True
