"""In-process rate limiter for feedback submission (public deployment, D5).

Deliberately in-memory: a single-worker Shiny deploy is the documented target, and a
persistent limiter would be a second consistency problem for no benefit. If the app is ever
run multi-worker, this becomes per-worker and the cap must be divided accordingly — stated
here so the limitation is visible rather than discovered.
"""

from __future__ import annotations

from collections import defaultdict

MAX_KEYS = 10_000


class RateLimiter:
    def __init__(self, max_per_window: int, window_s: int) -> None:
        self.max_per_window = max_per_window
        self.window_s = window_s
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._needs_sweep = False

    def allow(self, key: str, now: float) -> bool:
        cutoff = now - self.window_s
        if self._needs_sweep and len(self._hits) > 1000:
            self._sweep(cutoff)
        hits = [t for t in self._hits[key] if t >= cutoff]
        if len(hits) >= self.max_per_window:
            self._hits[key] = hits
            return False
        hits.append(now)
        self._hits[key] = hits
        if len(self._hits) > MAX_KEYS:
            self._needs_sweep = True
            self._sweep(cutoff)
        return True

    def _sweep(self, cutoff: float) -> None:
        """Remove stale keys, then evict oldest if still over MAX_KEYS.

        Evicting by oldest hit (rather than refusing requests) loosens the victim's own
        rate limit and never tightens it. Refusing new keys would instead lock out
        legitimate users behind an attacker-filled table—a denial of service against
        the innocent, which this design rejects.
        """
        stale_keys = [k for k, v in self._hits.items() if not v or v[-1] < cutoff]
        for k in stale_keys:
            del self._hits[k]
        if len(self._hits) > MAX_KEYS:
            oldest_key = min(self._hits.keys(), key=lambda k: self._hits[k][-1])
            del self._hits[oldest_key]
        self._needs_sweep = False
