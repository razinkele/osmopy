"""In-process rate limiter for feedback submission (public deployment, D5).

Deliberately in-memory: a single-worker Shiny deploy is the documented target, and a
persistent limiter would be a second consistency problem for no benefit. If the app is ever
run multi-worker, this becomes per-worker and the cap must be divided accordingly — stated
here so the limitation is visible rather than discovered.

Thread safety: `allow()` is an unsynchronised read-modify-write. Concurrent calls can
overshooting the cap; concurrent sweeps can raise KeyError. The target is a single-worker
Shiny deploy whose event loop has no await inside allow() and cannot interleave.
"""

from __future__ import annotations

import logging

from collections import defaultdict

MAX_KEYS = 10_000

_logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, max_per_window: int, window_s: int) -> None:
        self.max_per_window = max_per_window
        self.window_s = window_s
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._last_sweep = float("-inf")
        self._sweep_interval = max(1.0, window_s / 10)
        self._logged_saturation = False

    @property
    def at_capacity(self) -> bool:
        """True if the key table is at MAX_KEYS capacity.

        A caller can distinguish "you've hit your limit" from "the server is full"
        and word an error message accordingly. Refusals at capacity are fail-closed:
        new first-time keys are refused, but established users continue.
        """
        return len(self._hits) >= MAX_KEYS

    def allow(self, key: str, now: float) -> bool:
        cutoff = now - self.window_s
        if now < self._last_sweep:
            self._last_sweep = now
        if now - self._last_sweep >= self._sweep_interval:
            self._sweep_stale(cutoff)
            self._last_sweep = now
        hits = [t for t in self._hits.get(key, []) if t >= cutoff]
        if key not in self._hits and len(self._hits) >= MAX_KEYS:
            if not self._logged_saturation:
                _logger.warning(
                    "Rate limiter table at capacity (%d keys); refusing new keys",
                    MAX_KEYS,
                )
                self._logged_saturation = True
            return False
        if len(hits) >= self.max_per_window:
            self._hits[key] = hits
            return False
        hits.append(now)
        self._hits[key] = hits
        return True

    def _sweep_stale(self, cutoff: float) -> None:
        """Remove stale keys.

        We deliberately do not evict by oldest hit. Eviction would let an attacker
        who has exhausted their own rate limit flood the table to evict their own
        record and reset it, switching the limiter off. Refusing new keys only delays
        first-time submitters during a flood and is self-healing as entries expire.
        """
        stale_keys = [k for k, v in self._hits.items() if not v or v[-1] < cutoff]
        for k in stale_keys:
            del self._hits[k]
