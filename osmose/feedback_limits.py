"""In-process rate limiter for feedback submission (public deployment, D5).

Deliberately in-memory: a single-worker Shiny deploy is the documented target, and a
persistent limiter would be a second consistency problem for no benefit. If the app is ever
run multi-worker, this becomes per-worker and the cap must be divided accordingly — stated
here so the limitation is visible rather than discovered.

Thread safety: `allow()` is an unsynchronised read-modify-write. Concurrent calls can
overshoot the cap, and a concurrent sweep can raise either KeyError (a key deleted twice)
or RuntimeError: dictionary changed size during iteration (the table mutated while the
sweep scans it). The target is a single-worker Shiny deploy whose event loop has no await
inside allow() and cannot interleave.
"""

from __future__ import annotations

from osmose.logging import setup_logging

MAX_KEYS = 10_000

_log = setup_logging("osmose.feedback_limits")


class RateLimiter:
    def __init__(self, max_per_window: int, window_s: int) -> None:
        # A limiter configured with max_per_window=0 refuses every request AND stores an
        # empty-list slot for each refused key, turning a misconfiguration into a leak
        # against MAX_KEYS. Reject it at construction rather than at 3am.
        if max_per_window < 1:
            raise ValueError(f"max_per_window must be >= 1, got {max_per_window!r}")
        if window_s <= 0:
            raise ValueError(f"window_s must be > 0, got {window_s!r}")
        self.max_per_window = max_per_window
        self.window_s = window_s
        # A plain dict, deliberately NOT defaultdict(list): under a defaultdict any bare
        # `self._hits[k]` read inserts a phantom empty slot that counts toward MAX_KEYS,
        # quietly eroding the one bound this class exists to hold. Every access below is
        # explicit -- `.get` to read, assignment to write, `del` to remove.
        self._hits: dict[str, list[float]] = {}
        self._last_sweep = float("-inf")
        self._sweep_interval = max(1.0, window_s / 10)
        self._logged_saturation = False

    @property
    def at_capacity(self) -> bool:
        """Snapshot of whether the key table is at MAX_KEYS. Performs no sweep.

        Read after a refusal it narrows the cause, but only in one direction:

        - `at_capacity` False proves the refusal was the per-key cap -- that caller
          really has sent too many.
        - `at_capacity` True is AMBIGUOUS and must not be reported as saturation. An
          established key that is over its own per-key cap while the table happens to
          be full reads `allow() -> False` and `at_capacity -> True` at the same time,
          so `if not allowed and rl.at_capacity` mis-attributes an ordinary rate-limit
          hit to the server being full. Word the message to cover both causes rather
          than naming one.

        Being a snapshot it can also go stale in the other direction: a sweep during a
        later `allow()` may drop the table below MAX_KEYS with no call here.

        Refusals at capacity are fail-closed: new first-time keys are refused, but
        established users continue.
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
                _log.warning(
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
        """Remove stale keys, and re-arm the saturation warning if that drained the table.

        We deliberately do not evict by oldest hit. Eviction would let an attacker
        who has exhausted their own rate limit flood the table to evict their own
        record and reset it, switching the limiter off. Refusing new keys only delays
        first-time submitters during a flood and is self-healing as entries expire.
        """
        stale_keys = [k for k, v in self._hits.items() if not v or v[-1] < cutoff]
        for k in stale_keys:
            del self._hits[k]
        if self._logged_saturation and len(self._hits) < MAX_KEYS:
            # Re-arm the one-shot warning now that the table has drained. Without this a
            # SECOND saturation episode is completely silent, which defeats the purpose:
            # what an operator needs to see is that saturation RECURS, not just that it
            # happened once since process start.
            self._logged_saturation = False
