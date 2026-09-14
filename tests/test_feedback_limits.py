import logging

from osmose.feedback_limits import MAX_KEYS, RateLimiter


def test_allows_up_to_the_cap_then_blocks():
    rl = RateLimiter(max_per_window=3, window_s=60)
    assert [rl.allow("ip1", now=0.0) for _ in range(3)] == [True, True, True]
    assert rl.allow("ip1", now=0.0) is False


def test_window_expiry_restores_budget():
    rl = RateLimiter(max_per_window=1, window_s=60)
    assert rl.allow("ip1", now=0.0) is True
    assert rl.allow("ip1", now=59.0) is False
    assert rl.allow("ip1", now=61.0) is True


def test_keys_are_independent():
    rl = RateLimiter(max_per_window=1, window_s=60)
    assert rl.allow("a", now=0.0) is True
    assert rl.allow("b", now=0.0) is True


def test_pruning_bounds_memory():
    rl = RateLimiter(max_per_window=1, window_s=10)
    max_keys = 0
    for i in range(5000):
        rl.allow(f"ip{i}", now=float(i))
        max_keys = max(max_keys, len(rl._hits))
    # Track max over entire run, not endpoint sample
    assert max_keys <= 10_000


def test_burst_same_timestamp_exceeds_limit():
    """Attack: N distinct keys all at the same timestamp must not exceed MAX_KEYS."""
    rl = RateLimiter(max_per_window=1, window_s=60)
    for i in range(50_000):
        rl.allow(f"ip{i}", now=0.0)
    # Without proper eviction, _hits grows unbounded; with it, capped at MAX_KEYS
    assert len(rl._hits) <= 10_000


def test_attacker_self_reset_via_eviction():
    """Attack: attacker exhausts limit, floods table to evict their own key and reset it.

    Eviction-based designs allow this: attacker hits 3/3 requests (refused on 4th),
    then floods with ~10,005 throwaway keys at the same timestamp. The attacker's own
    key becomes the oldest-by-hit (all throwaway keys are fresh), gets evicted first,
    and the next attacker request succeeds. Eviction lets the attacker switch limits off.

    Correct design: never evict. Only refuse new keys when table is full.
    With refusal, the attacker's key remains refused and stays in the table until expired.
    """
    rl = RateLimiter(max_per_window=3, window_s=60)
    # Attacker exhausts their limit
    assert rl.allow("attacker", now=0.0) is True
    assert rl.allow("attacker", now=0.0) is True
    assert rl.allow("attacker", now=0.0) is True
    # Fourth request from attacker is refused
    assert rl.allow("attacker", now=0.0) is False
    # Attacker floods with throwaway keys to fill the table
    for i in range(12_000):
        rl.allow(f"throwaway_{i}", now=0.0)
    # With eviction, attacker's key would be evicted (oldest hit). With refusal only,
    # the attacker key stays refused. Assert it is still refused.
    assert rl.allow("attacker", now=0.0) is False


def test_existing_key_works_when_table_full():
    """Fail-closed: only refuse NEW keys, not existing ones.

    An established requester who is still within the window should continue to work
    even when the table is full and new first-time requesters are refused.
    """
    rl = RateLimiter(max_per_window=2, window_s=60)
    # Existing user makes 1 request (stays within budget)
    assert rl.allow("existing_user", now=0.0) is True
    # Fill the table with other keys
    for i in range(12_000):
        rl.allow(f"new_ip_{i}", now=0.0)
    # Existing user should still be able to make another request (second of 2)
    assert rl.allow("existing_user", now=0.0) is True
    # But a new key should be refused (table full)
    assert rl.allow("brand_new_ip", now=0.0) is False


def test_backward_clock_step_reenables_sweep():
    """FIX 1: a backward wall-clock step must not stall the sweep permanently.

    The sweep gate is ``now - self._last_sweep >= self._sweep_interval``. If the
    clock steps backward (NTP correction, VM restore, manual set), that difference
    turns negative and never recovers until the clock catches up. Since the limiter
    never evicts, the sweep is the ONLY path by which a full table becomes non-full,
    so the table stays saturated and every first-time key is refused.

    The discriminator is that user-visible refusal, not a bound: the ``<= MAX_KEYS``
    ceiling holds either way because refusal at capacity is fail-closed.

    Four beats, and all four are needed:
      1. a call at a large ``now`` anchors ``_last_sweep`` high;
      2. the clock steps BACKWARD and the table is filled to capacity -- the FIX-1
         guard only RE-ARMS the gate here (delta becomes 0), it does not sweep, so
         guarded and unguarded behave identically through this beat;
      3. time advances by more than ``_sweep_interval`` past the backward point
         while staying below the old forward anchor;
      4. only the guarded limiter sweeps at beat 3/4 and accepts a newcomer.

    The expected final size is 2, not 1: the beat-1 anchor was recorded under the
    pre-step clock, so its timestamp sits in the future of every backward-clock
    cutoff and can never go stale. What is reclaimed is the 9_999 flood keys.
    """
    rl = RateLimiter(max_per_window=1, window_s=60)
    assert rl._sweep_interval == 6.0

    # Beat 1: anchor _last_sweep at a large forward time.
    assert rl.allow("anchor", now=1_000_000.0) is True

    # Beat 2: clock steps backward; fill the table to exactly MAX_KEYS.
    for i in range(MAX_KEYS - 1):
        rl.allow(f"flood{i}", now=100.0)
    assert len(rl._hits) == MAX_KEYS

    # Beats 3-4: >= _sweep_interval past the backward point, still far below the
    # forward anchor, and past the 60 s window so every flood key is stale.
    # Guarded: sweep reclaims the floods and the newcomer is accepted.
    # Unguarded: 200.0 - 1_000_000.0 is negative, no sweep ever runs again, the
    # table is still full, and the newcomer is refused.
    assert rl.allow("newcomer", now=200.0) is True
    assert len(rl._hits) == 2


def test_at_capacity_property():
    """FIX 2b: at_capacity property exposes table fullness for caller."""
    rl = RateLimiter(max_per_window=1, window_s=60)
    assert rl.at_capacity is False
    for i in range(10_000):
        rl.allow(f"ip{i}", now=0.0)
    assert rl.at_capacity is True
    assert len(rl._hits) == 10_000


def test_saturation_warning_logged_once(caplog):
    """FIX 2c: exactly one operator-visible WARNING when the table saturates.

    Asserts on real log records, not on the private ``_logged_saturation`` flag:
    the flag is bookkeeping, the WARNING is the feature. Deleting the
    ``_logger.warning(...)`` call while keeping the flag must turn this red.
    """
    logger_name = "osmose.feedback_limits"
    rl = RateLimiter(max_per_window=1, window_s=60)
    with caplog.at_level(logging.WARNING, logger=logger_name):
        for i in range(MAX_KEYS):
            assert rl.allow(f"ip{i}", now=0.0) is True
        # Filling to exactly MAX_KEYS refuses nobody, so nothing is logged yet.
        caplog.clear()
        for i in range(20):
            assert rl.allow(f"newcomer{i}", now=0.0) is False

    warnings = [r for r in caplog.records if r.name == logger_name and r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert str(MAX_KEYS) in message, message


def test_sweep_reclaims_expired_keys():
    """FIX 3: Sweep must actually reclaim expired keys, not be a no-op.

    This test will fail if _sweep_stale is replaced with pass.
    """
    rl = RateLimiter(max_per_window=1, window_s=60)
    for i in range(5_000):
        rl.allow(f"ip{i}", now=0.0)
    assert len(rl._hits) == 5_000
    rl.allow("later", now=1_000.0)
    assert len(rl._hits) == 1
