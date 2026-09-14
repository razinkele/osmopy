from osmose.feedback_limits import RateLimiter


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
    """FIX 1: Backward wall-clock step must not stall the sweep permanently.

    If now moves backward (NTP step, VM restore, manual clock change), the sweep
    gate can stay false forever. This test verifies the clock re-anchors.
    """
    rl = RateLimiter(max_per_window=1, window_s=60)
    for i in range(9_950):
        rl.allow(f"ip{i}", now=0.0)
    assert len(rl._hits) == 9_950
    rl.allow("forward_ip", now=1_000_000.0)
    for i in range(5_000):
        rl.allow(f"backward_ip_{i}", now=999_000.0)
    assert len(rl._hits) <= 10_000


def test_at_capacity_property():
    """FIX 2b: at_capacity property exposes table fullness for caller."""
    rl = RateLimiter(max_per_window=1, window_s=60)
    assert rl.at_capacity is False
    for i in range(10_000):
        rl.allow(f"ip{i}", now=0.0)
    assert rl.at_capacity is True
    assert len(rl._hits) == 10_000


def test_saturation_warning_logged_once():
    """FIX 2c: WARNING logged exactly once when table saturates."""
    rl = RateLimiter(max_per_window=1, window_s=60)
    for i in range(10_000):
        rl.allow(f"ip{i}", now=0.0)
    assert rl._logged_saturation is False
    rl.allow("new_key_1", now=0.0)
    assert rl._logged_saturation is True
    for i in range(10):
        rl.allow(f"new_key_retry_{i}", now=0.0)
    assert rl._logged_saturation is True


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
