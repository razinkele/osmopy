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
    for i in range(5000):
        rl.allow(f"ip{i}", now=float(i))
    assert len(rl._hits) < 100  # old keys pruned, not retained forever
