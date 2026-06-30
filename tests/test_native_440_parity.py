def test_run_outputs_deterministic():
    from scripts.native_440_parity import max_rel_diff, run_outputs

    a = run_outputs("data/minimal", years=1)
    b = run_outputs("data/minimal", years=1)
    assert a, "run_outputs returned no metrics"
    # fixed seed -> the Python engine is bit-reproducible run-to-run
    assert max(max_rel_diff(a[k], b[k]) for k in a) == 0.0
