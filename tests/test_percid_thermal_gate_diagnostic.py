import subprocess
import sys


def test_diagnostic_runs_and_reports_both_percids():
    # --nyear 4 keeps the smoke test cheap (two short Baltic runs); the real A/B
    # is run by hand with the full horizon.
    out = subprocess.run(
        [sys.executable, "scripts/baltic_percid_thermal_gate_diagnostic.py", "--nyear", "4"],
        capture_output=True, text=True, timeout=900,
    )
    assert out.returncode == 0, out.stderr
    lo = out.stdout.lower()
    assert "perch" in lo and "pikeperch" in lo
    assert "overshoot" in lo
    # finding 2: the absolute-biomass axis must be reported (not only boom/bust)
    assert "mean_off" in lo and "mean_on" in lo
