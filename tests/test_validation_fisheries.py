from pathlib import Path

from osmose.validation import fisheries as fz

_FIXTURE = Path(__file__).parent / "fixtures" / "mortalityRate_sample.csv"


def test_read_mortality_recruits_real_csv():
    df = fz.read_mortality_recruits(_FIXTURE)
    assert ("F", "Recruits") in df.columns
    assert ("Mpred", "Recruits") in df.columns
    assert ("Mstarv", "Recruits") in df.columns
    assert ("Madd", "Recruits") in df.columns
    assert len(df) > 0
    assert df[("F", "Recruits")].notna().all()
