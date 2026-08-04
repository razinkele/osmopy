"""The certifier separates well-assessed targets from low-confidence ones (2026-08-04).

ICES does not assess Baltic pikeperch, perch, smelt or stickleback;
``data/baltic/reference/biomass_targets.csv`` sources them as literature estimates at weight
<= 0.3 and notes that the coarse grid under-resolves species concentrated in estuaries and
lagoons. Scoring them pass/fail alongside category-1 analytical assessments made the headline
verdict a statement about the weakest targets rather than about the model: the two species that
failed the old 7/9 were exactly the two lowest-weight rows in the file.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import baltic_stability_certify as cert  # noqa: E402

TARGETS = (
    Path(__file__).resolve().parents[1] / "data" / "baltic" / "reference" / "biomass_targets.csv"
)


def _rows():
    with open(TARGETS) as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def test_tiers_partition_focal_by_weight():
    assert set(cert.ASSESSED) | set(cert.INDICATIVE) == set(cert.FOCAL)
    assert not set(cert.ASSESSED) & set(cert.INDICATIVE)
    for sp in cert.ASSESSED:
        assert cert.TARGET_WEIGHT[sp] > cert.INDICATIVE_MAX_WEIGHT
    for sp in cert.INDICATIVE:
        assert cert.TARGET_WEIGHT[sp] <= cert.INDICATIVE_MAX_WEIGHT


def test_the_unassessed_species_are_the_indicative_ones():
    """ICES assesses none of these four; they must never drive the headline verdict."""
    assert set(cert.INDICATIVE) == {"perch", "pikeperch", "smelt", "stickleback"}
    assert set(cert.ASSESSED) == {"cod_west", "cod_east", "herring", "sprat", "flounder"}


def test_weights_come_from_the_stock_rows_not_the_catch_rows():
    """The file carries a parallel set of ``catch`` rows with different weights AND bounds.

    Reading those by mistake would silently swap the envelope for five species — e.g. herring's
    catch row is 209,803-405,326 t against its biomass row's 800,000-3,000,000 t — and would also
    reassign every weight to 0.5, dissolving the tier split entirely.
    """
    catch = {r["species"]: r for r in _rows() if r["reference_point_type"] == "catch"}
    assert catch, "fixture assumption: the file still has catch rows to be confused with"
    for sp, row in catch.items():
        if sp in cert.ENVELOPE:
            assert (float(row["lower_tonnes"]), float(row["upper_tonnes"])) != tuple(
                float(x) for x in cert.ENVELOPE[sp]
            ), (
                f"{sp}: catch bounds coincide with the stock envelope; this test can no longer detect the swap"
            )
    # herring is the clearest case: 0.5 in the catch rows, 1.0 in the biomass row.
    assert cert.TARGET_WEIGHT["herring"] == 1.0


def test_envelope_drift_against_the_source_file_raises():
    """A silent divergence would invalidate comparison with every prior certification."""
    original = cert.ENVELOPE["pikeperch"]
    cert.ENVELOPE["pikeperch"] = (original[0], original[1] + 1000)
    try:
        with pytest.raises(ValueError, match="Reconcile before certifying"):
            cert._load_target_weights()
    finally:
        cert.ENVELOPE["pikeperch"] = original


def test_headline_counts_only_assessed_species(capsys):
    """The old baseline read 7/9; both failures were indicative, so the headline is 5/5."""
    table = {
        sp: {
            "persists": True,
            "in_envelope": sp not in ("pikeperch", "smelt"),
            "min_biomass": 1.0e4,
            "late_mean_range": [1.0, 2.0],
        }
        for sp in cert.FOCAL
    }
    headline = cert._print_table("Python", table)
    out = capsys.readouterr().out

    assert headline == len(cert.ASSESSED) == 5
    assert "ASSESSED 5/5" in out
    assert "indicative 2/4" in out
    assert "7/9" in out, "the legacy all-species figure must stay visible for continuity"
