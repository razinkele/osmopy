"""Tests for BiomassTarget data model and CSV loader."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from osmose.calibration.targets import BiomassTarget, load_targets


class TestBiomassTarget:
    def test_required_fields(self) -> None:
        t = BiomassTarget(species="cod", target=120000, lower=60000, upper=250000)
        assert t.species == "cod"
        assert t.target == 120000
        assert t.lower == 60000
        assert t.upper == 250000

    def test_defaults(self) -> None:
        t = BiomassTarget(species="cod", target=1, lower=0, upper=2)
        assert t.weight == 1.0
        assert t.reference_point_type == "biomass"
        assert t.source == ""
        assert t.notes == ""

    def test_all_fields(self) -> None:
        t = BiomassTarget(
            species="cod",
            target=120000,
            lower=60000,
            upper=250000,
            weight=0.5,
            reference_point_type="ssb",
            source="ICES",
            notes="test",
        )
        assert t.reference_point_type == "ssb"
        assert t.source == "ICES"


class TestLoadTargets:
    def test_basic_csv(self, tmp_path: Path) -> None:
        csv = tmp_path / "targets.csv"
        csv.write_text(
            textwrap.dedent("""\
            species,target_tonnes,lower_tonnes,upper_tonnes,weight,reference_point_type,source,notes
            cod,120000,60000,250000,1.0,ssb,ICES,collapse
            herring,1500000,800000,3000000,1.0,biomass,ICES,complex
        """)
        )
        targets, metadata = load_targets(csv)
        assert len(targets) == 2
        assert targets[0].species == "cod"
        assert targets[0].target == 120000
        assert targets[0].reference_point_type == "ssb"
        assert targets[1].species == "herring"
        assert metadata == {}

    def test_backward_compat_no_reference_point_type(self, tmp_path: Path) -> None:
        """Old-format CSV without reference_point_type column defaults to 'biomass'."""
        csv = tmp_path / "targets.csv"
        csv.write_text(
            textwrap.dedent("""\
            species,target_tonnes,lower_tonnes,upper_tonnes,weight
            cod,120000,60000,250000,1.0
        """)
        )
        targets, metadata = load_targets(csv)
        assert len(targets) == 1
        assert targets[0].reference_point_type == "biomass"
        assert targets[0].source == ""
        assert targets[0].notes == ""

    def test_comment_lines_skipped(self, tmp_path: Path) -> None:
        csv = tmp_path / "targets.csv"
        csv.write_text(
            textwrap.dedent("""\
            # This is a comment
            # Another comment
            species,target_tonnes,lower_tonnes,upper_tonnes,weight
            cod,120000,60000,250000,1.0
        """)
        )
        targets, _ = load_targets(csv)
        assert len(targets) == 1

    def test_metadata_lines(self, tmp_path: Path) -> None:
        csv = tmp_path / "targets.csv"
        csv.write_text(
            textwrap.dedent("""\
            #! version: 1.0
            #! last_updated: 2026-04-15
            # Human comment
            species,target_tonnes,lower_tonnes,upper_tonnes,weight
            cod,120000,60000,250000,1.0
        """)
        )
        targets, metadata = load_targets(csv)
        assert len(targets) == 1
        assert metadata["version"] == "1.0"
        assert metadata["last_updated"] == "2026-04-15"

    def test_malformed_metadata_line_ignored(self, tmp_path: Path) -> None:
        """#! line without a colon is silently ignored."""
        csv = tmp_path / "targets.csv"
        csv.write_text(
            textwrap.dedent("""\
            #! no-colon-here
            species,target_tonnes,lower_tonnes,upper_tonnes,weight
            cod,120000,60000,250000,1.0
        """)
        )
        targets, metadata = load_targets(csv)
        assert len(targets) == 1
        assert metadata == {}

    def test_loads_real_baltic_csv(self) -> None:
        """Smoke test: loads the actual data file."""
        csv = Path("data/baltic/reference/biomass_targets.csv")
        if not csv.exists():
            pytest.skip("Baltic targets CSV not found")
        targets, _ = load_targets(csv)
        assert len(targets) >= 8
        species = [t.species for t in targets]
        assert "cod" in species
        assert "herring" in species
