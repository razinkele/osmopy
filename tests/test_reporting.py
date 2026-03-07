"""Tests for report generation."""

import numpy as np
import pandas as pd
import pytest

from osmose.reporting import generate_report, summary_table


@pytest.fixture
def mock_results(tmp_path):
    """Create minimal output files for reporting."""
    for sp in ["Anchovy", "Sardine"]:
        df = pd.DataFrame({"time": range(10), "biomass": np.random.rand(10) * 1000})
        df.to_csv(tmp_path / f"osm_biomass_{sp}.csv", index=False)
    df = pd.DataFrame({"time": range(10), "yield": np.random.rand(10) * 100})
    df.to_csv(tmp_path / "osm_yield_Anchovy.csv", index=False)
    return tmp_path


def test_summary_table(mock_results):
    from osmose.results import OsmoseResults

    res = OsmoseResults(mock_results)
    table = summary_table(res)
    assert not table.empty
    assert "species" in table.columns
    assert "biomass_mean" in table.columns


def test_generate_report_html(mock_results, tmp_path):
    from osmose.results import OsmoseResults

    res = OsmoseResults(mock_results)
    config = {"simulation.nspecies": "2", "simulation.time.nyear": "10"}

    output_path = tmp_path / "report.html"
    generate_report(res, config, output_path, fmt="html")
    assert output_path.exists()
    content = output_path.read_text()
    assert "OSMOSE" in content
    assert "Anchovy" in content


def test_generate_report_empty(tmp_path):
    from osmose.results import OsmoseResults

    res = OsmoseResults(tmp_path)  # empty dir
    output_path = tmp_path / "report.html"
    generate_report(res, {}, output_path, fmt="html")
    assert output_path.exists()
