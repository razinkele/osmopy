import plotly.io as pio


def test_osmose_templates_registered():
    from osmose.plotly_theme import ensure_templates

    ensure_templates()
    assert "osmose" in pio.templates
    assert "osmose-light" in pio.templates


def test_osmose_colors_has_8_entries():
    from osmose.plotly_theme import OSMOSE_COLORS

    assert len(OSMOSE_COLORS) == 8


def test_get_plotly_template():
    from osmose.plotly_theme import get_plotly_template

    assert get_plotly_template("dark") == "osmose"
    assert get_plotly_template("light") == "osmose-light"
