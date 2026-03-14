"""Generate HTML reports from OSMOSE simulation results."""

from __future__ import annotations

import html as html_mod
from pathlib import Path

import jinja2
import pandas as pd

from osmose.results import OsmoseResults


def report_summary_table(results: OsmoseResults) -> pd.DataFrame:
    """Create a summary table with mean/std per species per output type."""
    rows: list[dict] = []
    bio = results.biomass()
    if not bio.empty and "species" in bio.columns:
        for sp, group in bio.groupby("species"):
            rows.append(
                {
                    "species": sp,
                    "biomass_mean": (
                        group["biomass"].mean() if "biomass" in group.columns else None
                    ),
                    "biomass_std": (group["biomass"].std() if "biomass" in group.columns else None),
                }
            )

    yld = results.yield_biomass()
    if not yld.empty and "species" in yld.columns:
        for sp, group in yld.groupby("species"):
            matching = [r for r in rows if r["species"] == sp]
            if matching:
                matching[0]["yield_mean"] = (
                    group["yield"].mean() if "yield" in group.columns else None
                )
            else:
                rows.append(
                    {
                        "species": sp,
                        "yield_mean": (group["yield"].mean() if "yield" in group.columns else None),
                    }
                )

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def generate_report(
    results: OsmoseResults,
    config: dict[str, str],
    output_path: Path,
    fmt: str = "html",
    template_path: Path | None = None,
) -> Path:
    """Generate a report from OSMOSE results.

    Args:
        results: OsmoseResults instance.
        config: Flat config dict.
        output_path: Where to write the report.
        fmt: "html" (default).
        template_path: Optional custom Jinja2 template file. If None, uses the bundled template.

    Returns:
        Path to the generated report file.
    """
    output_path = Path(output_path)

    if fmt != "html":
        raise NotImplementedError(f"Report format '{fmt}' is not supported. Use 'html'.")

    # Build data
    table = report_summary_table(results)
    summary_html = (
        table.to_html(index=False, classes="table", escape=True) if not table.empty else "<p>No data</p>"
    )

    metadata = {
        "nspecies": html_mod.escape(str(config.get("simulation.nspecies", "?"))),
        "nyear": html_mod.escape(str(config.get("simulation.time.nyear", "?"))),
    }

    species_details = []
    bio = results.biomass()
    if not bio.empty and "species" in bio.columns:
        for sp in sorted(bio["species"].unique()):
            sp_data = bio[bio["species"] == sp]
            if "biomass" in sp_data.columns:
                species_details.append(
                    {
                        "name": sp,
                        "mean_biomass": sp_data["biomass"].mean(),
                    }
                )

    # Render template
    if template_path:
        loader: jinja2.BaseLoader = jinja2.FileSystemLoader(template_path.parent)
        template_name = template_path.name
    else:
        loader = jinja2.PackageLoader("osmose", "templates")
        template_name = "report.html"

    env = jinja2.Environment(loader=loader, autoescape=True)
    template = env.get_template(template_name)
    html = template.render(
        metadata=metadata,
        summary_html=summary_html,
        species_details=species_details,
    )

    output_path.write_text(html)
    return output_path
