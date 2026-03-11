"""Shared Plotly theme for OSMOSE charts."""

from osmose.plotly_theme import (
    PLOTLY_TEMPLATE,
    PLOTLY_TEMPLATE_LIGHT,
    ensure_templates,
    get_plotly_template,
)

# Register templates on import
ensure_templates()

__all__ = [
    "PLOTLY_TEMPLATE",
    "PLOTLY_TEMPLATE_LIGHT",
    "ensure_templates",
    "get_plotly_template",
]
