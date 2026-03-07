"""About and Help modal dialogs for the OSMOSE header.

These are rendered as static Bootstrap modals in the UI and triggered
client-side via data-bs-toggle, avoiding any server round-trip.
"""

from shiny import ui


def _bs_modal(modal_id: str, title: str, body, *, size: str = "lg"):
    """Build a Bootstrap 5 modal dialog as static HTML."""
    size_class = f"modal-{size}" if size else ""
    return ui.div(
        ui.div(
            ui.div(
                # Header
                ui.div(
                    ui.tags.h5(title, class_="modal-title"),
                    ui.tags.button(
                        type="button",
                        class_="btn-close",
                        **{"data-bs-dismiss": "modal", "aria-label": "Close"},
                    ),
                    class_="modal-header",
                ),
                # Body
                ui.div(body, class_="modal-body"),
                class_="modal-content",
            ),
            class_=f"modal-dialog {size_class} modal-dialog-scrollable",
        ),
        class_="modal fade",
        id=modal_id,
        tabindex="-1",
        **{"aria-labelledby": f"{modal_id}-label", "aria-hidden": "true"},
    )


def about_modal():
    """Return the full About OSMOSE modal dialog."""
    body = ui.TagList(
        ui.div(
            ui.tags.span("v0.1.0", class_="osmose-version-badge"),
            ui.tags.span("Python Interface", class_="osmose-version-label"),
            class_="osmose-about-version",
        ),
        ui.markdown(
            """
**OSMOSE** (Object-oriented Simulator of Marine Ecosystems) is an individual-based
model for exploring marine ecosystem dynamics. This Python interface provides
configuration, execution, calibration, and visualization of OSMOSE simulations.

---

### Tech Stack

| Component | Technology |
|-----------|------------|
| Web Framework | Shiny for Python |
| Plotting | Plotly |
| Calibration | pymoo (NSGA-II) |
| Sensitivity | SALib (Sobol) |
| GP Surrogate | scikit-learn |
| Data | xarray, pandas |
| Simulation | Java (OSMOSE engine) |

---

### Changelog

- **v0.1.0** — Initial release
  - Schema-driven parameter system (181 parameters)
  - Config I/O with OSMOSE CSV/properties format
  - Run page with async Java subprocess management
  - Results viewer with Plotly charts
  - Calibration: NSGA-II, GP surrogate, sensitivity analysis
  - Scenario management: save, load, fork, compare
  - Nautical Observatory theme

---

### License

Released under the **MIT License**.

[View on GitHub](https://github.com/razinkele/osmopy)
"""
        ),
    )
    return _bs_modal("aboutModal", "About OSMOSE", body, size="lg")


def help_modal():
    """Return the full Help / User Guide modal dialog."""
    body = ui.navset_pill(
        ui.nav_panel(
            "Getting Started",
            ui.markdown(
                """
### Welcome to OSMOSE

OSMOSE is an individual-based model for simulating marine ecosystems.
This interface lets you configure, run, and analyze simulations without
touching configuration files directly.

#### Quick Start

1. **Configure** your simulation using the tabs on the left sidebar
2. **Run** the simulation from the Execute section
3. **View results** with interactive charts
4. **Calibrate** parameters using optimization algorithms
5. **Save scenarios** to compare different configurations

#### Requirements

- Java Runtime Environment (JRE 8+) for the OSMOSE engine
- An OSMOSE JAR file (set the path in Setup or Run)
- Input data files (forcing, maps) in a working directory
"""
            ),
        ),
        ui.nav_panel(
            "Configure",
            ui.markdown(
                """
### Configuration Pages

#### Setup
General simulation parameters: simulation name, number of time steps,
number of species, output directory, and JAR path. These are the
foundational settings every simulation needs.

#### Grid & Maps
Define the spatial grid (longitude/latitude bounds, cell size) and
preview the simulation domain on an interactive map. Upload custom
land/sea masks if needed.

#### Forcing
Environmental forcing data: temperature, salinity, plankton fields.
Set file paths, variable names, and time step mappings for each
forcing variable.

#### Fishing
Configure fishing mortality by species. Set fishing rates, seasonal
patterns, and selectivity curves. Supports both constant and
time-varying mortality.

#### Movement
Species movement parameters: diffusion coefficients, habitat
preferences, and migration patterns. Configure independently for
each species in the simulation.
"""
            ),
        ),
        ui.nav_panel(
            "Execute",
            ui.markdown(
                """
### Running Simulations

#### Run Page
- Set the **JAR path** to your OSMOSE engine
- Click **Start** to launch the simulation
- Monitor progress in the live console output
- Use **Stop** to abort a running simulation

Output files are written to the directory specified in Setup.

#### Results Page
After a run completes, view results as interactive Plotly charts:
- **Biomass** time series by species
- **Abundance** trends
- **Size distribution** histograms
- **Spatial maps** with time animation
- **Diet composition** matrices

Use the species selector and time range controls to focus on
specific aspects of the output.
"""
            ),
        ),
        ui.nav_panel(
            "Optimize",
            ui.markdown(
                """
### Calibration & Sensitivity

#### Calibration (NSGA-II)
Multi-objective calibration using the NSGA-II genetic algorithm:
- Select parameters to calibrate and their bounds
- Define objective functions (e.g., match observed biomass)
- Set population size and number of generations
- Monitor convergence with real-time Pareto front plots

#### GP Surrogate
Gaussian Process surrogate model for faster calibration:
- Trains a surrogate on initial evaluations
- Explores the parameter space more efficiently
- Falls back to full simulation for validation

#### Sensitivity Analysis
Sobol sensitivity analysis (SALib):
- Identify which parameters most influence outputs
- First-order and total-order sensitivity indices
- Visualize parameter importance rankings
"""
            ),
        ),
        ui.nav_panel(
            "Manage",
            ui.markdown(
                """
### Scenarios & Advanced

#### Scenarios
Save and manage simulation configurations:
- **Save** the current configuration as a named scenario
- **Load** a saved scenario to restore all parameters
- **Fork** a scenario to create a variant
- **Compare** two scenarios side-by-side
- **Bulk export/import** scenarios as ZIP archives

#### Advanced
- **Import** configuration from CSV or properties files
  with a preview diff before merging
- **Export** the current configuration as CSV
- Direct access to raw parameter values for power users
"""
            ),
        ),
        id="help_tabs",
    )
    return _bs_modal("helpModal", "User Guide", body, size="xl")
