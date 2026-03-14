"""About and Help modal dialogs for the OSMOSE header.

These are rendered as static Bootstrap modals in the UI and triggered
client-side via data-bs-toggle, avoiding any server round-trip.
"""

from shiny import ui

from osmose import __version__


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
            ui.tags.span(f"v{__version__}", class_="osmose-version-badge"),
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
            "OSMOSE Model",
            ui.markdown(
                """
### What is OSMOSE?

**OSMOSE** (Object-oriented Simulator of Marine Ecosystems) is an
individual-based model (IBM) for simulating multispecies marine
ecosystems. It was developed at IRD (Institut de Recherche pour le
Developpement) and is designed to explore the combined effects of
fishing, climate change, and species interactions on fish communities.

Unlike traditional stock-assessment models that treat each species
independently, OSMOSE explicitly models **predator-prey interactions**
among all species simultaneously, creating emergent food-web dynamics.

---

### Core Principle: Size-Based Opportunistic Predation

OSMOSE is built on a single unifying hypothesis: **predation is an
opportunistic, size-structured process**. A fish can eat any other
organism that:

1. **Co-occurs spatially** — predator and prey occupy the same grid cell
2. **Fits within the size window** — the prey body size falls between
   the predator's minimum and maximum predator/prey size ratios

There are no fixed predator-prey links. The food web **emerges** from
the spatial overlap and relative body sizes of all organisms in the
system. This makes OSMOSE fundamentally different from models with
pre-defined diet matrices.

---

### Species Categories

OSMOSE distinguishes three types of organisms:

#### Focal Species (HTL — High Trophic Level)
The main modelled fish and invertebrate species. Each focal species
is represented as a population of **schools** — groups of individuals
sharing the same age, size, and location. Schools undergo the full
life cycle: growth, reproduction, predation, natural mortality,
starvation, fishing, and movement. Typically 5-15 focal species per
application.

#### Resource Groups (LTL — Low Trophic Level)
Plankton and lower trophic level organisms that serve as prey for
focal species but are not explicitly modelled as individuals. Their
biomass is provided as **external forcing** (typically from a
biogeochemical model such as ROMS-PISCES or NEMO-PISCES) in the form
of spatiotemporal NetCDF fields. Resource groups have a fixed size
range and trophic level, and their accessibility to fish predators is
controlled by a coefficient (0-1).

#### Background Species
Species that interact with focal species through predation but are
not the primary focus of the study. They have simplified dynamics
compared to focal species — typically constant biomass distributions
read from CSV maps. Background species fill ecological niches that
would otherwise be empty in a model with only a few focal species.

---

### Model Layers & Processes

OSMOSE simulates the following processes at each time step, applied
sequentially to every school in the system:

#### 1. Spatial Distribution & Movement
Each species has a set of **distribution maps** (CSV grids) defining
where individuals can be found. Maps vary by:
- **Season** — different maps for different time steps within a year
- **Age/size class** — juveniles may occupy different habitats than
  adults (e.g., nursery grounds vs. spawning grounds)
- **Year** — maps can change over simulation years

At each time step, schools are distributed across grid cells according
to the active map for their species, age, and season. This creates the
spatial overlap patterns that drive predation.

#### 2. Predation
The central process. For each predator school in each grid cell:
- All co-occurring organisms (schools + resource groups) within the
  predator's size window are identified as **accessible prey**
- Predation success depends on the predator's maximum ingestion rate
  and the relative biomass of available prey
- Prey biomass is removed proportionally — prey schools lose
  individuals to predation mortality

The **predator/prey size ratio** parameters (`predation.predPrey.sizeRatio.min`
and `.max`) define the feeding window. Typical values are 1.0-5.0,
meaning a predator eats prey 1x to 5x smaller than itself.

#### 3. Starvation
If a school's predation efficiency (actual food intake / maximum
possible intake) falls below a critical threshold
(`predation.efficiency.critical`), the school suffers **starvation
mortality**. This creates a feedback loop: when prey is scarce,
predators starve, reducing predation pressure, which allows prey to
recover.

#### 4. Growth
Fish grow according to the **von Bertalanffy growth equation**:

`L(t) = L∞ × (1 - exp(-K × (t - t₀)))`

Where L∞ is the asymptotic length, K is the growth coefficient, and
t₀ is the theoretical age at zero length. Weight is derived from
length via the allometric relationship `W = a × L^b`.

Growth parameters are species-specific and set from literature
(e.g., FishBase).

#### 5. Natural Mortality
In addition to predation and starvation, schools experience
**additional mortality** (also called "diverse" mortality) at a
constant rate. This represents all other causes of death not
explicitly modelled (disease, senescence, etc.). Separate rates can
be set for larvae and post-larval stages.

#### 6. Reproduction
Mature females (determined by size or age at maturity) produce eggs
proportional to their body weight and a fecundity parameter (eggs
per gram of mature female). Egg production follows a **seasonal
pattern** defined by a CSV file specifying reproductive activity at
each time step. New schools of age-0 fish are seeded into the
population each time step based on total egg production.

#### 7. Fishing Mortality
Fishing removes biomass at a specified rate per species per time step.
Parameters include:
- **Annual fishing mortality rate** (F, year⁻¹)
- **Age/size at recruitment** to the fishery
- **Seasonal patterns** (optional, via fishery selectivity maps)
- **Catchability** and **discards** (in the extended fisheries module)

Multiple fisheries can target different species with different gears,
selectivities, and spatial patterns.

---

### The Spatial Grid

The simulation domain is a **2D regular or curvilinear grid**
covering the study area. Two grid types are supported:

- **OriginalGrid** — regular latitude/longitude grid defined by
  upper-left and lower-right corners plus number of cells (nx × ny)
- **NcGrid** — irregular grid read from a NetCDF file with explicit
  latitude/longitude coordinates per cell (used for curvilinear grids
  from ocean models)

A **land/sea mask** determines which cells are ocean (habitable) and
which are land. The grid resolution typically ranges from 1/12° to
1/3° depending on the application area.

---

### Time Stepping

OSMOSE operates on a fixed time step, typically **bi-weekly or
monthly** (12 or 24 steps per year). At each step, all processes
are applied in sequence:

1. Movement (distribute schools to grid cells)
2. Predation (size-based feeding interactions)
3. Starvation (mortality from insufficient feeding)
4. Natural mortality (background death rate)
5. Fishing mortality (harvest removal)
6. Growth (von Bertalanffy length/weight update)
7. Reproduction (egg production and seeding)

The order of processes within a time step is important — predation
occurs before growth, so feeding success in a time step affects
the energy available for growth.

---

### End-to-End Coupling

OSMOSE is designed to be coupled with **hydrodynamic and
biogeochemical models** (e.g., ROMS, NEMO) to form "end-to-end"
ecosystem models:

```
Physical ocean model (ROMS/NEMO)
        ↓
Biogeochemical model (PISCES)  →  plankton biomass fields
        ↓
OSMOSE (fish + invertebrates)  →  emergent food web
        ↓
Fisheries & economic modules   →  catch, revenue, fleet dynamics
```

The biogeochemical model provides spatiotemporal plankton biomass
(the LTL forcing), which feeds into OSMOSE as the base of the food
web. OSMOSE then simulates all fish-to-fish and fish-to-plankton
interactions. Optional economic modules simulate fleet behavior and
market dynamics.

---

### Key References

- Shin & Cury (2001). Exploring fish community dynamics through
  size-dependent trophic interactions using a spatialized individual-based
  model. *Aquatic Living Resources*, 14(2), 65-80.
- Shin & Cury (2004). Using an individual-based model of fish
  assemblages to study the response of size spectra to changes in
  fishing. *Canadian Journal of Fisheries and Aquatic Sciences*,
  61(3), 414-431.
- Travers et al. (2009). Two-way coupling versus one-way forcing
  of plankton and fish models to predict ecosystem changes in the
  Benguela. *Ecological Modelling*, 220, 3089-3099.

For full documentation visit [osmose-model.org](https://osmose-model.org)
"""
            ),
        ),
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
