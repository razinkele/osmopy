# Does the existing thermal gate's shape fit cod and herring? Literature check before design

**Date:** 2026-08-10
**Question:** spec C1 wants temperature-dependent stock–recruitment for cod and herring
(motivated by Voss & Quaas 2026, doi:10.1093/icesjms/fsag033). A thermal recruitment gate already
exists (`osmose/engine/processes/thermal_gate.py`, disabled on Baltic — see
`docs/baltic_recruitment_pathway_2026-08-10.md`). It applies a **logistic response to a per-year
temperature series**, built for percids. Does that shape transfer?
**Answer: no, for both species — for different reasons.** Findings only; no design.

## What the existing gate computes

`logistic_response(temp, t50, slope) = 1 / (1 + exp(-(temp - t50)/slope))` — monotonically
**increasing** in temperature, saturating, 0.5 at `t50`. Normalised per year by `thermal_cap`
(`clip(r/r_ref, 0, 1)`) or `mean_preserving`. Provenance: percid year-class strength
(Pekcan-Hekim et al. 2011, *Ambio*; Olin et al. 2019, *Hydrobiologia*), where warm summers produce
strong year classes and the logistic encodes "cool years mostly fail, warm years above threshold
succeed".

Voss & Quaas report the opposite sign for the species C1 targets: climate-driven temperature
increase **negatively** impacts stock productivity of cod and herring. A monotonically increasing
response cannot represent a negative effect at any parameterisation — `t50`/`slope` change where
and how sharply it rises, never its direction.

## Herring: the mechanism is phenological, not a thermal response curve

Polte et al. (2021), *Front. Mar. Sci.* 8:589242, doi:10.3389/fmars.2021.589242 — "Reduced
Reproductive Success of Western Baltic Herring (*Clupea harengus*) as a Response to Warming
Winters" — establishes the causal chain, and it is not a temperature→survival curve:

* an in-situ **threshold of 3.5–4.5 °C triggers the onset of coastal spawning**;
* **warming winters** move that trigger earlier and lengthen the hatching window;
* "the late seasonal onset of cold periods, the corresponding elongation of the period where
  larvae hatch … and early larval hatching peaks **significantly reduced larval production**",
  propagating to juvenile abundance across the whole distribution area.

So the driver is **winter/spring phenology and match–mismatch**, not summer temperature magnitude.
Representing it as `f(annual temperature)` would encode the correlation while discarding the
mechanism — and the paper's own framing is that earlier work relating *climate indices* to
recruitment left "the ecological mechanisms … subject of speculation", which is precisely the
error a naive index-driven gate would repeat.

A defensible herring implementation needs a **spawning-phenology shift** (when the season starts,
how long it runs) — i.e. it acts on `reproduction`'s **seasonality vector**, not on a scalar egg
multiplier. That is a different insertion point in the pathway than the thermal gate occupies.

## Cod: temperature is not the primary recruitment control

Nothing retrieved supports a temperature-driven recruitment response for eastern Baltic cod
comparable to the percid case. The literature consistently attributes recruitment variation to:

* **hydrography — salinity and oxygen (reproductive volume)**: "recruitment is mainly governed by
  the prevailing environmental conditions in the Baltic" (Neuenfeldt 2000, *ICES JMS* 57:300–309,
  doi:10.1006/jmsc.2000.0647); "poor hydrographic ambient conditions for successful egg
  development (Köster et al. 2005)" (Neumann et al. 2017, *CJFAS* 74:833–842,
  doi:10.1139/cjfas-2016-0215);
* **predation on early life stages** by sprat and herring, and **cannibalism** (Neuenfeldt 2000;
  Neumann et al. 2017);
* fishing mortality and stock structure (Jonzén et al. 2002, *MEPS* 240:225–233,
  doi:10.3354/meps240225).

The model **already represents the first two**: the RV gate (hydrography) and the predation kernel
(clupeid egg predation, cannibalism). Adding a thermal gate for cod would stack a third,
multiplicative, weakly-evidenced modifier on top — and the pathway note establishes that gates
compose multiplicatively, so its effect would be conflated with the RV gate's, which is already
the dominant control (gate-off → cod_east 1.61× over ceiling).

Where temperature *is* documented for cod, it acts through mechanisms the thermal gate cannot
express: direct physiological effects on egg and larval survival, and thermal-window shifts — the
route taken by process models such as SCREI (Koenigstein et al. 2017, *Glob. Change Biol.*
24:526–535, doi:10.1111/gcb.13848), which resolves egg fertilisation, survival and development
times per life stage rather than multiplying an annual recruitment scalar.

## Consequences for C1, stated as constraints rather than a design

1. **Do not reuse the logistic as-is for either species.** It has the wrong sign for the effect
   Voss & Quaas describe and, for herring, the wrong mechanism entirely.
2. **Herring and cod need different insertion points.** Herring's evidence points at the
   seasonality vector (phenology); cod's, if pursued at all, at egg/larval survival — not at a
   shared annual multiplier.
3. **Cod may not warrant a thermal gate at all.** Its documented drivers are already in the model.
   The honest first question is whether adding one improves anything, not how to parameterise it.
4. **Any shared-shape shortcut is the trap.** The gate exists and is one config key away from
   being switched on for cod and herring. That convenience is exactly why the shape question had
   to be asked before a spec, not after.

## Method note

This check cost two literature queries and produced a constraint set that would have invalidated a
C1 spec written the obvious way — enable the existing gate, pick `t50`/`slope`, certify. Three
specs were withdrawn today for designing ahead of the evidence; this is the same discipline applied
before the fourth.
