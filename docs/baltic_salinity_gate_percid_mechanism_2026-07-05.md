# Why the cod salinity gate suppresses Baltic percids — mechanism investigation

- **Date:** 2026-07-05. **Method:** systematic-debugging on the merged salinity-gate feature (master `b212a09`). Two whole-food-web probes (OSMOSE Baltic, `nyear=15`, `ndt=24`, seed 0) run gate-OFF vs gate-ON (cod `sp0`, real bottom-salinity field `data/baltic/baltic_salinity_bottom_climatology.nc`), comparing late-window (final third) biomass and mortality-by-cause across the fish community. Real-world grounding verified via scite (Eklöf et al. 2020; Bergström et al. 2015 — both non-retracted).
- **Status:** explanation only, **not a fix**. This confirms the salinity gate is a *spatial-realism* feature that moves percids **down**, via the correct real-world mechanism, and would if anything worsen the existing percid overshoot.

---

## 1. TL;DR

- The naive prediction — "gate cod out of the low-salinity coast → percids sheltered from cod → percids up" — is **wrong**, and the model is behaviourally correct.
- Enabling the gate triggers a **three-spined-stickleback competitor / mesopredator release**: with cod excluded from the low-salinity coastal cells, stickleback (small, low-salinity-tolerant, co-located with percids) is freed from cod predation and **booms (+94%)**. The stickleback surge then suppresses percids through competition for shared zooplankton prey and predation on percid eggs/larvae.
- The gate **does** reduce direct cod→percid predation (perch predation mortality −49%), but percids die *more* — of starvation (perch starvation +13%) and reduced production (pikeperch recruitment-limited) — so net percid biomass falls (perch −35%, pikeperch −33%).
- The competing "cod concentrated in basins depletes the shared clupeid prey percids need" hypothesis is **refuted**: herring and sprat are flat (±1.6%).
- This is a documented Baltic **coastal regime shift** (cod/piscivore loss → stickleback dominance → coastal-piscivore recruitment collapse). The model reproduced it emergently — a genuine validation signal, not an artefact.

---

## 2. The evidence

### 2.1 Whole-food-web biomass, gate OFF → ON

| Species | OFF (t) | ON (t) | Δ | Reading |
|---|---:|---:|---:|---|
| cod (sp0) | 1 710.6 | 1 964.9 | **+14.9%** | cod concentrates in productive saline basins → does better |
| **stickleback (sp7)** | 202 057.7 | 392 055.2 | **+94.0%** | **the single largest response in the web** |
| perch (sp4) | 0.77 | 0.50 | −35.2% | down (near-collapse level; noisy but directionally clear) |
| pikeperch (sp5) | 277.1 | 185.5 | −33.1% | down (the real, non-noisy percid signal) |
| flounder (sp3) | 2 800.1 | 1 936.4 | −30.8% | other demersal, also down |
| herring (sp1) | 17 694 091 | 17 767 365 | +0.4% | flat |
| sprat (sp2) | 5 457 984 | 5 367 961 | −1.6% | flat |

Only the gate on cod (`sp0`) changes between the two runs, so **every** non-cod change is necessarily food-web-mediated. Stickleback is the one large mover; by elimination it is the driver of the percid decline.

### 2.2 Mortality-by-cause (late-window mean, gate OFF → ON) — the decisive evidence

- **perch:** predation **−49%**, starvation **+13%**. The gate *did* cut cod predation on perch — yet perch die *more*, of hunger. Less predation, worse outcome ⇒ the limiting pressure is not cod.
- **pikeperch:** predation −30%, starvation −36%, additional −32%, fishing −34% — **every cause falls ~30%, in lockstep with the −33% biomass**. That signature is a **production/recruitment decline** (fewer fish produced, food-limited), not a mortality-driven kill-off.
- **stickleback:** every cause up ~96–158% — a population explosion (more individuals ⇒ more absolute removals across all causes).
- **herring:** flat across every cause (predation +1%, starvation +0%) — the clupeid base is untouched.

### 2.3 The chain

```
gate ON
  → cod excluded from low-salinity coastal cells (and concentrated in saline basins; cod +14.9%)
  → stickleback (co-located in those coastal cells, low-salinity-tolerant) RELEASED from cod predation
  → stickleback booms (+94%)
  → competition for shared zooplankton  +  predation on percid eggs/larvae
  → perch starve (+13% starvation) ; pikeperch recruitment/production collapses (all causes −30%)
  → percids DOWN (perch −35%, pikeperch −33%)
```

Direct cod→percid predation relief (perch predation −49%) is real but is **outweighed** by the stickleback-mediated bottom-up/egg-predation loss.

---

## 3. Hypotheses tested and their verdicts

| Candidate mechanism | Verdict | Basis |
|---|---|---|
| (a) direct cod→percid predation *rose* (more overlap) | **Refuted** | perch predation mortality **−49%** (fell), pikeperch predation −30% |
| (b) cod concentrated in basins depletes shared clupeid/zoo prey → percid starvation | **Refuted as framed** | herring/sprat **flat (±1.6%)**; the competition is not for clupeids |
| (c) mesopredator/competitor release via stickleback | **Confirmed** | stickleback **+94%** (largest response); perch starvation +13%; pikeperch production-limited; only cod's movement changed |

The percid decline is driven by the **stickleback surge (competition + egg/larval predation)**, released by removing cod's spatial pressure from the coast.

---

## 4. Real-world grounding

The model emergently reproduced a well-documented Baltic dynamic: the loss of large coastal piscivore pressure releases three-spined stickleback, which then suppresses the recruitment of coastal piscivores (perch, pikeperch, pike) via competition and predation on their eggs and larvae.

- **Eklöf, J. S., et al. (2020).** A spatial regime shift from predator to prey dominance in a large coastal ecosystem. *Communications Biology*, 3, 459. https://doi.org/10.1038/s42003-020-01180-0 — documents the predator→prey (piscivore→stickleback) regime shift along the Baltic coast. *(retrieved via scite; no editorial notices)*
- **Bergström, U., Olsson, J., Casini, M., et al. (2015).** Declining coastal piscivore populations in the Baltic Sea: Where and when do sticklebacks matter? *Ambio*, 44(Suppl. 3), 462–471. https://doi.org/10.1007/s13280-015-0665-5 — links the stickleback increase to declining coastal piscivore populations and identifies where the effect bites. *(retrieved via scite; no editorial notices)*

Related corpus (retrieved, on-topic): "The rise of the three-spined stickleback – eco-evolutionary consequences of a mesopredator release"; "Increases of opportunistic species in response to ecosystem change: the case of the Baltic Sea three-spined stickleback" (*ICES J. Mar. Sci.*, 2022, https://doi.org/10.1093/icesjms/fsac073).

---

## 5. Limitation

A direct read of cod's **diet composition** (the cod stickleback/percid diet fraction, OFF vs ON) would be the final confirmatory link. It is unavailable: the Python engine does **not** write `dietMatrix` CSVs even with `output.diet.composition.enabled` set (verified — both the in-memory and a disk `run()` produced no `baltic_dietMatrix*.csv`). The explanation stands on three converging, independent channels — biomass, mortality-by-cause, and the spatial logic of the gate (only cod moved) — which agree without the diet matrix. Adding diet output on the Python engine is a separate, out-of-scope engine task.

---

## 6. Implication

Consistent with the merged feature's framing and [[project-baltic-percid-salinity-refuge-review]]: the salinity gate is a **spatial-realism correction, not a percid-overshoot fix**. It reduces percid biomass through the correct real-world mechanism (stickleback release), so it would *worsen* the standing percid overshoot rather than cure it. No further modelling action is warranted from this result; the finding is the deliverable.
