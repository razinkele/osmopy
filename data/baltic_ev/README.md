# baltic_ev — Ev-OSMOSE demonstration fixture

Cloned from `data/baltic/` on 2026-05-18 with bioenergetics + Ev-OSMOSE
genetics enabled. Used by the FIE-on-cod scientific demonstration
(see `docs/tutorials/fie-on-baltic-cod.md`).

**This fixture is NOT calibrated against ICES.** Absolute biomass is
not the target — directional trait response is. See the activation
design at `docs/superpowers/specs/2026-05-18-ev-osmose-activation-design.md`.

## Bioen parameter provenance

All values in `baltic_ev_param-bioen.csv`:

| Parameter | Source |
|---|---|
| `species.beta.sp{i}` | Placeholder ≈ 0.8 (allometric exponent ballpark). Source from Brander 1995 or Baltic-cod bioen study during implementation. |
| `species.bioen.maint.energy.c_m.sp{i}` | Placeholder. Source from Mehner & Wieser 1994 (coldwater gadoid metabolic rates) during implementation; units must match OSMOSE bioen kernel. |
| `species.bioen.assimilation.sp{i}` | 0.65 placeholder (close to OSMOSE default 0.7). |
| `species.bioen.maturity.r.sp{i}` | Placeholder reproductive-allocation fraction. |
| `predation.ingestion.rate.max.bioen.sp{i}` | Placeholder. The genetics demo only requires sp0 (cod) to be in a sensible biological range; rest are non-evolving and only affect background biomass scale. |

**Maturation gate is set as a static threshold, not a reaction norm.** `species.bioen.maturity.m0.sp{i}` and `.m1.sp{i}` MUST be set, otherwise both default to 0.0 (config.py:1783-1784) → `l_mature = m0 + m1*age = 0` → **every school (including egg-stage larvae) is always mature** → all net energy allocates to gonads from day 1 → cod growth is pathologically suppressed → cod never reaches the gear l50=35cm → fishery catches ~0 cod → ZERO FIE signal.

Cod sp0 gets m0=30cm (flat reaction norm — m1=0). This is a **simplifying choice**, not a single-paper literature value: Radtke & Grygiel (2013, https://doi.org/10.1111/jai.12135) report L50=34.8cm for southern Baltic cod males in 1990-2006; Svedäng et al. (2024, https://doi.org/10.1002/ece3.70382) document that eastern Baltic cod L50 has since *halved* to ~20cm. m0=30 sits between the two: cod mature before reaching the gear-vulnerable size (l50=35cm), which lets the FIE signal operate purely through "fast growers cross the gear threshold sooner, slow growers reproduce more often before capture" — clean isolation of the growth-rate pathway. Setting m0=35 would couple maturation timing to gear vulnerability and confound the demo. Pick of stock/era is documented; sensitivity to m0 ∈ {20, 30, 35} is a deferred follow-up.

Other species use length-at-50%-maturity values from generic Baltic life-history sources. This is NOT a maturation reaction norm with age-plasticity — that would require `m1 ≠ 0` and would let maturation co-evolve with growth. **This demo intentionally fixes maturation length to isolate the growth-rate FIE pathway** (the secondary pathway per Heino, Pauli & Dieckmann 2015, https://doi.org/10.1146/annurev-ecolsys-112414-054339). The dominant maturation-evolution FIE pathway documented for cod (Olsen et al. 2004) requires non-zero m1 + an evolving trait targeting `bioen_m0` or `bioen_m1`, which is listed in the spec's out-of-scope follow-ups.

## Genetic-trait parameters

See `baltic_ev_param-genetics.csv` (added in Task 8). Only cod (sp0) has a nonzero-variance
trait declared, targeting `bioen_i_max`.

## Citations

- Brander, K. M. (1995). The effect of temperature on growth of Atlantic
  cod (Gadus morhua L.). *ICES J. Mar. Sci.*, 52(1), 1-10.
- Mehner, T., & Wieser, W. (1994). Energetics and metabolic correlates
  of starvation in juvenile perch. *J. Fish Biol.*, 45(2), 325-333.
- Radtke, K. & Grygiel, W. (2013). Sexual maturation of cod in southern Baltic. *J. Appl. Ichthyol.*, 29(2). https://doi.org/10.1111/jai.12135
- Svedäng, H. et al. (2024). Centurial variation in size at maturity of eastern Baltic cod. *Ecol. Evol.*, 14(10). https://doi.org/10.1002/ece3.70382
- Heino, M., Pauli, B. D., & Dieckmann, U. (2015). Fisheries-induced evolution. *Annu. Rev. Ecol. Evol. Syst.*, 46, 461-480. https://doi.org/10.1146/annurev-ecolsys-112414-054339
- Olsen, E. M. et al. (2004). Maturation trends indicative of rapid evolution preceded the collapse of northern cod. *Nature*, 428: 932-935.
