# Scientific Review — Baltic Fish Life-Cycle Document

> An external-evidence review of the parameter values and biological claims in
> [`baltic-fish-lifecycle.md`](baltic-fish-lifecycle.md), cross-checked against three
> authoritative sources via their MCP servers:
> **scite** (primary literature), **ICES** (Baltic stock assessments + reference points),
> and **HELCOM** (HOLAS 3 regional ecosystem assessment).
> Date of review: 2026-06. Reviewer: automated multi-source check; verdicts and citations below.

---

## 1. Method and caveats

Each major claim in the life-cycle document was put to the source best able to test it: life-history
and food-web *mechanisms* to scite; *stock-level* reference points (F, B, M, maturity) to ICES; and
*regional ecosystem realism* (community status, habitat, stressors) to HELCOM HOLAS 3.

**Retrieval caveats (material to how strongly to read the verdicts):**
- **scite** returned discovery-level metadata (titles + DOIs, all verified to exist and be on-topic),
  but full-text excerpts and `editorialNotices` did not surface this session — so *exact numeric
  values* should be confirmed against the linked PDFs, and retraction status was not machine-checked
  (none expected for the journals/years cited).
- **ICES** quantitative assessment data (SAG) retrieved cleanly; the **Ecosystem Overview / publication
  text endpoints returned HTTP 403**, so the regime-shift/seal narrative is inferred from the
  assessment time series, not quoted verbatim.
- **HELCOM** HOLAS 3 assessment *layers* (the authoritative status data) retrieved cleanly; the
  `search_publications` tool is a narrow keyword index of helcom.fi posts, so a non-hit there means
  "not in the indexed posts", not "HELCOM has no view".

---

## 2. Overall assessment

**The Baltic life-cycle document is scientifically sound and, importantly, already honest about its
own main weakness** (the dated cod parameters, flagged in its §9). The food-web structure, the
spawning phenology, the coastal regime-shift framing, and the eutrophied/hypoxic LTL base are all
well-supported by the literature and by HOLAS 3. Two items warrant revision (one rationale, one
label), and several values for unassessed coastal species simply can't be validated against ICES.

| Claim / parameter group | scite | ICES | HELCOM | Overall |
|---|---|---|---|---|
| Cod–sprat–herring triad & post-collapse regime shift | ✅ Supported | ✅ (sprat ↑, cod collapsed) | ✅ (pelagics good where demersal failed) | **Well supported** |
| Eastern Baltic cod growth/maturation collapse | ✅ Direction | ✅ (tonnes-SSB dropped ~2019) | ✅ (demersal BQR failing + hypoxia) | **Supported; numbers are historical** |
| Cod L∞ 110 cm / maturity 38 cm / Bpa 120 kt anchor | ⚠ historical/high | ❌ dated (pre-collapse; relative units now) | ⚠ (collapse confirmed) | **Dated — keep only as explicit pre-collapse baseline** |
| Cod elevated background M (0.20) for seals | ⚠ rationale overstated | ✅ direction (ICES raised M) | ✅ (seal recovery real) | **Value OK; rationale should be revised** |
| Grey-seal cod removal 30–50 kt/yr; seal M 0.15–0.4 | ❌ not substantiated for cod | — | ✅ seal pop. large/recovered | **Revise: parasite-mediated, not direct predation** |
| Herring spring + autumn (bimodal) spawning | ✅ Supported | ✅ maturity/L∞ consistent | ✅ herring EFH layer | **Well supported** |
| Sprat spawning, fecundity, top-down control, F=0.32 | ✅ Supported | ✅ F≈Fmsy 0.34 | ✅ pelagic status | **Well supported / best-calibrated** |
| Stickleback boom + percid "wasp-waist" recruitment suppression | ✅ Supported | n/a | ⚠ no indicator | **Supported by literature (cite papers, not HELCOM)** |
| Flounder (single stock, F=0.04, maturity 22) | ✅ plausible | ❌ structural (ICES = 4 stocks, ratio RPs) | ✅ flounder EFH layers | **Biologically OK; structurally simplified** |
| Perch / pike-perch / smelt life-history | ✅ plausible | — not ICES-assessed | ⚠ coastal-fish group only | **Plausible; validate vs HELCOM/national, not ICES** |
| Spawning/nursery maps per stage | — | — | ✅ species EFH rasters exist | **Regionally realistic** |
| Eutrophied/hypoxic LTL base | — | — | ✅ Poor–Bad; deep basins fail O₂ | **Well supported** |

---

## 3. Findings by theme

### 3.1 The cod–clupeid food web (document §1, §3, §7) — well supported

The document's central food-web wiring — cod preying on sprat and herring (accessibility 0.4/0.4),
sprat/herring as zooplanktivores, and the post-cod-collapse "sprat release" — is the best-supported
part of the model. It is textbook central-Baltic ecology: Casini et al. (2008) demonstrate the
trophic cascade and threshold-like control switch; Köster & Möllmann (2000) establish clupeid
top-down pressure. ICES corroborates the *state* (cod F collapsed and SSB at historic low post-2019;
sprat above Bpa with F at Fmsy). HELCOM HOLAS 3 corroborates the *pattern*: demersal (cod) BQR
0.15–0.53 (failing) across the Baltic Proper, while pelagics (sprat/herring) score well in the Gulf
of Bothnia/Finland — exactly the "pelagics fine where demersals collapsed" signature.

### 3.2 Eastern Baltic cod parameters (document §5, §9) — supported in direction, values are historical

All three sources agree the **collapse is real** and the document's **§9 caveat is correct**. Svedäng
et al. (2024, *Ecology and Evolution*, the paper the document cites — verified genuine and on-point)
shows size-at-maturity tracks long-term growth conditions; Mion et al. (2021) and Hüssy et al. (2017)
document multidecadal growth retardation; ICES abandoned analytical SSB-in-tonnes ~2019 and the
directed fishery has been closed since 2019; HELCOM shows demersal status failing and the spawning
basins hypoxic.

Two refinements:
- **L∞ = 110 cm is high even as a historical value.** scite notes tagging-based growth
  reconstructions (Mion 2021, Hüssy 2017) imply a lower asymptotic length; if 110 cm is retained it
  should be sourced explicitly rather than asserted.
- **maturity 38 cm and Bpa 120 kt are pre-collapse.** Contemporary EBC matures far smaller (often
  <25 cm). The document already labels these historical (its §9 and the in-file config comment) — the
  review **confirms** that labelling and recommends keeping it prominent. Note ICES reference points
  for cod are now expressed in **relative index units**, so the absolute 120 kt anchor cannot be
  mapped onto the current ICES scale at all (a unit caveat worth adding).

### 3.3 The grey-seal mortality rationale (document §2, §4) — revise

This is the one place where the document's *rationale* outruns the evidence. The model raises cod
background M to 0.20 yr⁻¹ and attributes the increment to grey-seal predation (~30–50 kt/yr,
M≈0.15–0.4). The verdicts:
- **The value's direction is fine.** ICES did raise M-at-age for EBC (seals + poor condition) in the
  2019 benchmark, so an elevated background M is consistent. HELCOM confirms the grey-seal population
  is large and recovered (integrated BQR 0.30; the pop-trend indicator "fails" only because *recent*
  growth has slowed from a high base).
- **But the cod-specific magnitude is not substantiated.** scite found no central-Baltic source for
  ~30–50 kt/yr cod removal or a +0.15–0.4 cod M from grey seals; Hansson et al. (2018) explicitly
  conclude environment and fisheries impacts exceed seal predation, and the seal→cod link in the
  central Baltic is largely **indirect via the *Contracaecum osculatum* nematode** (seals are the
  definitive host; the worm degrades cod condition — Sokolova et al. 2020).

**Recommendation:** keep the elevated M value, but reframe its rationale — describe the 0.20 yr⁻¹ as a
lumped "unaccounted natural mortality" capturing **parasite-mediated condition loss, hypoxia, and some
seal predation**, rather than predominantly direct seal consumption. Cite Hansson (2018) and Sokolova
(2020) for the mechanism. (No code change is strictly required — this is a documentation/justification
fix — but if a future recalibration splits M, the seal component should be the smaller term.)

### 3.4 Clupeid reproduction & exploitation (document §4, §5) — well supported

- **Herring bimodal (spring-dominant + autumn) spawning** is supported: spring April–May peak is
  well established (Polte et al. 2023; western-Baltic spring-spawning work), and a distinct,
  smaller, reproductively-stressed autumn component is documented (Rajasilta et al. 2015). The
  model's bimodal season file is appropriate. ICES finds the maturity (18 cm) and L∞ (27 cm) standard
  for the small-bodied central-Baltic herring.
- **Sprat** is the best-calibrated stock in the whole model: protracted spring–summer spawning
  peaking ~June (Haslob et al. 2012), recruitment tightly coupled to temperature/transport
  (MacKenzie & Köster 2004; Baumann et al. 2006), top-down control by cod (Casini 2008) — and the
  model's **F = 0.32 yr⁻¹ sits just below ICES Fmsy = 0.34** (spr.27.22-32, 2024), with maturity 9 cm
  and L∞ 16 cm both standard.

### 3.5 Coastal species & the stickleback regime shift (document §3, §4) — supported by literature, not by ICES/HELCOM indicators

- **Perch, pike-perch, smelt are not ICES-assessed** (confirmed: zero Baltic stocks in the ICES SAG
  for *Perca*, *Sander*, *Osmerus*). Their life-history values are biologically plausible per scite
  (pike-perch L∞ ~90 cm and maturity ~40 cm are within the species' brackish range; perch L∞ 45 cm
  reasonable; smelt as a small short-lived osmerid). **Validate these against HELCOM coastal-fish
  indicators / national surveys, not ICES.** HELCOM does assess perch/pike-perch within its coastal
  "key species / piscivores" groups (better status north/SW, failing in the SE basins — Gulf of Riga
  and Eastern Gotland EQR 0.15), consistent with the model treating them as coastal species.
- **The stickleback boom and its suppression of perch/pike-perch recruitment** (the "wasp-waist" /
  mesopredator-release framing) is well supported in the primary literature (Olsson et al. 2022;
  Eklöf et al. 2020; Bergström et al. 2015) but is **not** in HELCOM's HOLAS 3 indicator set (no
  stickleback indicator/layer). The model's framing is justified — **cite the peer-reviewed papers,
  not HELCOM, for this mechanism.** (Note: the model does not explicitly encode a stickleback→percid
  egg-predation link beyond the accessibility matrix; this is a candidate future refinement, not a
  current error.)

### 3.6 Habitat & the LTL base (document §6, §1) — well supported

- HELCOM maintains **species-specific spawning Essential-Fish-Habitat rasters** for cod, herring,
  sprat, and flounder, and the cod spawning extent spans the central-Baltic deep basins — confirming
  the model's per-stage spawning-ground maps (Bornholm/Gdańsk/Gotland) are regionally sensible.
- The model's eutrophied, hypoxia-stressed LTL base is strongly corroborated: HOLAS 3 integrated
  eutrophication is **Poor–Bad** across the Baltic Proper, and the oxygen-debt indicator **fails in
  every deep basin** that hosts cod spawning — directly supporting the "hypoxic bottoms affecting cod
  eggs and benthos" rationale. (HELCOM has no discrete zooplankton-biomass indicator, so the
  zooplankton resource groups are supported on the productivity/phytoplankton side only.)

---

## 4. Prioritized recommendations for the model

1. **(High, documentation)** Keep the eastern-Baltic-cod §9 caveat front-and-centre, and add the
   ICES unit note: ICES no longer expresses cod (and central-Baltic herring) reference points in
   absolute tonnes, so the 120 kt recruitment anchor is a *historical* quantity that cannot be
   mapped onto the current ICES relative scale. If 110 cm L∞ is retained, source it explicitly
   (Svedäng 2024 / Mion 2021).
2. **(Medium, documentation/justification)** Revise the grey-seal rationale for cod's elevated
   background M (§2/§4): present 0.20 yr⁻¹ as lumped unaccounted M (parasite-mediated condition loss
   + hypoxia + some seal predation), citing Hansson (2018) and Sokolova (2020), rather than
   predominantly direct seal consumption of 30–50 kt/yr.
3. **(Low)** Note the flounder structural simplification: ICES manages Baltic flounder as **four**
   data-limited stocks (incl. the cryptic *Platichthys solemdali*) with ratio reference points; the
   model's single-flounder absolute F = 0.04 is biologically plausible but not validatable against
   the ICES framework.
4. **(Low)** Flag that perch/pike-perch/smelt cannot be validated against ICES; cross-check against
   HELCOM coastal-fish indicators / national data, and cite the coastal regime-shift literature for
   the stickleback mechanism.
5. **(Informational)** The sprat parameterization is exemplary (F just below ICES Fmsy) and can serve
   as the calibration template; herring and the food-web wiring need no change.

**Net:** no parameter is *wrong* in a way that invalidates the model; the corrections are (1) keeping
the historical-cod labelling explicit and unit-aware, and (2) re-justifying the seal-mortality term.
The document's existing caveats already anticipate most of this.

---

## 5. References

**Primary literature (via scite; DOI links — confirm exact figures against the PDFs):**
- Baumann, H., et al. (2006). Recruitment variability in Baltic Sea sprat is tightly coupled to temperature and transport patterns. *CJFAS.* https://doi.org/10.1139/f06-112
- Bergström, U., et al. (2015). Declining coastal piscivore populations in the Baltic Sea: where and when do sticklebacks matter? *Ambio.* https://doi.org/10.1007/s13280-015-0665-5
- Casini, M., et al. (2008). Trophic cascades promote threshold-like shifts in pelagic marine ecosystems. *PNAS.* https://doi.org/10.1073/pnas.0806649105
- Cook, R. M., et al. (2015). Grey seal predation impairs recovery of an over-exploited fish stock. *J. Applied Ecology.* https://doi.org/10.1111/1365-2664.12439
- Eklöf, J., et al. (2020). The rise of the three-spined stickleback — mesopredator release. *bioRxiv.* https://doi.org/10.1101/2020.05.08.083873
- Hansson, S., et al. (2018). The necessity of a holistic approach when managing marine mammal–fisheries interactions. *Ambio.* https://doi.org/10.1007/s13280-018-1131-y
- Haslob, H., et al. (2012). Seasonal variability of fecundity and spawning dynamics of Baltic sprat. *Fisheries Research.* https://doi.org/10.1016/j.fishres.2012.08.002
- Hüssy, K., et al. (2017). Historic changes in length distributions of three Baltic cod stocks: evidence of growth retardation. *Ecology and Evolution.* https://doi.org/10.1002/ece3.3173
- Köster, F. W., & Möllmann, C. (2000). Food consumption by clupeids in the central Baltic: top-down control? *ICES JMS.* https://doi.org/10.1006/jmsc.1999.0630
- MacKenzie, B. R., & Köster, F. W. (2004). Fish production and climate: sprat in the Baltic Sea. *Ecology.* https://doi.org/10.1890/02-0780
- Mion, M., et al. (2021). Multidecadal changes in fish growth rates estimated from tagging data: eastern Baltic cod. *Fish and Fisheries.* https://doi.org/10.1111/faf.12527
- Olsson, J., et al. (2022). Increases of opportunistic species in response to ecosystem change: the Baltic three-spined stickleback. *ICES JMS.* https://doi.org/10.1093/icesjms/fsac073
- Polte, P., et al. (2023). Early arrival of spring-spawning Atlantic herring relates to increasing winter seawater temperature. *J. Fish Biology.* https://doi.org/10.1111/jfb.15811
- Rajasilta, M., et al. (2015). Female ovarian abnormalities and reproductive failure of autumn-spawning herring in the Baltic Sea. *ICES JMS.* https://doi.org/10.1093/icesjms/fsv103
- Receveur, A., et al. (2020). Regional and stock-specific differences in contemporary growth of Baltic cod from tag-recapture data. *ICES JMS.* https://doi.org/10.1093/icesjms/fsaa104
- Sokolova, M., et al. (2020). Physiological condition of eastern Baltic cod infected with *Contracaecum osculatum*. *Conservation Physiology.* https://doi.org/10.1093/conphys/coaa093
- Svedäng, H., et al. (2024). Centurial variation in size at maturity of eastern Baltic cod mirrors conditions for growth. *Ecology and Evolution.* https://doi.org/10.1002/ece3.70382
- van Deurs/Mion, M., et al. (2019). Population density and temperature correlate with long-term trends in growth and maturation of herring and sprat. *PLOS ONE.* https://doi.org/10.1371/journal.pone.0212176

**ICES (Stock Assessment Database, assessment year 2024):** `cod.27.24-32`, `her.27.25-2932`,
`spr.27.22-32`, `fle.27.2223` (+ the split Baltic flounder stocks). Reference points and status as
returned by the ICES MCP / SAG.

**HELCOM HOLAS 3 (2023; reference period 2016–2021):** core indicators "Abundance of key fish
species" / "Abundance of coastal fish key functional groups" (MADS layers 405/406/417/434);
"Population trends and abundance of seals" / seal integrated assessment (layers 277/457); spawning
Essential-Fish-Habitat datasets for cod/flounder/herring/sprat (layers 45–50); "Oxygen debt below the
halocline" (layer 361); "Integrated Eutrophication Status" / pelagic-habitat eutrophication (layers
426/468). HELCOM (2018), *Report on coastal fish in the Baltic Sea* (J. Olsson, SLU Aqua),
https://helcom.fi/.

*Caveat: the ICES Ecosystem-Overview narrative and full scite text excerpts could not be retrieved
this session (403 / metadata-only); figures should be confirmed against the primary sources linked
above.*
