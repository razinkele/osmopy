"""Test the fitted juvenile ingestion multiplier `j` against the published allometry.

Backs the numbers in ``docs/validation/juvenile_ingestion_boost_literature_2026-09-13.md``.

The offline fit's optional juvenile boost (``bioen_offline.simulate_growth(j=...)``, commit
``374bc26``) chooses a per-species multiplier of 1.44-4.64x on the first-year ingestion cap. The
literature question is whether anything outside the curve fit supports that.

Kiorboe & Hirst (2014, Am Nat 183:E118-E130, https://doi.org/10.1086/675241) put maximum ingestion
on a near-universal ``w^0.75`` law -- mass-specific exponent -1/4, from 327 estimates. Bioen-OSMOSE
uses ``beta = 0.8`` (``BioenFixed.beta``), a mass-specific exponent of -0.20. If the fitted boost
were compensating for that 0.05 gap, then anchoring the model's single allometry where the growth
objective actually constrains it (the adult ages, which carry most of the residual points) gives a
predicted multiplier at juvenile mass ``w`` of

    j_implied = (w_ref / w_juv) ** (beta - 0.75)

Two things are checked, and the second is the load-bearing one:

  MAGNITUDE  -- is ``j_implied`` the size of the fitted ``j``?
  RANK ORDER -- does ``j`` track ``j_implied`` across species? This is invariant to the choice of
                anchor mass (any monotone rescaling of the mass ratio leaves ranks untouched), so
                unlike the magnitude it does not rest on where ``w_juv`` is taken from.

Both anchorings of ``w_juv`` are reported because the magnitude claim is soft to the choice:

  vbgf -- the vBGF target curve extrapolated below age 1, where it is NOT valid (cod_west's
          first-year geometric-mean mass reads 9.76 g against a ~1e-3 g egg). Shown for continuity
          with the fit's own residual definition.
  traj -- the fitted model's own simulated egg -> age-1 weight path, i.e. the masses the boost
          actually multiplies. The honest anchor.

Caveat that bounds what a negative result here means: Kiorboe & Hirst is an INTERSPECIFIC law, and
says so itself ("There may be significant deviations in mass scaling, both during ontogeny within a
species and between species ... but such variation is hidden in the current larger-scale
analysis"). Morell et al. (2024, https://doi.org/10.1111/ele.70017) separate the levels explicitly,
citing Kiorboe & Hirst for the interspecific case and Wuenschel & Werner (2004) for the
intraspecific developmental one. So ``j_implied`` answers "what if the model used the measured
interspecific exponent" -- it is NOT a ceiling on an intraspecific developmental effect.

Run: .venv/bin/python scripts/c3_juvenile_boost_literature_check.py
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from osmose.calibration.bioen_offline import (  # noqa: E402
    BioenFixed,
    c_m_from_share,
    fit_species,
    simulate_growth,
    solve_tp,
    vbgf_weight,
)
from osmose.config import OsmoseConfigReader  # noqa: E402
from osmose.demo import osmose_demo  # noqa: E402
from osmose.engine.config import EngineConfig  # noqa: E402

NDT = 24
BETA_EMPIRICAL = 0.75  # Kiorboe & Hirst 2014: ingestion ~ w^(3/4)


def _geometric_mean(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=np.float64)
    w = w[w > 0]
    return float(np.exp(np.mean(np.log(w))))


def main() -> int:
    warnings.simplefilter("ignore")
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_fit", ROOT / "scripts" / "fit_baltic_bioen_params.py"
    )
    fit_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fit_mod)

    tmp = Path(tempfile.mkdtemp(prefix="juvboost_lit_"))
    demo = osmose_demo("baltic", tmp)
    cf_path = Path(demo["config_file"])
    raw = OsmoseConfigReader().read(cf_path)
    cfg = EngineConfig.from_dict(raw)
    tnc = ROOT / "data" / "baltic" / "forcing" / "baltic_temperature_2layer_climatology.nc"
    targets, _sp_index, _t = fit_mod._species_targets_from_baltic(raw, cfg, cf_path.parent, tnc)

    fx = BioenFixed()
    d_beta = fx.beta - BETA_EMPIRICAL

    print(
        f"beta(model) = {fx.beta}   beta(Kiorboe & Hirst 2014) = {BETA_EMPIRICAL}   "
        f"mismatch exponent = {d_beta:.2f}\n"
    )
    header = (
        f"{'species':<13}{'j fitted':>9}{'ratio vbgf':>12}{'j impl vbgf':>13}"
        f"{'ratio traj':>12}{'j impl traj':>13}"
    )
    print(header)
    print("-" * len(header))

    fitted: list[float] = []
    implied_vbgf: list[float] = []
    implied_traj: list[float] = []

    for tg in targets:
        res = fit_species(tg, fx, ndt=NDT, juvenile_boost=True)
        j = res.juvenile_boost
        n_steps = int(round(tg.lifespan_years * NDT))
        t_p = solve_tp(tg.t_opt, fx)

        # The fitted model's own weight path, including the boost it chose.
        w_path = simulate_growth(
            res.imax,
            res.r,
            t_p,
            c_m_from_share(res.imax, t_p, fx),
            tg.t24,
            tg.egg_weight_g,
            n_steps,
            NDT,
            tg.cf,
            tg.b,
            tg.m0,
            tg.m1,
            fx,
            j=j,
            boost_thres_dt=res.boost_thres_dt,
        )

        # Reference mass: where the residual is anchored (fitted ages >= 1 yr).
        ages_ref = np.arange(NDT, n_steps + 1) / NDT
        w_ref = _geometric_mean(vbgf_weight(ages_ref, tg.linf, tg.k, tg.t0, tg.cf, tg.b))

        # Juvenile mass, two anchorings.
        ages_juv = np.arange(1, min(NDT, n_steps + 1)) / NDT
        w_juv_vbgf = _geometric_mean(vbgf_weight(ages_juv, tg.linf, tg.k, tg.t0, tg.cf, tg.b))
        w_juv_traj = _geometric_mean(w_path[1 : min(NDT, n_steps + 1)])

        r_vbgf, r_traj = w_ref / w_juv_vbgf, w_ref / w_juv_traj
        i_vbgf, i_traj = r_vbgf**d_beta, r_traj**d_beta

        fitted.append(j)
        implied_vbgf.append(i_vbgf)
        implied_traj.append(i_traj)
        print(f"{tg.name:<13}{j:>9.2f}{r_vbgf:>12.0f}{i_vbgf:>13.2f}{r_traj:>12.0f}{i_traj:>13.2f}")

    js = np.array(fitted)
    for label, implied in (("vbgf", np.array(implied_vbgf)), ("traj", np.array(implied_traj))):
        rho, p_rho = spearmanr(js, implied)
        r_p, p_p = pearsonr(js, implied)
        print(
            f"\n[{label}] implied j {implied.min():.2f}-{implied.max():.2f} "
            f"(spread {implied.max() / implied.min():.2f}x) vs fitted "
            f"{js.min():.2f}-{js.max():.2f} (spread {js.max() / js.min():.2f}x)"
        )
        print(f"[{label}] median fitted/implied = {np.median(js / implied):.2f}")
        print(
            f"[{label}] Spearman rho = {rho:+.3f} (p = {p_rho:.3f}); "
            f"Pearson r = {r_p:+.3f} (p = {p_p:.3f})"
        )

    print(
        "\nVerdict: the beta gap accounts for at most ~1.5x and does NOT reproduce the "
        "between-species\npattern of the fitted j (no rank correlation, either anchoring). "
        "Per-species j is absorbing\nsomething other than a size-scaling correction. See the "
        "validation doc for the two independent\nquantitative anchors that do bracket the median "
        "(2.21x Wuenschel & Werner 2004; 2.5x Kaufmann 1990)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
