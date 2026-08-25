# cod_west thermal Beverton-Holt fit (C1, spec decision 4)

**Generated:** 2026-08-25 · **n hatch years:** 29 (1993-2021, series historical span 1993-2021) · **stock:** cod.27.22-24 (cod_west, sp0)

Model: `ln(R) = -b0 + beta1*T + ln(SSB) - log1p(b3*SSB) + eps`, fit on the log scale by nonlinear least squares (`scipy.optimize.least_squares`), b3 bounded >= 0. R is age-1: R_{y+1} paired with SSB_y and SST-Q3_y (hatch year y).

## Fits

- **Primary:** beta1=-0.0276 (se=0.1927), p=0.8873, b0=-0.8120, b3=2.90093e-20, n=29
- **Detrended-T** (T ~ year OLS residuals): beta1=0.0995 (se=0.2022), p=0.6269, b0=-0.3418, b3=7.23228e-26, n=29
- **Leave-one-out-terminal** (drops hatch year 2021): beta1=-0.0721 (se=0.1907), p=0.7085, b0=-1.5209, b3=5.98824e-27, n=28

## Verdict (pre-registered, spec decision 4 -- not tunable)

`enabled = beta1 < 0 and p < 0.1 and beta1_detrended < 0` -> **False**

- primary beta1 = -0.0276, p = 0.8873
- detrended beta1 = 0.0995

## Data notes

- corr(SSB, T) over the fitted hatch years = -0.204. If |corr| is large, SSB and T are collinear regressors (both can trend over the record) and SE(beta1) widens as a genuine consequence -- not a fit defect; do not respond by adjusting x0, bounds, or the pre-registered thresholds.

## No cross-check

Voss & Quaas (2026, ICES JMS 83(4), doi:10.1093/icesjms/fsag033) cites Conradt (2023, Univ. Hamburg dissertation) for cod's coefficient; that value is published nowhere accessible and the paper carries no supplement. This self-fit is therefore the SOLE source for cod_west's beta1 -- there is no independent value to validate it against.
