#!/usr/bin/env python
"""Baltic herring growth and maturity from DATRAS BITS catch-at-age (CA) records.

Fetches (or reuses) BITS CA XML per year/quarter from the public DATRAS web service, keeps herring
(Clupea harengus, AphiaID 126417), and computes:

* length at 50% maturity (L50): count-weighted logistic MLE, mature ~ length, per quarter and
  region (west = SD 22-24, central = the rest; rectangles are assigned by the longitude band);
* maturity at age (proportion mature by ICES age);
* length at age (count-weighted mean by ICES age, 1 Jan birthday), with a von Bertalanffy fit on
  post-hatch age (May hatch assumed: Q1 -> age - 0.2, Q4 -> age + 0.45);
* the length-weight relationship from individual weights.

Maturity coding:
* standard BITS key (scale M6, codes 61-66): 61 immature, 62-65 mature, 66 excluded;
* with --national, Roman-numeral national keys are added under the BITS herring rule of
  Maioli et al. (2026, doi:10.21203/rs.3.rs-10856339/v1): stages I-II immature (Poland; Russia
  uses II as its lowest stage), III-VIII mature, IX excluded.

Usage:
    .venv/bin/python scripts/datras_herring_growth_maturity.py \
        --xml-dir <cache dir> --years 2018-2022 --quarters 1,4 --national \
        --out docs/diagnostics/herring_bits_growth_maturity_2018-2022.json
"""

from __future__ import annotations

import argparse
import json
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit, minimize

BASE = "https://datras.ices.dk/WebServices/DATRASWebService.asmx/getCAdata"
HERRING = 126417
ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8}


def _val(text: str | None):
    if text is None:
        return None
    s = text.strip()
    if s in ("", "-9", "-9.0", "-9.0000"):
        return None
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def load_ca(xml_dir: Path, year: int, quarter: int) -> list[dict]:
    path = xml_dir / f"BITS_CA_{year}_Q{quarter}.xml"
    if not path.exists():
        with urllib.request.urlopen(
            f"{BASE}?survey=BITS&year={year}&quarter={quarter}", timeout=300
        ) as r:
            path.write_bytes(r.read())
    content = path.read_bytes()
    # The stdlib parser is only exposed to XXE / entity-expansion attacks through a DTD, and the
    # DATRAS service never sends one: refuse any document that carries a DOCTYPE or ENTITY.
    if b"<!DOCTYPE" in content or b"<!ENTITY" in content:
        raise ValueError(f"{path}: refusing XML with a DTD or entity declarations")
    root = ET.fromstring(content)
    out = []
    for rec in root:
        row = {child.tag.split("}")[-1]: _val(child.text) for child in rec}
        if HERRING in (row.get("Valid_Aphia"), row.get("SpecCode")):
            out.append(row)
    return out


def length_cm(r: dict) -> float | None:
    lc, code = r.get("LngtClass"), r.get("LngtCode")
    if lc is None:
        return None
    return float(lc) if code in (1, "1") else float(lc) / 10.0  # "." / 0 / 5 -> mm


def region(r: dict) -> str | None:
    at, ac = r.get("AreaType"), r.get("AreaCode")
    if at == 4:
        return "west" if ac in (22, 23, 24) else "central"
    if at == 0 and isinstance(ac, str) and len(ac) == 4 and ac[2] in "GH" and ac[3].isdigit():
        return "west" if (10 if ac[2] == "G" else 20) + int(ac[3]) < 14 else "central"
    return None


def maturity_flag(code, national: bool) -> int | None:
    if code in (61, 1):
        return 0
    if code in (62, 63, 64, 65, 2, 3, 4, 5):
        return 1
    if national and isinstance(code, str) and code in ROMAN:
        return 0 if ROMAN[code] <= 2 else 1
    return None


def logistic_l50(L: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    def nll(p):
        z = np.clip(p[1] * (L - p[0]), -50, 50)
        pr = np.clip(1 / (1 + np.exp(-z)), 1e-12, 1 - 1e-12)
        return -np.sum(w * (y * np.log(pr) + (1 - y) * np.log(1 - pr)))

    a, b = minimize(
        nll,
        x0=[15.0, 1.0],
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000},
    ).x
    return {
        "L50": float(a),
        "L25": float(a - np.log(3) / b),
        "L75": float(a + np.log(3) / b),
        "slope": float(b),
        "n_fish": float(w.sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--xml-dir", type=Path, required=True)
    ap.add_argument("--years", default="2018-2022")
    ap.add_argument("--quarters", default="1,4")
    ap.add_argument("--national", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.xml_dir.mkdir(parents=True, exist_ok=True)
    y0, y1 = (int(x) for x in args.years.split("-"))
    quarters = [int(q) for q in args.quarters.split(",")]

    rows = []
    for year in range(y0, y1 + 1):
        for q in quarters:
            for r in load_ca(args.xml_dir, year, q):
                L = length_cm(r)
                if L is None or L <= 0:
                    continue
                rows.append(
                    {
                        "L": L,
                        "n": float(r.get("CANoAtLngt") or 1),
                        "age": r.get("Age"),
                        "q": q,
                        "mature": maturity_flag(r.get("Maturity"), args.national),
                        "w": r.get("IndWgt"),
                        "region": region(r),
                        "country": r.get("Country"),
                    }
                )
    out: dict = {
        "source": f"DATRAS BITS CA, {y0}-{y1}, Q{','.join(map(str, quarters))}, herring AphiaID {HERRING}",
        "national_keys_included": args.national,
        "n_records": len(rows),
        "n_fish": float(sum(r["n"] for r in rows)),
        "countries": sorted({str(r["country"]) for r in rows}),
        "maturity_L50": {},
        "maturity_at_age": {},
        "length_at_age": {},
    }
    for q in quarters:
        for reg in ("all", "central", "west"):
            sub = [
                r
                for r in rows
                if r["q"] == q and r["mature"] is not None and reg in ("all", r["region"])
            ]
            if len(sub) > 50:
                out["maturity_L50"][f"Q{q}/{reg}"] = logistic_l50(
                    np.array([r["L"] for r in sub]),
                    np.array([r["mature"] for r in sub], float),
                    np.array([r["n"] for r in sub]),
                )
            mat, laa = {}, {}
            for a in range(12):
                s_m = [r for r in sub if r["age"] == a]
                if s_m:
                    w = np.array([r["n"] for r in s_m])
                    mat[a] = {
                        "prop_mature": float(
                            (w * np.array([r["mature"] for r in s_m])).sum() / w.sum()
                        ),
                        "n_fish": float(w.sum()),
                    }
                s_l = [
                    r for r in rows if r["q"] == q and r["age"] == a and reg in ("all", r["region"])
                ]
                if len(s_l) >= 10:
                    w, L = np.array([r["n"] for r in s_l]), np.array([r["L"] for r in s_l])
                    m = float((w * L).sum() / w.sum())
                    laa[a] = {
                        "mean_L": m,
                        "sd_L": float(np.sqrt((w * (L - m) ** 2).sum() / w.sum())),
                        "n_fish": float(w.sum()),
                        "post_hatch_age": a - 0.2 if q == 1 else a + 0.45,
                    }
            if mat:
                out["maturity_at_age"][f"Q{q}/{reg}"] = mat
            if laa:
                out["length_at_age"][f"Q{q}/{reg}"] = laa

    t, L, w = [], [], []
    for q in quarters:
        for v in out["length_at_age"].get(f"Q{q}/all", {}).values():
            t.append(v["post_hatch_age"])
            L.append(v["mean_L"])
            w.append(v["n_fish"])
    p, _ = curve_fit(
        lambda t, linf, k, t0: linf * (1 - np.exp(-k * (t - t0))),
        np.array(t),
        np.array(L),
        p0=[25, 0.4, -0.5],
        sigma=1 / np.sqrt(np.array(w)),
        maxfev=20000,
    )
    out["vb_fit_all"] = {"Linf": float(p[0]), "K": float(p[1]), "t0": float(p[2])}
    lw = [(r["L"], r["w"]) for r in rows if r["w"] and r["w"] > 0]
    b, loga = np.polyfit(np.log([x[0] for x in lw]), np.log([x[1] for x in lw]), 1)
    out["length_weight"] = {"a": float(np.exp(loga)), "b": float(b), "n": len(lw)}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(
        json.dumps(
            {
                k: out[k]
                for k in ("n_fish", "countries", "maturity_L50", "vb_fit_all", "length_weight")
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
