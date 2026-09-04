#!/usr/bin/env python
"""Offline Java-form bioenergetics parameter-fit CLI (C3 spec §3.4, Task 8/11).

``--gate-b [OUT_DIR]``: fit ``osmose.calibration.bioen_offline`` against the 8 Bay-of-Biscay
demo species in ``data/examples/`` and write a runnable, bioen-on overlay at ``OUT_DIR``
(default ``data/examples_bioen/``) -- the Gate-B config Task 9's cross-engine parity run reads.
This is a SYNTHETIC fit for a demo config, not a validated species calibration: every species
gets the same ``t_opt = 15.0`` / isothermal ``t24`` (matching the config's own
``temperature.value``), because ``data/examples`` carries no cited growth-optimum literature to
fit against -- that is the whole point of Gate B (bioen-on, runnable, non-degenerate), not a
biological claim about Bay-of-Biscay species.

``--baltic``: fits the production Baltic species set (Task 11 -- not yet implemented here).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from osmose.calibration.bioen_offline import (
    BioenFixed,
    FitResult,
    SpeciesTargets,
    bioen_param_lines,
    fit_species,
)
from osmose.config.reader import OsmoseConfigReader

ROOT = Path(__file__).resolve().parent.parent

# RMS-length pin (brief step 5): not a hard gate -- Gate B needs a runnable, non-degenerate
# bioen-on config, not a perfect fit. A species over this prints a warning and is still written.
RMS_PIN_PCT = 15.0


def _species_targets_from_examples(
    raw: dict[str, str], n_sp: int
) -> tuple[list[SpeciesTargets], dict[str, int], dict[str, int], dict[str, float]]:
    """Build one ``SpeciesTargets`` per focal species from the read master config.

    ``t_opt``/``t24``: isothermal 15 C for every species (see module docstring) -- matches the
    Gate-B config's own ``temperature.value;15.0``. ``m1``: not present in ``data/examples`` for
    any species; Java default 0.0 (flat maturity threshold, no age dependence).
    """
    targets: list[SpeciesTargets] = []
    sp_index: dict[str, int] = {}
    zlayer: dict[str, int] = {}
    m0: dict[str, float] = {}
    for i in range(n_sp):
        name = raw[f"species.name.sp{i}"]
        linf = float(raw[f"species.linf.sp{i}"])
        k = float(raw[f"species.k.sp{i}"])
        t0 = float(raw[f"species.t0.sp{i}"])
        lifespan = float(raw[f"species.lifespan.sp{i}"])
        cf = float(raw[f"species.length2weight.condition.factor.sp{i}"])
        b = float(raw[f"species.length2weight.allometric.power.sp{i}"])
        egg_size = float(raw.get(f"species.egg.size.sp{i}", "0.0"))
        egg_weight_key = f"species.egg.weight.sp{i}"
        egg_weight_g = float(raw[egg_weight_key]) if egg_weight_key in raw else cf * egg_size**b
        m0_val = float(raw[f"species.maturity.size.sp{i}"])

        targets.append(
            SpeciesTargets(
                name=name,
                linf=linf,
                k=k,
                t0=t0,
                cf=cf,
                b=b,
                egg_weight_g=egg_weight_g,
                m0=m0_val,
                m1=0.0,
                lifespan_years=lifespan,
                t_opt=15.0,
                t24=np.full(24, 15.0),
            )
        )
        sp_index[name] = i
        zlayer[name] = 0
        m0[name] = m0_val
    return targets, sp_index, zlayer, m0


def _print_fit_table(results: list[FitResult]) -> None:
    header = (
        f"{'species':<14}{'imax':>8}{'r':>8}{'c_m':>14}{'t_p':>8}"
        f"{'rms%':>8}{'w_inf_fit':>12}{'w_inf_vb':>12}{'larv0.5y':>10}{'n_pts':>7}  pin"
    )
    print(header)
    print("-" * len(header))
    for res in results:
        flag = "OK" if res.rms_len_pct <= RMS_PIN_PCT else f"FAIL (>{RMS_PIN_PCT:g}%)"
        print(
            f"{res.name:<14}{res.imax:>8.3f}{res.r:>8.3f}{res.c_m:>14.4g}{res.t_p:>8.3f}"
            f"{res.rms_len_pct:>8.2f}{res.w_inf_fit_g:>12.1f}{res.w_inf_vb_g:>12.1f}"
            f"{res.larval_ratio_half_year:>10.3f}{res.n_points:>7d}  {flag}"
        )


def run_gate_b(out_dir: Path) -> list[FitResult]:
    examples_dir = ROOT / "data" / "examples"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(examples_dir, out_dir)

    master = out_dir / "osm_all-parameters.csv"
    reader = OsmoseConfigReader()
    raw = reader.read(master)
    n_sp = int(raw["simulation.nspecies"])

    fx = BioenFixed()
    targets, sp_index, zlayer, m0 = _species_targets_from_examples(raw, n_sp)
    notes = {
        tg.name: (
            "T_opt=15.0C isothermal placeholder (Gate-B synthetic fit -- data/examples carries "
            "no cited thermal optimum for this species; matches the config's own "
            "temperature.value)"
        )
        for tg in targets
    }

    results = [fit_species(tg, fx) for tg in targets]

    bioen_lines = bioen_param_lines(
        results, fx, zlayer=zlayer, sp_index=sp_index, background_imax={}, notes=notes, m0=m0
    )
    bioen_path = out_dir / "osm_param-bioen.csv"
    bioen_path.write_text("\n".join(bioen_lines) + "\n")

    append_lines = [
        "osmose.configuration.bioen;osm_param-bioen.csv",
        "module.bioenergetics.enabled;true",
        "simulation.bioen.phit.enabled;true",
        "simulation.bioen.fo2.enabled;false",
        "temperature.value;15.0",
        "oxygen.value;300.0",
    ]
    with master.open("a") as fh:
        fh.write("\n" + "\n".join(append_lines) + "\n")

    _print_fit_table(results)
    n_fail = sum(1 for res in results if res.rms_len_pct > RMS_PIN_PCT)
    print(
        f"\n{len(results)} species fitted, {len(results) - n_fail} within the {RMS_PIN_PCT:g}% RMS pin."
    )
    print(
        f"Wrote {bioen_path.relative_to(ROOT)} and appended bioen keys to {master.relative_to(ROOT)}."
    )
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--gate-b",
        nargs="?",
        const="data/examples_bioen",
        default=None,
        metavar="OUT_DIR",
        help="Fit the Bay-of-Biscay demo config (Task 8). Default OUT_DIR: data/examples_bioen",
    )
    parser.add_argument(
        "--baltic",
        action="store_true",
        help="Fit the production Baltic config (Task 11 -- not yet implemented)",
    )
    args = parser.parse_args(argv)

    if args.baltic:
        raise NotImplementedError("--baltic is implemented in Task 11 of the C3 bioen Stage-1 plan")
    if args.gate_b is not None:
        out_dir = Path(args.gate_b)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
        run_gate_b(out_dir)
        return
    parser.print_help()


if __name__ == "__main__":
    main()
