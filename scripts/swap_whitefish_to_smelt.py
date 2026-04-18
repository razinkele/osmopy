"""Swap sp6 from whitefish (Coregonus lavaretus) to European smelt (Osmerus eperlanus).

Smelt is more abundant than whitefish in the Baltic and plays a different
ecological role — small pelagic forage fish rather than salmonid predator.

Parameter sources:
  - BITS CPUE/length 2021-2023 (ICES DATRAS, AphiaID 126736): max 26 cm,
    p99=24 cm, modal 16-18 cm → Linf ≈ 25 cm.
  - FishBase Osmerus eperlanus: K=0.25-0.45, L50=9-11 cm, W=0.005*L^3.05.
  - Baltic literature: spring spawner (Feb-May), lifespan 5-8 yr,
    relative fecundity ~1000 eggs/g.

Run from repo root:
    .venv/bin/python scripts/swap_whitefish_to_smelt.py
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

DATA = Path("data/baltic")
BACKUP_SUFFIX = ".pre-smelt-swap.bak"

# (key, new_value) tuples for sp6 parameters in baltic_param-species.csv
SPECIES_PARAMS_SP6 = {
    "species.name.sp6": "smelt",
    "species.lifespan.sp6": "7",
    "species.lInf.sp6": "25.0",
    "species.K.sp6": "0.35",
    "species.t0.sp6": "-0.30",
    "species.length2weight.condition.factor.sp6": "0.00500",
    "species.length2weight.allometric.power.sp6": "3.050",
    "species.egg.size.sp6": "0.09",
    "species.egg.weight.sp6": "0.0005",
    "species.maturity.size.sp6": "10.0",
    "species.relativefecundity.sp6": "1000",
}

# Reproduction seasonality for smelt: spring spawner (peak Mar-Apr, steps 4-9).
# 24 biweekly steps/year; weights normalized to 1.0.
SMELT_SPAWNING_WEIGHTS = {
    4: 0.05,   # Feb late
    5: 0.12,
    6: 0.20,   # Mar
    7: 0.25,
    8: 0.20,   # Apr
    9: 0.12,
    10: 0.05,  # May early
    11: 0.01,
}


def _backup(p: Path) -> None:
    bak = p.with_suffix(p.suffix + BACKUP_SUFFIX)
    if not bak.exists() and p.exists():
        shutil.copy2(p, bak)


def update_species_csv() -> None:
    path = DATA / "baltic_param-species.csv"
    _backup(path)
    lines = path.read_text().splitlines()
    out = []
    for line in lines:
        # Update header comment
        if "sp6=Whitefish" in line:
            line = line.replace("sp6=Whitefish", "sp6=Smelt")
        # Update Coregonus lavaretus reference
        if "Coregonus lavaretus" in line:
            line = line.replace("Coregonus lavaretus", "Osmerus eperlanus")
        # Key=value lines
        if ";" in line and not line.startswith("#"):
            key = line.split(";", 1)[0].strip()
            if key in SPECIES_PARAMS_SP6:
                line = f"{key};{SPECIES_PARAMS_SP6[key]}"
        out.append(line)
    path.write_text("\n".join(out) + "\n")
    print(f"  {path.name}: updated 11 sp6 parameters")


def rename_movement_maps() -> None:
    for stage in ("juvenile", "adult", "spawning"):
        src = DATA / "maps" / f"whitefish_{stage}.csv"
        dst = DATA / "maps" / f"smelt_{stage}.csv"
        if src.exists() and not dst.exists():
            _backup(src)
            shutil.move(str(src), str(dst))
            print(f"  renamed whitefish_{stage}.csv → smelt_{stage}.csv")
        elif dst.exists():
            print(f"  smelt_{stage}.csv already exists — skip")


def update_movement_config() -> None:
    path = DATA / "baltic_param-movement.csv"
    _backup(path)
    text = path.read_text()

    # Replace section header comment
    text = re.sub(
        r"# sp6: Whitefish.*?\n",
        "# sp6: Smelt — juvenile (0-1, everywhere), adult (1-6, pelagic), spawning (Feb-May, coastal)\n",
        text,
    )

    # Rename species references
    text = text.replace(";whitefish\n", ";smelt\n")
    text = text.replace("/whitefish_", "/smelt_")
    text = text.replace("Whitefish: Steps 18-23 = Oct-Dec. VERIFIED — autumn/winter spawner in northern Baltic.",
                        "Smelt: Steps 4-11 = Feb-May. Spring spawner (coastal/estuarine).")

    # Update age bounds and step ranges for smelt (lifespan 7 → lastAge 6)
    # Juvenile: ages 0-1 (was 0-2)
    text = re.sub(r"(movement\.lastAge\.map19);2\b", r"\g<1>;1", text)
    # Adult: ages 1-6 (was 2-21)
    text = re.sub(r"(movement\.initialAge\.map20);2\b", r"\g<1>;1", text)
    text = re.sub(r"(movement\.lastAge\.map20);21\b", r"\g<1>;6", text)
    # Adult active non-spawning (May-Dec), steps 0-3, 12-23
    text = re.sub(
        r"movement\.steps\.map20;0;1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17\b",
        "movement.steps.map20;0;1;2;3;12;13;14;15;16;17;18;19;20;21;22;23",
        text,
    )
    # Spawning: ages 1-6 (was 2-21)
    text = re.sub(r"(movement\.initialAge\.map21);2\b", r"\g<1>;1", text)
    text = re.sub(r"(movement\.lastAge\.map21);21\b", r"\g<1>;6", text)
    # Spawning steps 4-11 (Feb-May)
    text = re.sub(
        r"movement\.steps\.map21;18;19;20;21;22;23\b",
        "movement.steps.map21;4;5;6;7;8;9;10;11",
        text,
    )

    path.write_text(text)
    print(f"  {path.name}: updated sp6 species name, map files, age bounds, step ranges")


def update_reproduction_seasonality() -> None:
    path = DATA / "reproduction" / "reproduction-seasonality-sp6.csv"
    _backup(path)
    lines = [
        '"Time (year)";"Smelt"',
    ]
    n_steps = 24
    for i in range(n_steps):
        t = i / n_steps
        w = SMELT_SPAWNING_WEIGHTS.get(i, 0.0)
        lines.append(f'"{t:.8f}";"{w:.6f}"')
    path.write_text("\n".join(lines) + "\n")
    print(f"  {path.name}: replaced autumn-spawner weights with spring-spawner (Feb-May)")


def update_fishery_csvs() -> None:
    for name in ("fishery-catchability.csv", "fishery-discards.csv"):
        path = DATA / name
        _backup(path)
        text = path.read_text()
        text = text.replace("gill_whitefish", "gill_smelt")
        # Rename the species row label "whitefish" (standalone at start of line)
        text = re.sub(r"^whitefish,", "smelt,", text, flags=re.MULTILINE)
        path.write_text(text)
        print(f"  {name}: renamed whitefish → smelt, gill_whitefish → gill_smelt")


def update_predation_accessibility() -> None:
    path = DATA / "predation-accessibility.csv"
    _backup(path)
    # Replace the header column and the prey row for sp6.
    # Old row: "whitefish;0.05;0;0;0;0;0;0;0;0;0;0;0;0;0"
    # Smelt is forage fish: eaten heavily by cod/perch/pikeperch, occasional by flounder.
    new_smelt_row = "smelt;0.6;0;0;0.1;0.5;0.6;0;0;0;0;0;0;0;0"

    text = path.read_text()
    # Rename predator column header
    text = text.replace(";whitefish;", ";smelt;")
    # Replace the smelt prey row (was whitefish)
    text = re.sub(
        r"^whitefish;[^\n]*$",
        new_smelt_row,
        text,
        flags=re.MULTILINE,
    )
    # Smelt as predator column (column index 7 in zero-based including header col):
    # The existing whitefish-predator column has non-zero entries for
    # Diatoms(0.5), Dinoflagellates(0.5), Microzooplankton(0.6), Mesozooplankton(0.8),
    # Macrozooplankton(0.4), Benthos(0.4), and stickleback(0.1).
    # For smelt: reduce phytoplankton (smelt is zooplanktivore/piscivore), reduce
    # benthos, keep zooplankton, reduce stickleback (size mismatch).
    # We update whole lines to keep widths consistent.
    predator_col_updates = {
        "cod": "0.05",
        "herring": "0",
        "sprat": "0",
        "flounder": "0",
        "perch": "0",
        "pikeperch": "0",
        "smelt": "0",  # no cannibalism
        "stickleback": "0.05",
        "Diatoms": "0.1",
        "Dinoflagellates": "0.1",
        "Microzooplankton": "0.3",
        "Mesozooplankton": "0.8",
        "Macrozooplankton": "0.6",
        "Benthos": "0.1",
    }
    out_lines = []
    for line in text.splitlines():
        if ";" not in line or line.startswith("v Prey"):
            out_lines.append(line)
            continue
        parts = line.split(";")
        prey = parts[0]
        if prey in predator_col_updates and len(parts) >= 8:
            parts[7] = predator_col_updates[prey]  # column 7 = smelt (was whitefish)
        out_lines.append(";".join(parts))
    path.write_text("\n".join(out_lines) + "\n")
    print(f"  {path.name}: updated smelt prey row + smelt predator column")


def update_biomass_targets() -> None:
    path = DATA / "reference" / "biomass_targets.csv"
    _backup(path)
    text = path.read_text()
    # Old whitefish row: low biomass (~15000 t) and Gulf-of-Bothnia note
    # Smelt is ~2-4× more abundant in the Baltic; use 60000 t as mid estimate
    text = re.sub(
        r"^whitefish,[^\n]*$",
        "smelt,60000,20000,120000,0.3,biomass,Baltic smelt (Osmerus eperlanus) — widely distributed; dominant forage fish in Gulf of Bothnia/Finland",
        text,
        flags=re.MULTILINE,
    )
    path.write_text(text)
    print(f"  {path.name}: replaced whitefish target with smelt (60 kt mid estimate)")


def update_all_parameters_comment() -> None:
    path = DATA / "baltic_all-parameters.csv"
    _backup(path)
    text = path.read_text()
    text = text.replace(
        "8 focal species: Cod, Herring, Sprat, Flounder, Perch, Pike-perch, Whitefish, Stickleback",
        "8 focal species: Cod, Herring, Sprat, Flounder, Perch, Pike-perch, Smelt, Stickleback",
    )
    path.write_text(text)
    print(f"  {path.name}: updated header comment")


def update_fishing_config() -> None:
    path = DATA / "baltic_param-fishing.csv"
    _backup(path)
    text = path.read_text()
    text = text.replace("gill_whitefish", "gill_smelt")
    path.write_text(text)
    print(f"  {path.name}: renamed fishery gill_whitefish → gill_smelt")


def main() -> None:
    print("Swapping sp6: whitefish → smelt…")
    update_species_csv()
    rename_movement_maps()
    update_movement_config()
    update_reproduction_seasonality()
    update_fishery_csvs()
    update_predation_accessibility()
    update_biomass_targets()
    update_all_parameters_comment()
    update_fishing_config()
    print(f"\nBackups created with suffix {BACKUP_SUFFIX}")


if __name__ == "__main__":
    main()
