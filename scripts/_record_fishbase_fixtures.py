"""Record tiny FishBase/SeaLifeBase parquet slices for tests (re-runnable).

Pulls only the rows for Gadus morhua (FishBase) and Carcinus maenas (SeaLifeBase)
so fixtures stay small. Run: .venv/bin/python scripts/_record_fishbase_fixtures.py
Requires network (one-off); CI never runs this.
"""
from pathlib import Path

from osmose import fishbase

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "fishbase"
OUT.mkdir(parents=True, exist_ok=True)
TARGETS = {"fb": ("Gadus", "morhua"), "slb": ("Carcinus", "maenas")}
TABLES = ["species", "popgrowth", "poplw", "maturity"]


def main() -> None:
    for db, (genus, species) in TARGETS.items():
        sp = fishbase._load_table("species", db)
        code = int(sp[(sp.Genus == genus) & (sp.Species == species)].SpecCode.iloc[0])
        for table in TABLES:
            df = fishbase._load_table(table, db)
            col = "Speccode" if "Speccode" in df.columns else "SpecCode"
            if table == "species":
                slice_ = df[(df.Genus == genus) & (df.Species == species)]
            else:
                slice_ = df[df[col] == code]
            slice_.to_parquet(OUT / f"{db}_{table}.parquet")
            print(f"{db}/{table}: {len(slice_)} rows (code={code})")


if __name__ == "__main__":
    main()
