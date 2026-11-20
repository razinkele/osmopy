"""Allele pools must not depend on the process (#344).

``TraitRegistry.from_config`` seeded each trait's pool with ``hash(name)``. Python randomises the
hash of a string per process (PYTHONHASHSEED), so a genetics run drew different pools in each
process, whatever its seed. The seed is now a stable hash of the trait name.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from osmose.engine.genetics.trait import TraitRegistry

ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "evolution.trait.imax.target": "ingestion_rate",
    "evolution.trait.imax.mean.sp0": "3.5",
    "evolution.trait.imax.var.sp0": "0.1",
    "evolution.trait.imax.nlocus.sp0": "5",
    "evolution.trait.imax.nval.sp0": "20",
    "evolution.trait.m0.target": "bioen_m0",
    "evolution.trait.m0.mean.sp0": "10.0",
    "evolution.trait.m0.var.sp0": "1.0",
    "evolution.trait.m0.nlocus.sp0": "4",
    "evolution.trait.m0.nval.sp0": "12",
}


def _digest(registry: TraitRegistry) -> str:
    h = hashlib.sha256()
    for name, trait in registry.traits.items():
        h.update(name.encode())
        for pools in trait.allele_pool:
            for values in pools:
                h.update(np.asarray(values, dtype=float).tobytes())
    return h.hexdigest()


def test_the_pools_are_the_same_in_every_process():
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from tests.test_genetics_allele_seed import CONFIG, _digest; "
        "from osmose.engine.genetics.trait import TraitRegistry; "
        "print(_digest(TraitRegistry.from_config(CONFIG, 1)))"
    )
    digests = set()
    for hash_seed in ("0", "1", "12345"):
        env = os.environ | {"PYTHONHASHSEED": hash_seed}
        out = subprocess.run(
            [sys.executable, "-c", code, str(ROOT)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        digests.add(out.stdout.strip())
    assert len(digests) == 1, digests
    # ...and they are this process's too.
    assert digests == {_digest(TraitRegistry.from_config(CONFIG, 1))}


def test_two_traits_get_different_pools():
    registry = TraitRegistry.from_config(CONFIG, 1)
    imax = registry.traits["imax"].allele_pool[0][0]
    m0 = registry.traits["m0"].allele_pool[0][0]
    # Scale-free comparison: the first values of each pool, standardised by their own sd.
    assert not np.allclose(imax[:4] / imax.std(), m0[:4] / m0.std())
