"""The parameter bundle (plan task 0.3).

The pure-Python half always runs. The cross-language half needs the built engine and runs in a
probe subprocess (the CLR never enters the pytest process). It skips without the engine, unless
``OSMOSE_REQUIRE_CSHARP=1`` turns the skip into a failure. Acceptance:
- a bundle written by Python, loaded by C# and saved again, is byte-identical;
- a bundle C# builds from Python's NumPy addresses saves byte-identical to Python's own file.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from osmose.engine_cs import csharp_engine_available
from osmose.engine_cs.bundle import HEADER_SIZE, RECORD_SIZE, Bundle, BundleError

ROOT = Path(__file__).resolve().parents[1]
REQUIRE = os.environ.get("OSMOSE_REQUIRE_CSHARP") == "1"


def sample() -> Bundle:
    b = Bundle(contract_version=1)
    b.set_int("time.n_steps", 240)
    b.set_float("run.deadline_seconds", 12.5)
    b.set_array("growth.linf", np.array([10.0, 20.5, 33.25]))
    b.set_array("grid.ocean_mask", np.array([[1, 0, 1], [1, 0, 1]], dtype=np.uint8))
    b.set_array("species.n_schools", np.array([20, 30], dtype=np.int32))
    b.set_array("movement.maps.index.offsets", np.array([0, 5, 9], dtype=np.int64))
    b.set_array("resources.forcing", np.arange(24.0).reshape(2, 3, 4))
    b.set_array("state.empty", np.zeros(0))
    return b


def test_python_round_trip(tmp_path):
    p = tmp_path / "a.bin"
    sample().save(p)
    loaded = Bundle.load(p)
    assert loaded.contract_version == 1
    assert loaded.names() == sample().names()
    assert loaded.get("time.n_steps") == 240
    assert loaded.get("run.deadline_seconds") == 12.5
    np.testing.assert_array_equal(loaded.get("resources.forcing"), np.arange(24.0).reshape(2, 3, 4))
    assert loaded.get("grid.ocean_mask").dtype == np.uint8
    assert loaded.get("state.empty").shape == (0,)
    loaded.save(tmp_path / "b.bin")
    assert (tmp_path / "a.bin").read_bytes() == (tmp_path / "b.bin").read_bytes()


def test_file_is_canonical_whatever_the_insertion_order(tmp_path):
    a, b = sample(), Bundle(1)
    for name in reversed(a.names()):
        v = a.get(name)
        if isinstance(v, np.ndarray):
            b.set_array(name, v)
        elif isinstance(v, int):
            b.set_int(name, v)
        else:
            b.set_float(name, v)
    a.save(tmp_path / "a.bin")
    b.save(tmp_path / "b.bin")
    assert (tmp_path / "a.bin").read_bytes() == (tmp_path / "b.bin").read_bytes()


def test_arrays_are_64_byte_aligned_in_the_file(tmp_path):
    sample().save(tmp_path / "a.bin")
    raw = (tmp_path / "a.bin").read_bytes()
    count = struct.unpack_from("<I", raw, 16)[0]
    for i in range(count):
        kind, offset = raw[HEADER_SIZE + i * RECORD_SIZE + 64], 0
        if kind == 3:
            offset = struct.unpack_from("<q", raw, HEADER_SIZE + i * RECORD_SIZE + 104)[0]
        assert offset % 64 == 0


@pytest.mark.parametrize("bad", ["", "Growth.linf", "growth..linf", "a-b", "x" * 64])
def test_invalid_names_rejected(bad):
    with pytest.raises(BundleError):
        Bundle(1).set_int(bad, 1)


def test_no_silent_casts_or_duplicates():
    b = Bundle(1)
    with pytest.raises(BundleError, match="float32"):
        b.set_array("a.x", np.zeros(3, dtype=np.float32))
    with pytest.raises(BundleError, match="bool"):
        b.set_array("a.y", np.zeros(3, dtype=bool))
    with pytest.raises(BundleError):
        b.set_int("a.z", 1.5)  # type: ignore[arg-type]
    with pytest.raises(BundleError):
        b.set_int("a.z", True)
    with pytest.raises(BundleError):
        b.set_array("a.w", np.zeros((1, 1, 1, 1, 1)))
    b.set_int("a.k", 1)
    with pytest.raises(BundleError, match="already set"):
        b.set_float("a.k", 2.0)


def test_big_endian_input_is_stored_little_endian(tmp_path):
    b = Bundle(1)
    b.set_array("a.x", np.array([1.5, 2.5], dtype=">f8"))
    b.save(tmp_path / "a.bin")
    np.testing.assert_array_equal(Bundle.load(tmp_path / "a.bin").get("a.x"), [1.5, 2.5])


def test_load_lists_every_corrupt_record(tmp_path):
    p = tmp_path / "a.bin"
    sample().save(p)
    raw = bytearray(p.read_bytes())
    names = sample().names()
    rec = {n: HEADER_SIZE + i * RECORD_SIZE for i, n in enumerate(names)}
    raw[rec["grid.ocean_mask"] + 65] = 9  # unknown dtype
    raw[rec["growth.linf"] + 112] = 99  # byte length disagrees
    raw[rec["species.n_schools"] + 104] += 4  # misaligned
    p.write_bytes(raw)
    with pytest.raises(BundleError) as exc:
        Bundle.load(p)
    msg = str(exc.value)
    assert "3 problem(s)" in msg
    for n in ("grid.ocean_mask", "growth.linf", "species.n_schools"):
        assert n in msg


def test_load_rejects_a_non_bundle(tmp_path):
    p = tmp_path / "junk.bin"
    p.write_bytes(b"\0" * 100)
    with pytest.raises(BundleError, match="bad magic"):
        Bundle.load(p)


@pytest.fixture(scope="module")
def cross(tmp_path_factory) -> dict:
    ok, why = csharp_engine_available()
    if not ok:
        if REQUIRE:
            pytest.fail(f"OSMOSE_REQUIRE_CSHARP=1 but the C# engine is unavailable: {why}")
        pytest.skip(
            f"C# engine unavailable ({why}). To run: bash scripts/build_csharp_engine.sh, "
            "then OSMOSE_REQUIRE_CSHARP=1 pytest tests/test_csharp_bundle.py"
        )
    d = tmp_path_factory.mktemp("bundle")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "_csharp_bundle_probe.py"), str(d)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stderr[-4000:]}"
    return {"dir": d, **json.loads(proc.stdout.strip().splitlines()[-1])}


def test_python_file_loaded_and_saved_by_csharp_is_byte_identical(cross):
    d = cross["dir"]
    assert (d / "python.bin").read_bytes() == (d / "csharp_roundtrip.bin").read_bytes()


def test_csharp_bundle_from_numpy_addresses_matches_python_file(cross):
    d = cross["dir"]
    assert (d / "python.bin").read_bytes() == (d / "csharp_from_addresses.bin").read_bytes()


def test_csharp_reads_the_values_python_wrote(cross):
    assert cross["values"] == {
        "time.n_steps": 240,
        "run.deadline_seconds": 12.5,
        "growth.linf": [10.0, 20.5, 33.25],
        "grid.ocean_mask.shape": [2, 3],
    }
