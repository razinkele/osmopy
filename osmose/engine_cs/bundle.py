"""The parameter bundle on the Python side (spec section 8.2, plan task 0.3).

A :class:`Bundle` is a set of named scalars and arrays, written and read in the same canonical
file format as the C# ``SimulationConfig`` (documented there), so a file either side writes is
byte-identical to the other's. :meth:`Bundle.to_csharp` hands the arrays to the engine by address,
with no copy. Pure NumPy: importing this never loads the CLR.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = b"OSMBNDL1"
FILE_FORMAT_VERSION = 1
HEADER_SIZE = 64
RECORD_SIZE = 128
ALIGNMENT = 64
MAX_NAME_LENGTH = 63

KIND_INT, KIND_FLOAT, KIND_ARRAY = 1, 2, 3
# On-disk dtype codes and the C# DType member each maps to.
DTYPES: dict[np.dtype, tuple[int, str]] = {
    np.dtype("<f8"): (1, "F8"),
    np.dtype("<i4"): (2, "I4"),
    np.dtype("<i8"): (3, "I8"),
    np.dtype("u1"): (4, "U1"),
}
_BY_CODE = {code: dt for dt, (code, _) in DTYPES.items()}
_NAME = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)*$")


class BundleError(ValueError):
    """A bundle entry or file is malformed; the message lists every problem found."""


def _align(n: int) -> int:
    return (n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def _check_name(name: str) -> str:
    if not (0 < len(name) <= MAX_NAME_LENGTH and _NAME.match(name)):
        raise BundleError(
            f"invalid entry name {name!r}: lowercase dot-separated [a-z0-9_] words, "
            f"at most {MAX_NAME_LENGTH} bytes"
        )
    return name


class Bundle:
    """Named scalars and arrays for one ``RunSimulation`` call.

    Arrays must already have one of the four bundle dtypes (float64, int32, int64, uint8): the
    bundle never casts silently. They are stored C-contiguous and little-endian, copied only if
    the input was not.
    """

    def __init__(self, contract_version: int) -> None:
        self.contract_version = int(contract_version)
        self._entries: dict[str, tuple[int, Any]] = {}
        self._mmap: np.memmap | None = None

    def __len__(self) -> int:
        return len(self._entries)

    def names(self) -> list[str]:
        """Entry names in canonical (sorted) order."""
        return sorted(self._entries)

    def _add(self, name: str, kind: int, value: Any) -> None:
        if _check_name(name) in self._entries:
            raise BundleError(f"entry {name!r} is already set")
        self._entries[name] = (kind, value)

    def set_int(self, name: str, value: int) -> None:
        if isinstance(value, bool | float) or not isinstance(value, int | np.integer):
            raise BundleError(f"{name}: expected an integer, got {type(value).__name__}")
        if not -(2**63) <= int(value) < 2**63:
            raise BundleError(f"{name}: {value} does not fit in int64")
        self._add(name, KIND_INT, int(value))

    def set_float(self, name: str, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, int | float | np.floating | np.integer):
            raise BundleError(f"{name}: expected a float, got {type(value).__name__}")
        self._add(name, KIND_FLOAT, float(value))

    def set_array(self, name: str, array: np.ndarray) -> None:
        if not isinstance(array, np.ndarray):
            raise BundleError(f"{name}: expected a NumPy array, got {type(array).__name__}")
        dt = array.dtype.newbyteorder("<") if array.dtype.byteorder == ">" else array.dtype
        if dt not in DTYPES:
            raise BundleError(
                f"{name}: dtype {array.dtype} is not one of float64, int32, int64, uint8"
            )
        if not 1 <= array.ndim <= 4:
            raise BundleError(f"{name}: {array.ndim} dimensions, a bundle array has 1 to 4")
        self._add(name, KIND_ARRAY, np.ascontiguousarray(array, dtype=dt))

    def get(self, name: str) -> int | float | np.ndarray:
        return self._entries[name][1]

    def save(self, path: str | Path) -> None:
        """Write the canonical file (format documented in the C# ``SimulationConfig``)."""
        names = self.names()
        data_offset = _align(HEADER_SIZE + len(names) * RECORD_SIZE)
        offsets: dict[str, int] = {}
        cursor = data_offset
        for name in names:
            kind, value = self._entries[name]
            if kind == KIND_ARRAY:
                offsets[name] = cursor
                cursor = _align(cursor + value.nbytes)

        header = bytearray(HEADER_SIZE)
        header[:8] = MAGIC
        struct.pack_into(
            "<IiIIQQ", header, 8, FILE_FORMAT_VERSION, self.contract_version, len(names), 0,
            HEADER_SIZE, data_offset,
        )  # fmt: skip
        with open(path, "wb") as f:
            f.write(header)
            for name in names:
                kind, value = self._entries[name]
                rec = bytearray(RECORD_SIZE)
                rec[: len(name)] = name.encode("ascii")
                if kind == KIND_ARRAY:
                    dims = list(value.shape) + [0] * (4 - value.ndim)
                    struct.pack_into("<BBB", rec, 64, kind, DTYPES[value.dtype][0], value.ndim)
                    struct.pack_into("<4qqq", rec, 72, *dims, offsets[name], value.nbytes)
                else:
                    bits = (
                        value
                        if kind == KIND_INT
                        else struct.unpack("<q", struct.pack("<d", value))[0]
                    )
                    struct.pack_into("<B", rec, 64, kind)
                    struct.pack_into("<q", rec, 104, bits)
                f.write(rec)
            for name in names:
                kind, value = self._entries[name]
                if kind == KIND_ARRAY:
                    f.write(b"\0" * (offsets[name] - f.tell()))
                    value.tofile(f)
            f.write(b"\0" * (cursor - f.tell()))

    @classmethod
    def load(cls, path: str | Path) -> Bundle:
        """Read a bundle file. Arrays are read-only views of a memory map; every structural
        problem is reported at once."""
        raw = np.memmap(path, dtype=np.uint8, mode="r")
        problems: list[str] = []
        if len(raw) < HEADER_SIZE or bytes(raw[:8]) != MAGIC:
            raise BundleError(f"{path}: not a bundle file (too short or bad magic)")
        fmt, contract, count, _, table, data_offset = struct.unpack_from("<IiIIQQ", raw, 8)
        if fmt != FILE_FORMAT_VERSION:
            raise BundleError(f"{path}: file format {fmt}, this reader reads {FILE_FORMAT_VERSION}")
        table_end = table + count * RECORD_SIZE
        if table != HEADER_SIZE or table_end > len(raw) or data_offset < table_end:
            raise BundleError(f"{path}: table of {count} entries does not fit")
        bundle = cls(contract)
        previous = None
        for i in range(count):
            rec = raw[table + i * RECORD_SIZE : table + (i + 1) * RECORD_SIZE]
            name = bytes(rec[:64]).split(b"\0", 1)[0].decode("ascii", "replace")
            where = f"entry {i} ({name})"
            if not (0 < len(name) <= MAX_NAME_LENGTH and _NAME.match(name)):
                problems.append(f"{where}: invalid name")
                continue
            if previous is not None and previous >= name:
                problems.append(f"{where}: table not sorted or duplicated after {previous}")
            previous = name
            kind, dcode, rank = struct.unpack_from("<BBB", rec, 64)
            *dims, bits, nbytes = struct.unpack_from("<4qqq", rec, 72)
            if kind in (KIND_INT, KIND_FLOAT):
                value: Any = (
                    bits if kind == KIND_INT else struct.unpack("<d", struct.pack("<q", bits))[0]
                )
                bundle._entries[name] = (kind, value)
                continue
            if kind != KIND_ARRAY or dcode not in _BY_CODE or not 1 <= rank <= 4:
                problems.append(f"{where}: bad kind {kind}, dtype {dcode} or rank {rank}")
                continue
            dt = _BY_CODE[dcode]
            shape = tuple(dims[:rank])
            expected = int(np.prod(shape, dtype=np.int64)) * dt.itemsize
            if min(shape) < 0 or nbytes != expected:
                problems.append(
                    f"{where}: shape {shape} needs {expected} bytes, table says {nbytes}"
                )
                continue
            if bits % ALIGNMENT or bits < data_offset or bits + nbytes > len(raw):
                problems.append(f"{where}: data at offset {bits} is misaligned or outside the file")
                continue
            if expected == 0:
                arr = np.empty(shape, dtype=dt)
            else:
                arr = np.frombuffer(raw, dtype=dt, count=expected // dt.itemsize, offset=bits)
            bundle._entries[name] = (KIND_ARRAY, arr.reshape(shape))
        if problems:
            raise BundleError(
                f"{path}: {len(problems)} problem(s):\n  - " + "\n  - ".join(problems)
            )
        bundle._mmap = raw
        return bundle

    def to_csharp(self, engine: Any) -> Any:
        """Build the engine's ``SimulationConfig`` over this bundle's buffers, by address.

        The arrays stay owned by this bundle: keep it alive until the engine call has returned
        (spec section 8.5, buffer lifetime).
        """
        config = engine.SimulationConfig(self.contract_version)
        dtype_enum = engine.DType
        for name in self.names():
            kind, value = self._entries[name]
            if kind == KIND_INT:
                config.SetInt(name, value)
            elif kind == KIND_FLOAT:
                config.SetFloat(name, value)
            else:
                dims = list(value.shape) + [-1] * (4 - value.ndim)
                member = getattr(dtype_enum, DTYPES[value.dtype][1])
                config.SetArray(name, member, value.ctypes.data if value.size else 0, *dims)
        return config
