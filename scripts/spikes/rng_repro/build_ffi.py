# scripts/spikes/rng_repro/build_ffi.py
"""Compile mt19937.c into cffi modules: portable (-O3) and native (-O3 -march=native)."""
from __future__ import annotations

from pathlib import Path

from cffi import FFI

HERE = Path(__file__).resolve().parent
CDEF = """
void cell_rng(int64_t seed, int n, int32_t* out_pred, int32_t* out_starv,
              int32_t* out_fish, int32_t* out_nat, int32_t* out_orders);
void cell_rng_bench(int64_t seed, int n, int n_iter, int32_t* out_pred, int32_t* out_starv,
                    int32_t* out_fish, int32_t* out_nat, int32_t* out_orders);
"""


def build(variant: str) -> str:
    ffi = FFI()
    ffi.cdef(CDEF)
    flags = ["-O3"] if variant == "portable" else ["-O3", "-march=native"]
    ffi.set_source(f"_rng_{variant}", '#include <stdint.h>\n'
                   + (HERE / "mt19937.c").read_text(), extra_compile_args=flags)
    return ffi.compile(tmpdir=str(HERE))


if __name__ == "__main__":
    for v in ("portable", "native"):
        print(v, "->", build(v))
