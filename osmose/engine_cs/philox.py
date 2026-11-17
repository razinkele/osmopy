"""The C# engine's random numbers, reproduced in Python (spec section 5.3, plan task 0.5).

The C# engine draws every number from Philox4x64-10, which is also what ``numpy.random.Philox``
implements, so Python can compute any word the engine draws. Later tests use this to feed the
Numba oracle kernel the same permutations the C# kernel gets (spec section 10.1). Pure NumPy;
importing it never loads the CLR.

**The NumPy call.** ``numpy.random.Philox`` increments its 256-bit counter *before* generating
a block, so the block for counter ``c`` comes from ``Philox(counter=c - 1, key=k)`` followed by
``random_raw(4)``. Counter and key are little-endian multi-word integers: ``c0`` and ``K0`` are
the least significant words.
"""

from __future__ import annotations

import numpy as np

SCHEME = 1
DOMAIN_CONSTANT = 0x4F534D4F53450000  # "OSMOSE\0\0", XOR-ed with the scheme into K1
_M64 = (1 << 64) - 1


def splitmix64(x: int) -> int:
    """Vigna's SplitMix64 finaliser (``Philox.SplitMix64`` in C#)."""
    z = (x + 0x9E3779B97F4A7C15) & _M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return z ^ (z >> 31)


def run_key(seed: int, scheme: int = SCHEME) -> tuple[int, int]:
    """The engine's key for a run (``PhiloxKey.ForRun``): ``(SplitMix64(seed), DOMAIN ^ scheme)``."""
    return splitmix64(seed & _M64), DOMAIN_CONSTANT ^ (scheme & 0xFFFFFFFF)


def step_word(step: int, substep: int) -> int:
    """The counter's third word: ``(step << 16) | substep`` (``PhiloxStream.StepWord``)."""
    if step < 0 or not 0 <= substep <= 0xFFFF:
        raise ValueError(f"step must be >= 0 and substep in [0, 65535], got {step}, {substep}")
    return (step << 16) | substep


def block(key: tuple[int, int], c0: int, c1: int, c2: int, c3: int) -> list[int]:
    """The four 64-bit words of the block for counter ``(c0, c1, c2, c3)``."""
    counter = (c0 | c1 << 64 | c2 << 128 | c3 << 192) - 1  # NumPy increments before generating
    gen = np.random.Philox(counter=counter % (1 << 256), key=key[0] | key[1] << 64)
    return [int(w) for w in gen.random_raw(4)]
