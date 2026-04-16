"""Java-compatible Xorshift random number generator.

Replicates OSMOSE Java's ``XSRandom`` (extends ``java.util.Random``) for
bit-exact parity when both engines use fixed seeds.  The Xorshift state
uses 64-bit signed arithmetic matching Java's ``long`` type.

Java source: ``fr.ird.osmose.util.XSRandom``

Usage::

    from osmose.engine.rng_compat import XorshiftRNG
    rng = XorshiftRNG(seed=42)
    rng.next_int(10)    # uniform int in [0, 10)
    rng.next_double()   # uniform float in [0, 1)
    rng.permutation(5)  # Fisher-Yates shuffle matching Java's shuffleArray

Note: seed=0 is a fixed point (all outputs are 0). Use a nonzero seed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_MASK64 = (1 << 64) - 1


def _to_signed64(x: int) -> int:
    """Convert to Java's signed 64-bit long representation."""
    x &= _MASK64
    if x >= (1 << 63):
        x -= 1 << 64
    return x


class XorshiftRNG:
    """Java XSRandom-compatible Xorshift PRNG.

    Matches Java's ``protected int next(int nbits)`` exactly, which is the
    primitive underlying ``nextInt()``, ``nextDouble()``, and ``nextFloat()``.

    Verified against Java reference output for seeds 1, 42, and large values.
    """

    def __init__(self, seed: int) -> None:
        self._seed = _to_signed64(seed)

    def _next(self, nbits: int) -> int:
        """Java's ``next(int nbits)`` — advance state and return nbits bits.

        Java source::

            long x = seed;
            x ^= (x << 21);
            x ^= (x >>> 35);  // unsigned right shift
            x ^= (x << 4);
            seed = x;
            x &= ((1L << nbits) - 1);
            return (int) x;
        """
        x = self._seed
        # x ^= (x << 21)
        x = _to_signed64(x ^ ((x << 21) & _MASK64))
        # x ^= (x >>> 35)  — Java's >>> is logical (unsigned) right shift
        x = _to_signed64(x ^ ((x & _MASK64) >> 35))
        # x ^= (x << 4)
        x = _to_signed64(x ^ ((x << 4) & _MASK64))
        self._seed = x
        return int(x & ((1 << nbits) - 1))

    def next_int(self, n: int) -> int:
        """Uniform random int in [0, n), matching ``java.util.Random.nextInt(n)``."""
        if n <= 0:
            raise ValueError(f"bound must be positive, got {n}")
        if n & (n - 1) == 0:  # power of two
            return (n * self._next(31)) >> 31
        while True:
            bits = self._next(31)
            val = bits % n
            if bits - val + (n - 1) >= 0:
                return val

    def next_double(self) -> float:
        """Uniform random float in [0, 1), matching ``java.util.Random.nextDouble()``."""
        return ((self._next(26) << 27) + self._next(27)) / float(1 << 53)

    def next_float(self) -> float:
        """Uniform random float in [0, 1), matching ``java.util.Random.nextFloat()``."""
        return self._next(24) / float(1 << 24)

    def permutation(self, n: int) -> NDArray[np.int32]:
        """Fisher-Yates shuffle matching Java's ``MortalityProcess.shuffleArray``.

        Java iterates ``for (int i = a.length; i > 1; i--)`` and swaps
        ``a[i-1]`` with ``a[random.nextInt(i)]``.
        """
        perm = np.arange(n, dtype=np.int32)
        for i in range(n, 1, -1):
            j = self.next_int(i)
            perm[i - 1], perm[j] = perm[j], perm[i - 1]
        return perm

    def shuffle(self, arr: list | NDArray) -> None:
        """In-place Fisher-Yates shuffle matching Java's shuffleArray."""
        n = len(arr)
        for i in range(n, 1, -1):
            j = self.next_int(i)
            arr[i - 1], arr[j] = arr[j], arr[i - 1]
