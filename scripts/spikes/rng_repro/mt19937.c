/* scripts/spikes/rng_repro/mt19937.c
 * NumPy-legacy MT19937 + init_genrand scalar seeding + rk_interval masked-rejection
 * bounded integers + Fisher-Yates permutation/shuffle. Targets CPython legacy
 * RandomState (== Numba np.random, per spec 0a). Parity gate (Task 3) validates.
 */
#include <stdint.h>

#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL
#define UPPER_MASK 0x80000000UL
#define LOWER_MASK 0x7fffffffUL

typedef struct { uint32_t mt[N]; int mti; } mt_state;

/* NumPy-legacy RandomState(scalar) and Numba np.random.seed(scalar) seed the MT state via
 * the SINGLE-INTEGER path init_genrand(seed) — NOT init_by_array. (Verified empirically:
 * RandomState(0)'s first uint32 == init_genrand(0) == 2357136044, whereas init_by_array({0})
 * gives a different stream.) cell_rng therefore seeds with init_genrand. */
static void init_genrand(mt_state* s, uint32_t seed) {
    s->mt[0] = seed;
    for (int i = 1; i < N; i++)
        s->mt[i] = (uint32_t)(1812433253UL * (s->mt[i-1] ^ (s->mt[i-1] >> 30)) + (uint32_t)i);
    s->mti = N;
}

static uint32_t genrand_uint32(mt_state* s) {
    uint32_t y;
    static const uint32_t mag01[2] = {0x0UL, MATRIX_A};
    if (s->mti >= N) {
        int kk;
        for (kk = 0; kk < N - M; kk++) {
            y = (s->mt[kk] & UPPER_MASK) | (s->mt[kk+1] & LOWER_MASK);
            s->mt[kk] = s->mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (; kk < N - 1; kk++) {
            y = (s->mt[kk] & UPPER_MASK) | (s->mt[kk+1] & LOWER_MASK);
            s->mt[kk] = s->mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        y = (s->mt[N-1] & UPPER_MASK) | (s->mt[0] & LOWER_MASK);
        s->mt[N-1] = s->mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];
        s->mti = 0;
    }
    y = s->mt[s->mti++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);
    return y;
}

/* NumPy-legacy rk_interval: returns a uniform integer in [0, max] via masked rejection. */
static uint32_t rk_interval(uint32_t max, mt_state* s) {
    if (max == 0) return 0;
    uint32_t mask = max;
    mask |= mask >> 1; mask |= mask >> 2; mask |= mask >> 4;
    mask |= mask >> 8; mask |= mask >> 16;
    uint32_t value;
    while ((value = (genrand_uint32(s) & mask)) > max) { }
    return value;
}

/* NumPy-legacy Fisher-Yates: for i = n-1 down to 1, j = rk_interval(i), swap(arr[i], arr[j]). */
static void fisher_yates(int32_t* arr, int n, mt_state* s) {
    for (int i = n - 1; i > 0; i--) {
        uint32_t j = rk_interval((uint32_t)i, s);
        int32_t t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
}

void cell_rng(int64_t seed, int n, int32_t* out_pred, int32_t* out_starv,
              int32_t* out_fish, int32_t* out_nat, int32_t* out_orders) {
    mt_state s;
    uint32_t key = (uint32_t)(seed & 0xFFFFFFFF);  /* uint32 reduction (spec 0a) */
    init_genrand(&s, key);                          /* scalar seeding (matches RandomState/Numba) */

    int32_t* outs[4] = {out_pred, out_starv, out_fish, out_nat};
    for (int p = 0; p < 4; p++) {
        for (int i = 0; i < n; i++) outs[p][i] = i;   /* arange(n) */
        fisher_yates(outs[p], n, &s);                 /* permutation */
    }
    int32_t causes[4] = {0, 1, 2, 3};                 /* created ONCE, carried over */
    for (int i = 0; i < n; i++) {
        fisher_yates(causes, 4, &s);                  /* shuffle in place */
        out_orders[i*4 + 0] = causes[0];
        out_orders[i*4 + 1] = causes[1];
        out_orders[i*4 + 2] = causes[2];
        out_orders[i*4 + 3] = causes[3];
    }
}
