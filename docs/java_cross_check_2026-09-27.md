# Python ↔ Java cross-engine check — 2026-09-27

First execution of the Python-vs-Java cross-check that the deep-review plan has
carried as unverified since 2026-09-12. **Verdict: `REVIEW`** — 59 of 70
species×metric pairs pass formal equivalence, all 70 stay within 1 order of
magnitude, and the residue concentrates on **two species**.

## Why this was previously thought impossible

The plan recorded the check as blocked because `github.com/osmose-model` returns
**403**. That was measured with `curl`, and it is true of HTTPS *web* fetches — but
this session's git proxy serves **anonymous `git clone`** of public repositories
regardless. The blocker was the instrument, not the network. `curl` reachability is
not a proxy for `git` reachability here; test the operation you actually need.

## Provenance of the JAR — read before quoting these numbers

There is **no official release binary in play**. The jar was built from source:

| | |
|---|---|
| Source | `github.com/osmose-model/osmose` @ `008f74b` (default branch) |
| Version | **4.4.1** (`fr.ird:osmose:4.4.1`), 249 Java files |
| Built with | OpenJDK 21.0.10, Maven, `mvn -B -DskipTests package` |
| Artifact | `osmose-4.4.1-jar-with-dependencies.jar`, 24 MB |

Two deviations from a clean upstream build, both deliberate and both recorded
because they bear on trust in the artifact:

1. **`java/local/ml/options/options/1.0.0/options-1.0.0.jar` had to be recovered
   from the repository's own history.** `.gitattributes` routes `*.jar` through Git
   LFS, the anonymous git lane does not serve LFS objects, and git-lfs is not
   installed — so it cloned as a 130-byte pointer and Maven died on
   `ZipException: zip END header not found`. It is **not** on Maven Central (404).
   Rather than fetch a jar from a third party, it was taken from commit
   `65e3a0ce`, which precedes `34ea8dcb "Move files to LFS"`: a real 12,899-byte
   archive containing exactly the six `ml/options/*` classes `Osmose.java` imports.
   Same repository, same project, just an earlier commit — no third-party download.
2. Maven had **cached the corrupt pointer** into `~/.m2`, so the rebuild kept
   failing after the file on disk was already valid. Clearing that one artifact
   directory fixed it. (Worth knowing generally: a corrupt artifact resolved once
   from a `file:` repository persists in `~/.m2` and survives the source being
   repaired.)

`ml.options` is a command-line option parser used only by `Osmose.java`; it touches
no simulation code. The jar's CLI was verified to accept the `-P<key>=<value>`
overrides the harness relies on.

Only the **4.4.1 arm** ran. The 4.3.3 reference arm that
`cross_engine_parity_440.py` can also report was not built, so there is no
second Java opinion here.

## Method

`scripts/cross_engine_parity_440.py --engines python,4.4.1 --n 16 --years 10
--spinup-years 2`, on `data/eec_full/eec_all-parameters.csv` (14 focal species).
Completed in **830 s**.

Cross-engine RNG streams diverge by construction (NumPy PCG64 vs Java MT19937), so
the comparison is statistical, not bit-exact: per species and metric, TOST
(two one-sided t-tests) on the post-spinup mean over 16 varied-seed reps, log10
scaled. Equivalence margin Δ = **0.48 log10 (3.0×)** for biomass/yield/abundance and
**0.176 (1.5×)** for mean_weight/mean_size.

`d = log10(Python).mean() − log10(Java).mean()` (`cross_engine_parity_440.py:323`),
so **positive d means Python runs higher than Java**.

No species collapsed, no arm came back empty, and nothing was dropped as
unevaluated — so every one of the 70 cells is a real comparison.

## Result by metric

| Metric | Δ | Equivalent | Not established |
|---|---|---|---|
| biomass | 3.0× | 11/14 | lesserSpottedDogfish, mackerel, sardine |
| yield | 3.0× | 11/14 | lesserSpottedDogfish, mackerel, sardine |
| abundance | 3.0× | 12/14 | mackerel, sardine |
| mean_weight | 1.5× | 11/14 | horseMackerel, lesserSpottedDogfish, whiting |
| **mean_size** | 1.5× | **14/14** | — |
| **Total** | | **59/70** | 11 |

**`mean_size` is equivalent for every species**, with a maximum absolute deviation
of 0.08 log10 (1.2×). Size structure — the thing most sensitive to growth,
predation-window and mortality wiring — agrees tightly across the two engines.

## The 11 non-equivalent pairs, separated by cause

`eq=n` from TOST means "equivalence not established", which is **not** the same as
"the engines disagree". It also fires when the confidence interval is too wide to
conclude anything. Separating those two cases is the whole interpretation:

| Metric | Species | d | fold | \|d\|+ci90 | Δ | Reading |
|---|---|---|---|---|---|---|
| yield | lesserSpottedDogfish | +0.52 | 3.31× | 0.66 | 0.48 | **real** — d alone exceeds Δ |
| mean_weight | lesserSpottedDogfish | +0.30 | 2.00× | 0.38 | 0.176 | **real** — d alone exceeds Δ |
| abundance | mackerel | +0.47 | 2.95× | 0.67 | 0.48 | real, marginal |
| biomass | lesserSpottedDogfish | +0.45 | 2.82× | 0.57 | 0.48 | real, marginal |
| biomass | mackerel | +0.36 | 2.29× | 0.53 | 0.48 | real, marginal |
| yield | mackerel | +0.34 | 2.19× | 0.51 | 0.48 | real, marginal |
| mean_weight | whiting | −0.15 | 0.71× | 0.17 | 0.176 | effect *inside* Δ; bound not (see below) |
| biomass | sardine | +0.27 | 1.86× | 0.64 | 0.48 | underpowered (ci ±0.37 > d) |
| yield | sardine | +0.25 | 1.78× | 0.62 | 0.48 | underpowered (ci ±0.37 > d) |
| abundance | sardine | +0.19 | 1.55× | 0.57 | 0.48 | underpowered (ci ±0.38 > d) |
| mean_weight | horseMackerel | +0.08 | 1.20× | 0.24 | 0.176 | underpowered (ci ±0.16 > d) |

So the 11 reduce to:

- **Two species carry a genuine signal.** `lesserSpottedDogfish` (biomass 2.8×,
  yield 3.3×, mean_weight 2.0×) and `mackerel` (biomass 2.3×, yield 2.2×,
  abundance 3.0×), Python higher in every case. These are the ones worth
  investigating.
- **`whiting` mean_weight** fails equivalence without its *effect* being over the
  margin. The point estimate, 0.71× (a 1.41× reduction, Python *lower* — the only
  one of the eleven in that direction), sits **inside** the 1.5× margin; what lands
  past Δ is the equivalence *bound* |d|+ci90. Its ci90 of ±0.02 is the tightest of
  the eleven, so unlike the four below this is a well-measured small effect rather
  than an unresolved one. Note the printed 2dp values (0.15 + 0.02 = 0.17) appear
  inside Δ = 0.176: the `eq=n` verdict comes from the harness's unrounded
  arithmetic, and the table as printed cannot be used to re-derive it. Do not read
  this row as "Python's whiting are 1.5× off".
- **Four are verdicts about statistical power, not about the engines.**
  `sardine` (all three) and `horseMackerel` mean_weight have confidence intervals
  wider than their effects — `sardine`'s ±0.37–0.38 and `horseMackerel`'s ±0.16–0.31
  mark them as the high-inter-seed-variance species on this config. More reps would
  settle them; 16 does not.

Do not read the four underpowered rows as divergence, and do not read them as
agreement either. They were not resolved.

## Relationship to the documented "within 1 OoM" claim

CLAUDE.md states cross-engine equivalence as "within 1 OoM (14/14 EEC)". **That
claim holds**: the largest deviation anywhere is 0.52 log10 (3.3×), comfortably
inside one order of magnitude, and the harness's catastrophic-divergence tripwire
did not fire for any species. The `REVIEW` verdict here comes from the *stricter*
TOST criterion at 3×, which is a different and tighter question than the 1-OoM
tripwire. The two are not in conflict; a reader who knows only the 1-OoM figure
should not be surprised by this, but also should not assume it implies 3×
equivalence.

## Not fixed: `scripts/validate_engines.py` cannot run at all on 4.4.1

This was the script the plan named for the cross-check, and it fails before
simulating. For `NETCDF_BIOMASS` forcing, 4.4.1 reads `species.biomass.file.spN` /
`.varname.spN` (prefix chosen at `ResourceForcing.java:214`, thrown at `:244`),
while the bundled `data/examples` Bay of Biscay config supplies the 4.3.x
`species.file.spN`. The script hands Java the raw config with no 4.4.x key
migration, even though `osmose/config/aliases.py:_emit_resource_biomass_forcing`
already synthesizes exactly those keys for a ≥4.4.0 target.
`cross_engine_parity_440.py` stages properly, which is why it runs and this does not.

Needs a decision rather than a guess: migrate the config when staging for Java, or
declare the script 4.3.3-only. Its failure *reporting* was fixed separately
(`446809bd`) — it had been printing SLF4J noise instead of the actual exception.

Related: the `species.biomass.nsteps.year` staging documented in
`cross_engine_parity_440.py` is a **different** failure — it only applies once
`.file.spN` resolves — and its docstring attributes the requirement to Java 4.3.3
when 4.4.1 enforces it identically.

## Reproducing

The jar is gitignored (`.gitignore:70`) and the clone is outside the repo, so
neither is committed. To redo this from scratch:

```
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/osmose-model/osmose /tmp/osmose-src
git -C /tmp/osmose-src checkout --detach 008f74b
git -C /tmp/osmose-src checkout 65e3a0ce -- java/local/ml/options/options/1.0.0/options-1.0.0.jar
rm -rf ~/.m2/repository/ml/options/options/1.0.0
mvn -B -DskipTests -f /tmp/osmose-src/pom.xml package
cp /tmp/osmose-src/inst/java/osmose-4.4.1-jar-with-dependencies.jar osmose-java/
PYTHONPATH=. .venv/bin/python scripts/cross_engine_parity_440.py \
  --engines python,4.4.1 --n 16 --years 10 --spinup-years 2
```

Every line above the build is load-bearing, and three of them are easy to get wrong:

- **Line 2 pins the source.** The original run cloned the default branch, where
  `008f74b` merely happened to be HEAD on 2026-09-27. Omit this and a later reader
  builds whatever the branch has advanced to, against provenance that claims
  `008f74b` — silently comparing the Python engine to a different Java engine than
  the one these numbers came from.
- **The clone is deliberately not `--depth 1`.** Both pinned commits must be
  reachable, and a shallow clone of a moving branch may contain neither.
- **Line 3 restores one FILE, not a tree**, and must come after line 2. It is
  `checkout <commit> -- <path>`, which leaves HEAD at `008f74b` (verified: HEAD
  reads `008f74ba` afterwards). Dropping it does not "avoid building a different
  revision" — it makes the build fail outright on the LFS pointer.
- **Line 4 is not optional.** Maven caches the corrupt pointer under `~/.m2` on the
  first failed attempt, so the build keeps failing with the same `ZipException`
  after the working tree is already correct.
