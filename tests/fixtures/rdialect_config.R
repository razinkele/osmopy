# Synthetic R-dialect OSMOSE config — hand-written for tests, NOT vendored upstream content.
# Mirrors the shape of osmose-model/osmose-ben's osmose-ben.R: `key = value` lines in a .R file.
#
# Provenance for the real keys this mirrors (verified 2026-07-17):
#   economy.enabled             -> osmose-ben/osmose-ben_v4.x_develop/osmose-ben.R:1048
#   surveys.name.sr1            -> osmose-ben/osmose-ben_v4.x_develop/osmose-ben.R (surveys block)
#   simulation.restart.enabled  -> osmose-ben/osmose-ben_v4.x_develop/osmose-ben.R
#
# VALUES HERE ARE CHOSEN FOR TEST COVERAGE AND DO NOT MIRROR THE CORPUS.
# Notably: the real osmose-ben.R:1048 is `economy.enabled = FALSE`. We set TRUE so the
# migration produces an observable value. Do NOT read this fixture as evidence that real
# R configs enable economics -- none do, which is exactly why that trap is LATENT and
# output.tl.enabled is the guide's headline instead.

simulation.nspecies = 2
species.name.sp0 = anchovy
species.name.sp1 = sardine

fisheries.check.enabled = FALSE
output.weight.enabled = TRUE

# pre-4.4.0 key that the 4.4.0 compat shim migrates.
# Real corpus value is FALSE (osmose-ben.R:1048); TRUE here only to make the migration visible.
economy.enabled = TRUE

# unsupported module -> strict validation MUST report these as unknown
surveys.enabled.sr1 = TRUE
surveys.name.sr1 = acousticSurvey

# allowlisted-but-unread -> strict validation must NOT report this
simulation.restart.enabled = TRUE
