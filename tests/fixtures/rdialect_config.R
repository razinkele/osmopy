# Synthetic R-dialect OSMOSE config — hand-written for tests, NOT vendored upstream content.
# Mirrors the shape of osmose-model/osmose-ben's osmose-ben.R: `key = value` lines in a .R file.
#
# Provenance -- every key below appears VERBATIM in the cited file (re-grepped 2026-07-17):
#   economy.enabled          -> osmose-ben/osmose-ben_v4.x_develop/osmose-ben.R:1048
#   surveys.name.sr1         -> osmose-ben/osmose-ben_v4.x_develop/osmose-ben.R:757
#   output.restart.enabled   -> osmose-ben/osmose-ben_v4.x_develop/osmose-ben.R:784
#
# NOTE: use `output.restart.enabled` (the real PRE-4.4.0 R key), NOT
# `simulation.restart.enabled`. The latter is the POST-shim canonical name and appears in
# ZERO R config files -- an earlier draft cited it as if osmose-ben.R contained it, which was
# a false provenance citation in the very fixture whose job is provenance discipline.
# Using the real key is also STRICTLY BETTER: the shim migrates it to
# simulation.restart.enabled, so the fixture now demonstrates the actual trap end-to-end
# (real R key -> shim -> allowlisted-but-unread canonical key -> silence).
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

# The real pre-4.4.0 R key (osmose-ben.R:784, value FALSE there too). The shim migrates it to
# simulation.restart.enabled, which is allowlisted-but-unread -> strict validation must NOT
# report it, and the Python engine silently ignores it. That is the trap, end to end.
output.restart.enabled = FALSE
