"""Surrogate-based Bayesian UQ for OSMOSE calibration.

Thin subpackage boundary: no eager re-exports and no heavy imports here, so
later phases' optional dependencies (emcee, dynesty, arviz) stay lazy. Import
from submodules directly, e.g. ``from osmose.calibration.uq.emulator import
GPEmulator``.
"""
