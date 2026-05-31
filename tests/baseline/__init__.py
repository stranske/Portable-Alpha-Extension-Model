"""Portable-Alpha-Extension-Model app behavior baseline kit.

Built on the shared ``baseline_kit`` package. This directory holds only the
app-specific pieces (adapter, catalog, invariant bounds); the generic harness
(directional engine, invariant assertion, golden glue, coverage manifest) is
imported from ``baseline_kit`` -- the same core used by Trend_Model_Project and
trip-planner. It drives the real simulation (``pa_core.facade.run_single``).
"""
