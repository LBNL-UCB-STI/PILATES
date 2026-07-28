"""Retired legacy restart-equivalence harness.

The former suite modeled unsupported mid-stage restoration through ``StageRunner``,
manifest replay, and a fake scenario dictionary.  Supported whole-stage and
whole-year native execution remains covered by ``test_golden_stub_workflow``;
the committed ``beam_run_completed -> beam_postprocess`` boundary is exercised
by the dedicated checkpoint tests through the public native dispatch.
"""
