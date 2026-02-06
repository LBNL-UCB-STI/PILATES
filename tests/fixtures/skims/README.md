# Mini skims fixture

Source:
- `/Users/zaneedell/git/PILATES/pilates/activitysim/output/cache/skims.zarr`

Selection:
- Zones: `otaz` and `dtaz` first 5 (indices 0..4)
- Time periods: all available in source
- Variables (subset to exercise core + transit + ridehail logic):
  - Auto: `SOV_*`, `SOVTOLL_*`, `HOV2_*`, `HOV3_*`, `HOV3TOLL_*`
  - Ridehail: `RH_POOLED_TRIPS`, `RH_POOLED_REJECTIONPROB`, `RH_SOLO_TRIPS`
  - Transit: `WLK_TRN_WLK_FAILURES`, `WLK_TRN_WLK_TOTIVT`, `WLK_LOC_WLK_TRIPS`, `WLK_LOC_WLK_TOTIVT`, `WLK_COM_WLK_TRIPS`, `WLK_EXP_WLK_TOTIVT`, `WLK_EXP_WLK_FAILURES`, `WLK_EXP_WLK_IWAIT`
  - 2D: `DISTWALK`

Notes:
- `SOV_FREE` does not exist in the source store, so it is not included.
- The fixture is written as Zarr v2 with consolidated metadata.
- `mini_skims_expected.zarr` is a golden output created by running
  `_merge_beam_skims_to_zarr` on a zeroed main skims store with
  `mini_skims.zarr` as the partial skims input.
