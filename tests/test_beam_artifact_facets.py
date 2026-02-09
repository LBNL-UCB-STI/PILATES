from pilates.workflows.steps import (
    _activitysim_output_facet_meta,
    _atlas_artifact_facet_meta,
    _beam_artifact_facets,
    _beam_log_facet_meta,
    _beam_postprocess_split_facet_meta,
    _urbansim_output_facet_meta,
)


def test_beam_artifact_facets_for_linkstats_parquet_sub_iteration():
    facets = _beam_artifact_facets("linkstats_parquet_2018_0_sub1")
    assert facets == {
        "artifact_family": "linkstats_parquet",
        "year": 2018,
        "iteration": 0,
        "beam_sub_iteration": 1,
    }


def test_beam_artifact_facets_for_phys_sim_linkstats():
    facets = _beam_artifact_facets(
        "linkstats_unmodified_parquet__y2030__i7__phys_sim_iter2__beam_sub_iter0"
    )
    assert facets == {
        "artifact_family": "linkstats_unmodified_phys_sim_iter_parquet",
        "year": 2030,
        "iteration": 7,
        "phys_sim_iteration": 2,
        "beam_sub_iteration": 0,
    }

    meta = _beam_log_facet_meta("raw_od_skims_zarr_2030_7_sub0")
    assert meta["facet_schema_version"] == "v1"
    assert meta["facet"]["artifact_family"] == "raw_od_skims_zarr"


def test_beam_postprocess_split_facet_meta():
    split_meta = _beam_postprocess_split_facet_meta(
        "events_parquet_2018_0_type_PathTraversal"
    )
    assert split_meta["facet"]["artifact_family"] == "events_parquet_split"
    assert split_meta["facet"]["event_type"] == "PathTraversal"
    assert split_meta["facet"]["year"] == 2018
    assert split_meta["facet"]["iteration"] == 0

    links_meta = _beam_postprocess_split_facet_meta("path_traversal_links_2018_0")
    assert links_meta["facet"]["artifact_family"] == "path_traversal_links"
    assert links_meta["facet"]["year"] == 2018
    assert links_meta["facet"]["iteration"] == 0


def test_activitysim_output_facet_meta():
    meta = _activitysim_output_facet_meta(
        "persons_asim_out",
        year=2030,
        iteration=3,
    )
    assert meta["facet"]["artifact_family"] == "persons"
    assert meta["facet"]["year"] == 2030
    assert meta["facet"]["iteration"] == 3
    assert meta["facet_schema_version"] == "v1"
    assert meta["facet_index"] is True


def test_urbansim_output_facet_meta():
    meta = _urbansim_output_facet_meta(
        "usim_input_archive_2035",
        forecast_year=2035,
    )
    assert meta["facet"]["artifact_family"] == "usim_input_archive"
    assert meta["facet"]["year"] == 2035
    assert meta["facet_schema_version"] == "v1"
    assert meta["facet_index"] is True


def test_atlas_artifact_facet_meta_for_adopt_inputs():
    meta = _atlas_artifact_facet_meta(
        "adopt/baseline/new_vehicles_biannual_values_2030",
        run_scenario="baseline",
        forecast_year=2035,
        artifact_family="atlas_run_input",
    )
    assert meta["facet"]["artifact_family"] == "atlas_run_input"
    assert meta["facet"]["input_group"] == "adopt"
    assert meta["facet"]["scenario"] == "baseline"
    assert meta["facet"]["input_year"] == 2030
    assert meta["facet"]["forecast_year"] == 2035


def test_atlas_artifact_facet_meta_for_vehicle_mapping():
    meta = _atlas_artifact_facet_meta(
        "vehicle_type_mapping_evMandForced2",
        run_scenario="baseline",
        forecast_year=2030,
        artifact_family="atlas_preprocess_output",
    )
    assert meta["facet"]["artifact_family"] == "atlas_preprocess_output"
    assert meta["facet"]["input_group"] == "vehicle_type_mapping"
    assert meta["facet"]["scenario"] == "zev_mandate"
