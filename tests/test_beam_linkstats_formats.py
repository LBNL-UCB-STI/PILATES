from pilates.beam.outputs import BeamRunOutputs
from pilates.beam.runner import _select_latest_linkstats_path
from pilates.generic.records import FileRecord, RecordStore


def test_beam_run_outputs_promotes_parquet_linkstats_to_canonical_key(tmp_path):
    parquet_path = tmp_path / "it.2.linkstats.parquet"
    parquet_path.write_text("stub")
    plans_path = tmp_path / "it.2.plans.csv.gz"
    plans_path.write_text("stub")

    outputs = BeamRunOutputs(
        beam_output_dir=tmp_path,
        raw_outputs={
            "linkstats_parquet_2018_0": parquet_path,
            "beam_plans_out_2018_0": plans_path,
        },
    )

    by_key = {short_name: path for short_name, path, _ in outputs._iter_record_items()}
    assert by_key["linkstats"] == parquet_path
    assert by_key["beam_plans_out"] == plans_path


def test_beam_run_outputs_publication_selection_skips_sub_iteration_linkstats(tmp_path):
    base_linkstats = tmp_path / "it.2.linkstats.csv.gz"
    sub_linkstats = tmp_path / "it.2.sub.linkstats.parquet"
    for path in (base_linkstats, sub_linkstats):
        path.write_text("stub")

    outputs = BeamRunOutputs(
        beam_output_dir=tmp_path,
        raw_outputs={
            "linkstats_2018_0": base_linkstats,
            "linkstats_parquet_2018_0_sub1": sub_linkstats,
        },
    )

    selected = outputs.promoted_linkstats_for_publication()

    assert selected == ("linkstats_2018_0", base_linkstats)


def test_beam_run_outputs_publication_selection_prefers_latest_non_sub_iteration_plans(
    tmp_path,
):
    plans_iter0 = tmp_path / "it.0.plans.xml.gz"
    plans_iter1 = tmp_path / "it.1.plans.xml.gz"
    plans_sub = tmp_path / "it.1.sub.plans.xml.gz"
    for path in (plans_iter0, plans_iter1, plans_sub):
        path.write_text("stub")

    outputs = BeamRunOutputs(
        beam_output_dir=tmp_path,
        raw_outputs={
            "beam_plans_out_2018_0": plans_iter0,
            "beam_plans_out_2018_1": plans_iter1,
            "beam_plans_out_2018_1_sub1": plans_sub,
        },
    )

    selected = outputs.promoted_plans_for_publication()

    assert selected == ("beam_plans_out_2018_1", plans_iter1)


def test_select_latest_linkstats_path_accepts_parquet_and_csv_keys(tmp_path):
    beam_output_root = tmp_path / "beam_output"
    beam_input_root = tmp_path / "beam_input"

    csv_path = beam_output_root / "seattle" / "it.2" / "2.linkstats.csv.gz"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("stub")

    parquet_path = beam_output_root / "seattle" / "it.2" / "2.linkstats.parquet"
    parquet_path.write_text("stub")

    records = RecordStore(
        recordList=[
            FileRecord(
                file_path=str(parquet_path), short_name="linkstats_parquet_2018_0"
            ),
            FileRecord(file_path=str(csv_path), short_name="linkstats_2018_0"),
            FileRecord(
                file_path=str(
                    beam_output_root / "seattle" / "it.1" / "1.linkstats.parquet"
                ),
                short_name="linkstats_parquet_2018_0_sub1",
            ),
        ]
    )

    selected = _select_latest_linkstats_path(
        records,
        abs_beam_input=str(beam_input_root),
        abs_beam_output=str(beam_output_root),
    )

    assert selected == "/app/output/seattle/it.2/2.linkstats.csv.gz"


def test_select_latest_linkstats_path_uses_parquet_when_csv_absent(tmp_path):
    beam_output_root = tmp_path / "beam_output"
    beam_input_root = tmp_path / "beam_input"

    parquet_path = beam_output_root / "seattle" / "it.2" / "2.linkstats.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.write_text("stub")

    records = RecordStore(
        recordList=[
            FileRecord(
                file_path=str(parquet_path), short_name="linkstats_parquet_2018_0"
            )
        ]
    )

    selected = _select_latest_linkstats_path(
        records,
        abs_beam_input=str(beam_input_root),
        abs_beam_output=str(beam_output_root),
    )

    assert selected == "/app/output/seattle/it.2/2.linkstats.parquet"
