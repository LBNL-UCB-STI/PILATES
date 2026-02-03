from types import SimpleNamespace

from pilates.activitysim.outputs import ActivitySimPostprocessOutputs
from pilates.workflows import steps


def test_activitysim_postprocess_logs_content_hash(monkeypatch, tmp_path) -> None:
    captured = {}

    def _fake_make_generic_step_function(**kwargs):
        captured["output_logger"] = kwargs["output_logger"]
        return lambda *args, **inner_kwargs: None

    monkeypatch.setattr(
        steps, "_make_generic_step_function", _fake_make_generic_step_function
    )

    steps.make_activitysim_postprocess_step(
        coupler=SimpleNamespace(),
        outputs_holder=SimpleNamespace(),
    )

    output_logger = captured["output_logger"]
    calls = []

    def _log_output_only(*, key, path, description, **meta):
        calls.append((key, meta))

    monkeypatch.setattr(steps, "log_output_only", _log_output_only)

    outputs = ActivitySimPostprocessOutputs(
        usim_datastore_h5=None,
        asim_output_dir=tmp_path,
        processed_outputs={
            "asim_input_skims_zarr_archived": tmp_path / "skims.zarr"
        },
        processed_output_hashes={
            "asim_input_skims_zarr_archived": "abc123"
        },
    )

    output_logger(
        outputs,
        settings=SimpleNamespace(),
        state=SimpleNamespace(),
        workspace=SimpleNamespace(),
        holder=SimpleNamespace(),
    )

    assert calls == [("asim_input_skims_zarr_archived", {"content_hash": "abc123"})]
