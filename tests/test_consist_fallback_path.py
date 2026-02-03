from unittest.mock import Mock, patch

from pilates.generic.runner import GenericRunner


@patch("pilates.generic.runner.cr.consist_available", return_value=False)
@patch("pilates.generic.runner.GenericRunner._run_container_direct")
def test_consist_disabled_runs_without_tracker(mock_direct, mock_consist_available):
    mock_direct.return_value = True

    result = GenericRunner.run_container(
        client=None,
        settings=Mock(),
        image="img",
        volumes={},
        command="cmd",
        model_name="model",
    )

    assert result is True
    assert mock_direct.called
