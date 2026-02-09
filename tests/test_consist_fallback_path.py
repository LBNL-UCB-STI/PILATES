from unittest.mock import Mock, patch

import pytest

from pilates.generic.runner import GenericRunner


@patch("pilates.generic.runner.cr.current_tracker", return_value=None)
@patch("pilates.generic.runner.GenericRunner._run_container_direct")
def test_missing_tracker_raises_before_fallback(mock_direct, _mock_current_tracker):
    with pytest.raises(RuntimeError, match="Consist tracker must be active"):
        GenericRunner.run_container(
            client=None,
            settings=Mock(),
            image="img",
            volumes={},
            command="cmd",
            model_name="model",
        )

    assert not mock_direct.called
