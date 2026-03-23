"""Shared production-like harness for workflow contract tests."""

from __future__ import annotations

import inspect
from typing import Any, Callable


class CouplerStub:
    """Minimal in-memory coupler for workflow contract tests."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._values[key] = value

    def set_from_artifact(self, key: str, value: Any) -> None:
        self._values[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def pop(self, key: str, default: Any = None) -> Any:
        return self._values.pop(key, default)

    def keys(self) -> list[str]:
        return list(self._values.keys())

    def require(self, key: str) -> Any:
        if key not in self._values:
            raise KeyError(f"Coupler missing key={key!r}")
        return self._values[key]


class FakeScenario:
    """
    Scenario stub that records calls and enforces production-like coupling.

    Explicit ``inputs`` stay as explicit call arguments. ``input_keys`` must
    already be present in the coupler before the step starts.
    """

    def __init__(self, coupler: CouplerStub) -> None:
        self.coupler = coupler
        self.calls: list[dict[str, Any]] = []

    def run(self, **kwargs: Any) -> dict[str, str]:
        binding = kwargs.get("binding")
        if binding is not None:
            inputs = binding.inputs or {}
            input_keys = binding.input_keys or []
            optional_input_keys = binding.optional_input_keys or []
        else:
            inputs = kwargs.get("inputs") or {}
            input_keys = kwargs.get("input_keys") or []
            optional_input_keys = kwargs.get("optional_input_keys") or []
        fn = kwargs["fn"]
        model = kwargs.get("model")
        if model is None:
            step_meta = getattr(fn, "__consist_step__", None)
            model = getattr(step_meta, "model", None)
        self.calls.append(
            {
                "fn_name": getattr(fn, "__name__", "<unknown>"),
                "model": model,
                "inputs": dict(inputs),
                "input_keys": list(input_keys),
                "optional_input_keys": list(optional_input_keys),
                "binding": binding,
            }
        )

        for key in input_keys:
            self.coupler.require(key)

        execution_options = kwargs.get("execution_options")
        runtime_kwargs = kwargs.get("runtime_kwargs") or getattr(
            execution_options, "runtime_kwargs", None
        )
        fn_kwargs = dict(runtime_kwargs or {})
        sig = inspect.signature(fn)
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in sig.parameters.values()
        )
        if accepts_kwargs:
            fn_kwargs.update(inputs)
            for key in input_keys:
                fn_kwargs.setdefault(key, self.coupler.get(key))
        else:
            allowed = set(sig.parameters.keys())
            for key, value in inputs.items():
                if key in allowed:
                    fn_kwargs[key] = value
            for key in input_keys:
                if key in allowed:
                    fn_kwargs.setdefault(key, self.coupler.get(key))

        fn(**fn_kwargs)
        return {"status": "ok"}


class DummyPreprocessor:
    """Deterministic preprocessor stub backed by ``record_builder``."""

    def __init__(self, model_name: str, record_builder: Callable[..., Any]) -> None:
        self.model_name = model_name
        self._record_builder = record_builder

    def preprocess(
        self,
        workspace: Any,
        previous_records: Any = None,
        activity_demand_outputs: Any = None,
        previous_beam_outputs: Any = None,
        beam_preprocess_inputs: Any = None,
        **kwargs: Any,
    ) -> Any:
        return self._record_builder(self.model_name, "preprocess")


class DummyRunner:
    """Deterministic runner stub backed by ``record_builder``."""

    def __init__(self, model_name: str, record_builder: Callable[..., Any]) -> None:
        self.model_name = model_name
        self._record_builder = record_builder

    def run(
        self,
        input_store: Any,
        workspace: Any,
        extra_inputs: Any = None,
        previous_beam_outputs: Any = None,
    ) -> Any:
        return self._record_builder(self.model_name, "run")


class DummyPostprocessor:
    """Deterministic postprocessor stub backed by ``record_builder``."""

    def __init__(self, model_name: str, record_builder: Callable[..., Any]) -> None:
        self.model_name = model_name
        self._record_builder = record_builder

    def postprocess(
        self,
        raw_outputs: Any,
        workspace: Any,
        model_run_hash: Any = None,
    ) -> Any:
        return self._record_builder(self.model_name, "postprocess")
