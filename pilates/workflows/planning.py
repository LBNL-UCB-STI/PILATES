from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Tuple

from pilates.atlas.inputs import (
    atlas_run_years,
    atlas_static_input_keys_for_interval,
)
from pilates.config.models import PilatesConfig, load_config
from pilates.utils.io import apply_runtime_flags, compute_model_enabled_flags
from pilates.workflows.artifact_keys import (
    FINAL_SKIMS_OMX,
    USIM_DATASTORE_BASE_H5,
    USIM_DATASTORE_CURRENT_H5,
    USIM_FORECAST_OUTPUT,
    USIM_H5_UPDATED,
)
from pilates.workflows.catalog import (
    workflow_step_contracts_by_name,
    workflow_step_spec_for_step_name,
)
if TYPE_CHECKING:
    from pilates.workflows.surface import EnabledWorkflowSurface


_RUN_GLOBAL_EXTERNAL_ARTIFACT_KEYS = {
    FINAL_SKIMS_OMX,
}
_FILTER_ATLAS_LINEAGE_FUTURE_VEHICLE_SERIES = True


def _slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z_]+", "_", value)
    slug = slug.strip("_")
    return slug or "node"


@dataclass
class PlannedStepRun:
    id: str
    sequence: int
    step_name: str
    stage_name: str
    phase: str
    label: str
    year: Optional[int] = None
    forecast_year: Optional[int] = None
    iteration: Optional[int] = None
    atlas_year: Optional[int] = None
    optional: bool = False
    depends_on: List[str] = field(default_factory=list)
    upstream_step_inputs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlannedArtifact:
    id: str
    sequence: int
    label: str
    instance_key: str
    artifact_key: str
    canonical_key: str
    producer_step_run_id: Optional[str]
    year: Optional[int] = None
    forecast_year: Optional[int] = None
    iteration: Optional[int] = None
    atlas_year: Optional[int] = None
    optional: bool = False
    external: bool = False
    dynamic: bool = False
    family: Optional[str] = None
    path_role: Optional[str] = None
    resolved_path_hint: Optional[str] = None
    path_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlannedEdge:
    id: str
    source: str
    target: str
    kind: str
    artifact_key: Optional[str] = None
    optional: bool = False
    dynamic: bool = False
    family: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContractGap:
    id: str
    step_run_id: str
    kind: str
    message: str
    severity: str = "warning"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StaticExecutionPlan:
    config_path: Optional[str]
    step_contracts: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any]
    assumptions: List[str]
    step_runs: List[PlannedStepRun] = field(default_factory=list)
    artifacts: List[PlannedArtifact] = field(default_factory=list)
    edges: List[PlannedEdge] = field(default_factory=list)
    contract_gaps: List[ContractGap] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_path": self.config_path,
            "metadata": self.metadata,
            "assumptions": list(self.assumptions),
            "step_contracts": self.step_contracts,
            "step_runs": [item.to_dict() for item in self.step_runs],
            "artifacts": [item.to_dict() for item in self.artifacts],
            "edges": [item.to_dict() for item in self.edges],
            "contract_gaps": [item.to_dict() for item in self.contract_gaps],
        }


def _attach_enabled_flags(settings: PilatesConfig) -> Dict[str, bool]:
    enabled_flags = compute_model_enabled_flags(settings)
    apply_runtime_flags(settings, enabled_flags)
    return enabled_flags


def load_settings_for_planning(config_path: str) -> PilatesConfig:
    settings = load_config(config_path)
    _attach_enabled_flags(settings)
    return settings


def _interval_step_for_year(settings: PilatesConfig, year: int) -> int:
    return 7 if year == 2010 else int(settings.run.travel_model_freq)


def _iter_planning_years(
    settings: PilatesConfig,
    *,
    surface: "EnabledWorkflowSurface",
) -> List[Dict[str, int]]:
    years: List[Dict[str, int]] = []
    land_use_enabled = surface.stage_enabled("land_use")
    current_year = int(settings.run.start_year)
    end_year = int(settings.run.end_year)

    while current_year <= end_year:
        if land_use_enabled:
            forecast_year = min(
                current_year + _interval_step_for_year(settings, current_year),
                end_year,
            )
        else:
            forecast_year = current_year
        years.append(
            {
                "year": current_year,
                "forecast_year": forecast_year,
            }
        )
        if not land_use_enabled:
            break
        if forecast_year <= current_year:
            break
        current_year = forecast_year

    return years


def _atlas_sub_years(year: int, forecast_year: int) -> List[int]:
    years = [year]
    if forecast_year <= year:
        return years
    years.extend(range(year + 2, forecast_year + 1, 2))
    return years


def _full_skim_run_schedule(settings: PilatesConfig) -> str:
    beam_cfg = getattr(settings, "beam", None)
    skim_cfg = getattr(beam_cfg, "full_skim", None) if beam_cfg is not None else None
    if skim_cfg is None:
        return "disabled"
    return str(getattr(skim_cfg, "run_schedule", "standalone"))


def _should_run_full_skim(settings: PilatesConfig, iteration: int) -> bool:
    schedule = _full_skim_run_schedule(settings)
    if schedule == "standalone":
        return True
    if schedule == "after_each_iteration":
        return True
    if schedule == "after_final_iteration":
        return iteration == int(settings.run.supply_demand_iters) - 1
    return False


def _effective_supply_demand_iterations(
    settings: PilatesConfig,
    *,
    surface: "EnabledWorkflowSurface",
) -> int:
    total_iters = int(settings.run.supply_demand_iters)
    activity_enabled = surface.stage_enabled("activity_demand")
    if not activity_enabled and total_iters > 1:
        return 1
    return total_iters


def _format_dynamic_family(family: str, step_run: PlannedStepRun) -> str:
    scope = {
        "year": step_run.year,
        "forecast_year": step_run.forecast_year,
        "iteration": step_run.iteration,
        "atlas_year": step_run.atlas_year,
    }
    try:
        return family.format(**scope)
    except Exception:
        return family


def _scope_suffix(step_run: PlannedStepRun) -> str:
    parts: List[str] = []
    if step_run.year is not None:
        parts.append("y%s" % step_run.year)
    if step_run.forecast_year is not None and step_run.forecast_year != step_run.year:
        parts.append("fy%s" % step_run.forecast_year)
    if step_run.iteration is not None:
        parts.append("i%s" % step_run.iteration)
    if step_run.atlas_year is not None:
        parts.append("sy%s" % step_run.atlas_year)
    return ":".join(parts)


def _artifact_instance_key(
    canonical_key: str,
    *,
    step_run: Optional[PlannedStepRun],
    external: bool,
    scope_external: bool = True,
) -> str:
    if external or step_run is None:
        prefix = "external"
        suffix = (
            _scope_suffix(step_run)
            if step_run is not None and scope_external
            else ""
        )
    else:
        prefix = step_run.step_name
        suffix = _scope_suffix(step_run)
    if suffix:
        return "%s:%s:%s" % (prefix, suffix, canonical_key)
    return "%s:%s" % (prefix, canonical_key)


def _artifact_label(
    canonical_key: str,
    *,
    step_run: Optional[PlannedStepRun],
    external: bool,
    dynamic: bool,
    family: Optional[str],
    scope_external: bool = True,
) -> str:
    parts = [canonical_key]
    if step_run is not None and (not external or scope_external):
        scope = _scope_suffix(step_run)
        if scope:
            parts.append(scope)
    if external:
        parts.append("external")
    elif step_run is not None:
        parts.append(step_run.step_name)
    if dynamic and family is not None:
        parts.append(family)
    return "\n".join(parts)


def _specialize_contract_for_planned_step_run(
    contract: Mapping[str, Any],
    *,
    settings: PilatesConfig,
    step_name: str,
    atlas_year: Optional[int],
) -> Dict[str, Any]:
    specialized = dict(contract)

    if (
        not _FILTER_ATLAS_LINEAGE_FUTURE_VEHICLE_SERIES
        or atlas_year is None
        or step_name not in {"atlas_preprocess", "atlas_run"}
    ):
        return specialized

    run_years = atlas_run_years(settings)
    interval_start_year = min(run_years) if run_years else int(atlas_year)
    atlas_static_keys = atlas_static_input_keys_for_interval(
        settings,
        interval_start_year=interval_start_year,
        interval_end_year=int(atlas_year),
    )

    if step_name == "atlas_preprocess":
        specialized["optional_output_keys"] = list(
            dict.fromkeys(
                [
                    key
                    for key in specialized["optional_output_keys"]
                    if not str(key).startswith("adopt/")
                ]
                + list(atlas_static_keys)
            )
        )
        return specialized

    specialized["optional_input_keys"] = list(
        dict.fromkeys(
            [
                key
                for key in specialized["optional_input_keys"]
                if not str(key).startswith("adopt/")
            ]
            + list(atlas_static_keys)
        )
    )
    return specialized


def _planned_workspace_root(settings: PilatesConfig) -> Optional[str]:
    run_cfg = getattr(settings, "run", None)
    if run_cfg is None:
        return None
    output_dir = getattr(run_cfg, "output_directory", None)
    if not output_dir:
        return None
    run_name = getattr(run_cfg, "output_run_name", None)
    if run_name:
        return os.path.join(str(output_dir), str(run_name))
    return str(output_dir)


def _planned_usim_datastore_paths(
    settings: PilatesConfig,
    *,
    forecast_year: Optional[int],
) -> tuple[Optional[str], Optional[str]]:
    workspace_root = _planned_workspace_root(settings)
    urbansim_cfg = getattr(settings, "urbansim", None)
    run_cfg = getattr(settings, "run", None)
    if workspace_root is None or urbansim_cfg is None or run_cfg is None:
        return None, None

    region = getattr(run_cfg, "region", None)
    region_mappings = getattr(urbansim_cfg, "region_mappings", None) or {}
    region_to_region_id = (
        region_mappings.get("region_to_region_id", {})
        if isinstance(region_mappings, Mapping)
        else getattr(region_mappings, "region_to_region_id", {})
    )
    region_id = region_to_region_id.get(region) if region is not None else None
    mutable_data_folder = getattr(urbansim_cfg, "local_mutable_data_folder", None)
    input_template = getattr(urbansim_cfg, "input_file_template", None)
    output_template = getattr(urbansim_cfg, "output_file_template", None)
    if (
        region_id is None
        or not mutable_data_folder
        or not input_template
        or not output_template
    ):
        return None, None

    usim_dir = os.path.join(workspace_root, str(mutable_data_folder))
    input_path = os.path.join(usim_dir, input_template.format(region_id=region_id))
    output_path = (
        os.path.join(usim_dir, output_template.format(year=forecast_year))
        if forecast_year is not None
        else None
    )
    return input_path, output_path


def _planned_artifact_path_metadata(
    settings: PilatesConfig,
    *,
    artifact_key: str,
    step_run: Optional[PlannedStepRun],
    external: bool,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    del external
    input_path, forecast_output_path = _planned_usim_datastore_paths(
        settings,
        forecast_year=step_run.forecast_year if step_run is not None else None,
    )

    if artifact_key == USIM_DATASTORE_BASE_H5:
        return (
            "semantic_base_datastore",
            input_path,
            "Static/base UrbanSim datastore handle. It may physically match the current datastore handle in some runs.",
        )

    if artifact_key == USIM_FORECAST_OUTPUT:
        return (
            "forecast_output_datastore",
            forecast_output_path,
            "Forecast-year UrbanSim output datastore emitted by the runner before postprocessing merges it back into the mutable input slot.",
        )

    if artifact_key == USIM_H5_UPDATED:
        return (
            "current_mutable_datastore",
            input_path,
            "Updated current UrbanSim datastore handle republished onto the mutable input slot for downstream steps.",
        )

    if artifact_key != USIM_DATASTORE_CURRENT_H5:
        return None, None, None

    producer_step_name = step_run.step_name if step_run is not None else None
    if producer_step_name == "urbansim_run":
        return (
            "forecast_output_datastore",
            forecast_output_path,
            "UrbanSim runner output for the forecast year.",
        )

    if producer_step_name in {
        "urbansim_postprocess",
        "atlas_postprocess",
        "activitysim_postprocess",
    }:
        return (
            "current_mutable_datastore",
            input_path,
            "Current mutable UrbanSim datastore handoff. This often reuses the same physical input-slot path as the base datastore handle.",
        )

    if producer_step_name == "urbansim_preprocess":
        return (
            "current_input_seed",
            input_path,
            "Current UrbanSim datastore handle at preprocessing time. This is the seeded mutable input-slot file for the runner.",
        )

    return (
        "current_datastore_handle",
        input_path,
        "Current UrbanSim datastore handle. Depending on the stage boundary, this may point at the mutable input slot or a forecast output datastore.",
    )


class _PlanBuilder:
    def __init__(
        self,
        *,
        settings: PilatesConfig,
        surface: "EnabledWorkflowSurface",
        config_path: Optional[str],
        include_postprocessing: bool,
    ) -> None:
        self.settings = settings
        self.surface = surface
        self.config_path = config_path
        self.include_postprocessing = include_postprocessing
        self.contracts = workflow_step_contracts_by_name(settings=self.settings)
        self.plan = StaticExecutionPlan(
            config_path=config_path,
            step_contracts=self.contracts,
            metadata=self._build_metadata(),
            assumptions=self._build_assumptions(),
        )
        self._step_sequence = 0
        self._artifact_sequence = 0
        self._edge_sequence = 0
        self._gap_sequence = 0
        self._latest_step_by_name: Dict[str, str] = {}
        self._latest_artifact_by_key: Dict[str, str] = {}
        self._latest_artifact_by_dynamic_label: Dict[str, str] = {}
        self._artifacts_by_id: Dict[str, PlannedArtifact] = {}
        self._external_artifact_ids: Dict[str, str] = {}

    def _stage_enabled(self, stage_name: str) -> bool:
        return self.surface.stage_enabled(stage_name)

    def _iter_years(self) -> List[Dict[str, int]]:
        return _iter_planning_years(
            self.settings,
            surface=self.surface,
        )

    def _effective_supply_demand_iterations(self) -> int:
        return _effective_supply_demand_iterations(
            self.settings,
            surface=self.surface,
        )

    def _scope_external_artifact(self, artifact_key: str, *, dynamic: bool) -> bool:
        if dynamic:
            return True
        return artifact_key not in _RUN_GLOBAL_EXTERNAL_ARTIFACT_KEYS

    def _build_metadata(self) -> Dict[str, Any]:
        years = self._iter_years()
        enabled_flags = self.surface.profile.to_dict()
        return {
            "start_year": int(self.settings.run.start_year),
            "end_year": int(self.settings.run.end_year),
            "travel_model_freq": int(self.settings.run.travel_model_freq),
            "supply_demand_iters": self._effective_supply_demand_iterations(),
            "enabled_flags": enabled_flags,
            "years": years,
            "full_skim_schedule": _full_skim_run_schedule(self.settings),
            "include_postprocessing": self.include_postprocessing,
        }

    def _build_assumptions(self) -> List[str]:
        assumptions = [
            "Artifact edges come from workflow_step_contracts_by_name(); step ordering is tracked separately via depends_on and upstream_step_inputs.",
            "Missing or blank contracts are preserved as underdeclared gaps instead of being guessed from runtime code.",
        ]
        if self.include_postprocessing:
            assumptions.append(
                "Postprocessing is included when settings.postprocessing is present, even though WorkflowState does not currently advertise it via enabled_stages."
            )
        return assumptions

    def build(self) -> StaticExecutionPlan:
        for year_info in self._iter_years():
            self._add_year_steps(
                year=year_info["year"],
                forecast_year=year_info["forecast_year"],
            )
        return self.plan

    def _add_year_steps(self, *, year: int, forecast_year: int) -> None:
        if self._stage_enabled("land_use"):
            self._add_step_run("urbansim_preprocess", year=year, forecast_year=forecast_year)
            self._add_step_run("urbansim_run", year=year, forecast_year=forecast_year)
            self._add_step_run(
                "urbansim_postprocess",
                year=year,
                forecast_year=forecast_year,
            )

        if self._stage_enabled("vehicle_ownership_model"):
            for atlas_year in _atlas_sub_years(year, forecast_year):
                self._add_step_run(
                    "atlas_preprocess",
                    year=year,
                    forecast_year=forecast_year,
                    atlas_year=atlas_year,
                )
                self._add_step_run(
                    "atlas_run",
                    year=year,
                    forecast_year=forecast_year,
                    atlas_year=atlas_year,
                )
                self._add_step_run(
                    "atlas_postprocess",
                    year=year,
                    forecast_year=forecast_year,
                    atlas_year=atlas_year,
                )

        if self._stage_enabled("supply_demand_loop"):
            self._add_supply_demand_steps(year=year, forecast_year=forecast_year)

        if self.include_postprocessing:
            self._add_step_run("postprocessing", year=year, forecast_year=forecast_year)

    def _add_supply_demand_steps(self, *, year: int, forecast_year: int) -> None:
        total_iters = self._effective_supply_demand_iterations()
        activity_enabled = self._stage_enabled("activity_demand")
        traffic_enabled = self._stage_enabled("traffic_assignment")
        compile_added = False
        schedule = _full_skim_run_schedule(self.settings)

        for iteration in range(total_iters):
            if activity_enabled:
                self._add_step_run(
                    "activitysim_preprocess",
                    year=year,
                    forecast_year=forecast_year,
                    iteration=iteration,
                )
                if not compile_added:
                    self._add_step_run(
                        "activitysim_compile",
                        year=year,
                        forecast_year=forecast_year,
                        iteration=iteration,
                    )
                    compile_added = True
                self._add_step_run(
                    "activitysim_run",
                    year=year,
                    forecast_year=forecast_year,
                    iteration=iteration,
                )
                self._add_step_run(
                    "activitysim_postprocess",
                    year=year,
                    forecast_year=forecast_year,
                    iteration=iteration,
                )

            if not traffic_enabled:
                continue

            self._add_step_run(
                "beam_preprocess",
                year=year,
                forecast_year=forecast_year,
                iteration=iteration,
            )
            if schedule == "standalone":
                self._add_step_run(
                    "beam_full_skim",
                    year=year,
                    forecast_year=forecast_year,
                    iteration=iteration,
                )
                continue

            self._add_step_run(
                "beam_run",
                year=year,
                forecast_year=forecast_year,
                iteration=iteration,
            )
            self._add_step_run(
                "beam_postprocess",
                year=year,
                forecast_year=forecast_year,
                iteration=iteration,
            )
            if _should_run_full_skim(self.settings, iteration):
                self._add_step_run(
                    "beam_full_skim",
                    year=year,
                    forecast_year=forecast_year,
                    iteration=iteration,
                )

    def _make_step_label(
        self,
        step_name: str,
        *,
        year: Optional[int],
        forecast_year: Optional[int],
        iteration: Optional[int],
        atlas_year: Optional[int],
    ) -> str:
        parts = [step_name]
        if year is not None:
            parts.append("year=%s" % year)
        if forecast_year is not None and forecast_year != year:
            parts.append("forecast=%s" % forecast_year)
        if atlas_year is not None:
            parts.append("subyear=%s" % atlas_year)
        if iteration is not None:
            parts.append("iter=%s" % iteration)
        return "\n".join(parts)

    def _add_step_run(
        self,
        step_name: str,
        *,
        year: Optional[int] = None,
        forecast_year: Optional[int] = None,
        iteration: Optional[int] = None,
        atlas_year: Optional[int] = None,
    ) -> PlannedStepRun:
        contract = _specialize_contract_for_planned_step_run(
            self.contracts[step_name],
            settings=self.settings,
            step_name=step_name,
            atlas_year=atlas_year,
        )
        self._step_sequence += 1
        step_id = "step_%03d_%s" % (self._step_sequence, _slugify(step_name))
        step_run = PlannedStepRun(
            id=step_id,
            sequence=self._step_sequence,
            step_name=step_name,
            stage_name=str(contract["stage_name"]),
            phase=str(contract["phase"]),
            label=self._make_step_label(
                step_name,
                year=year,
                forecast_year=forecast_year,
                iteration=iteration,
                atlas_year=atlas_year,
            ),
            year=year,
            forecast_year=forecast_year,
            iteration=iteration,
            atlas_year=atlas_year,
            optional=bool(contract["optional"]),
            depends_on=list(contract["depends_on"]),
            upstream_step_inputs=list(contract["upstream_step_inputs"]),
        )
        self.plan.step_runs.append(step_run)
        self._add_declared_input_edges(step_run, contract)
        self._add_dependency_edges(step_run, contract)
        self._add_declared_output_edges(step_run, contract)
        self._add_contract_gaps(step_run, contract)
        self._latest_step_by_name[step_name] = step_id
        return step_run

    def _edge_id(self) -> str:
        self._edge_sequence += 1
        return "edge_%04d" % self._edge_sequence

    def _gap_id(self) -> str:
        self._gap_sequence += 1
        return "gap_%04d" % self._gap_sequence

    def _artifact_id(self) -> str:
        self._artifact_sequence += 1
        return "artifact_%04d" % self._artifact_sequence

    def _create_external_artifact(
        self,
        artifact_key: str,
        *,
        optional: bool,
        dynamic: bool,
        family: Optional[str],
        step_run: PlannedStepRun,
    ) -> str:
        scope_external = self._scope_external_artifact(artifact_key, dynamic=dynamic)
        path_role, resolved_path_hint, path_notes = _planned_artifact_path_metadata(
            self.settings,
            artifact_key=artifact_key,
            step_run=step_run,
            external=True,
        )
        scope_parts = [artifact_key, str(dynamic)]
        if scope_external:
            scope_parts.extend(
                [
                    str(step_run.year),
                    str(step_run.forecast_year),
                    str(step_run.iteration),
                    str(step_run.atlas_year),
                ]
            )
        external_key = "|".join(scope_parts)
        artifact_id = self._external_artifact_ids.get(external_key)
        if artifact_id is not None:
            return artifact_id

        instance_key = _artifact_instance_key(
            artifact_key,
            step_run=step_run,
            external=True,
            scope_external=scope_external,
        )
        artifact_id = self._artifact_id()
        artifact = PlannedArtifact(
            id=artifact_id,
            sequence=self._artifact_sequence,
            label=_artifact_label(
                artifact_key,
                step_run=step_run,
                external=True,
                dynamic=dynamic,
                family=family,
                scope_external=scope_external,
            ),
            instance_key=instance_key,
            artifact_key=artifact_key,
            canonical_key=artifact_key,
            producer_step_run_id=None,
            year=step_run.year if scope_external else None,
            forecast_year=step_run.forecast_year if scope_external else None,
            iteration=step_run.iteration if scope_external else None,
            atlas_year=step_run.atlas_year if scope_external else None,
            optional=optional,
            external=True,
            dynamic=dynamic,
            family=family,
            path_role=path_role,
            resolved_path_hint=resolved_path_hint,
            path_notes=path_notes,
        )
        self.plan.artifacts.append(artifact)
        self._artifacts_by_id[artifact_id] = artifact
        self._external_artifact_ids[external_key] = artifact_id
        return artifact_id

    def _create_step_output_artifact(
        self,
        artifact_key: str,
        *,
        step_run: PlannedStepRun,
        optional: bool,
        dynamic: bool,
        family: Optional[str],
    ) -> str:
        path_role, resolved_path_hint, path_notes = _planned_artifact_path_metadata(
            self.settings,
            artifact_key=artifact_key,
            step_run=step_run,
            external=False,
        )
        instance_key = _artifact_instance_key(
            artifact_key,
            step_run=step_run,
            external=False,
        )
        artifact_id = self._artifact_id()
        artifact = PlannedArtifact(
            id=artifact_id,
            sequence=self._artifact_sequence,
            label=_artifact_label(
                artifact_key,
                step_run=step_run,
                external=False,
                dynamic=dynamic,
                family=family,
            ),
            instance_key=instance_key,
            artifact_key=artifact_key,
            canonical_key=artifact_key,
            producer_step_run_id=step_run.id,
            year=step_run.year,
            forecast_year=step_run.forecast_year,
            iteration=step_run.iteration,
            atlas_year=step_run.atlas_year,
            optional=optional,
            external=False,
            dynamic=dynamic,
            family=family,
            path_role=path_role,
            resolved_path_hint=resolved_path_hint,
            path_notes=path_notes,
        )
        self.plan.artifacts.append(artifact)
        self._artifacts_by_id[artifact_id] = artifact
        if dynamic:
            self._latest_artifact_by_dynamic_label[artifact_key] = artifact_id
        else:
            self._latest_artifact_by_key[artifact_key] = artifact_id
        self.plan.edges.append(
            PlannedEdge(
                id=self._edge_id(),
                source=step_run.id,
                target=artifact_id,
                kind="produces",
                artifact_key=artifact_key,
                optional=optional,
                dynamic=dynamic,
                family=family,
            )
        )
        return artifact_id

    def _resolve_input_artifact_id(
        self,
        artifact_key: str,
        *,
        step_run: PlannedStepRun,
        optional: bool,
        dynamic: bool,
        family: Optional[str],
    ) -> str:
        if dynamic:
            artifact_id = self._latest_artifact_by_dynamic_label.get(artifact_key)
        else:
            artifact_id = self._latest_artifact_by_key.get(artifact_key)
        if artifact_id is not None:
            return artifact_id
        return self._create_external_artifact(
            artifact_key,
            optional=optional,
            dynamic=dynamic,
            family=family,
            step_run=step_run,
        )

    def _add_declared_input_edges(
        self,
        step_run: PlannedStepRun,
        contract: Mapping[str, Any],
    ) -> None:
        for artifact_key in contract["input_keys"]:
            artifact_id = self._resolve_input_artifact_id(
                str(artifact_key),
                step_run=step_run,
                optional=False,
                dynamic=False,
                family=None,
            )
            self.plan.edges.append(
                PlannedEdge(
                    id=self._edge_id(),
                    source=artifact_id,
                    target=step_run.id,
                    kind="consumes",
                    artifact_key=str(artifact_key),
                )
            )

        for artifact_key in contract["optional_input_keys"]:
            artifact_id = self._resolve_input_artifact_id(
                str(artifact_key),
                step_run=step_run,
                optional=True,
                dynamic=False,
                family=None,
            )
            self.plan.edges.append(
                PlannedEdge(
                    id=self._edge_id(),
                    source=artifact_id,
                    target=step_run.id,
                    kind="consumes",
                    artifact_key=str(artifact_key),
                    optional=True,
                )
            )

        for family in contract["dynamic_input_families"]:
            resolved_family = _format_dynamic_family(str(family), step_run)
            artifact_id = self._resolve_input_artifact_id(
                resolved_family,
                step_run=step_run,
                optional=False,
                dynamic=True,
                family=str(family),
            )
            self.plan.edges.append(
                PlannedEdge(
                    id=self._edge_id(),
                    source=artifact_id,
                    target=step_run.id,
                    kind="consumes",
                    artifact_key=resolved_family,
                    dynamic=True,
                    family=str(family),
                )
            )

    def _add_dependency_edges(
        self,
        step_run: PlannedStepRun,
        contract: Mapping[str, Any],
    ) -> None:
        for upstream_name in list(contract["depends_on"]) + list(
            contract["upstream_step_inputs"]
        ):
            upstream_step_id = self._latest_step_by_name.get(str(upstream_name))
            if upstream_step_id is None:
                continue
            self.plan.edges.append(
                PlannedEdge(
                    id=self._edge_id(),
                    source=upstream_step_id,
                    target=step_run.id,
                    kind="depends_on",
                    artifact_key=str(upstream_name),
                )
            )

    def _add_declared_output_edges(
        self,
        step_run: PlannedStepRun,
        contract: Mapping[str, Any],
    ) -> None:
        for artifact_key in contract["output_keys"]:
            self._create_step_output_artifact(
                str(artifact_key),
                step_run=step_run,
                optional=False,
                dynamic=False,
                family=None,
            )

        for artifact_key in contract["optional_output_keys"]:
            self._create_step_output_artifact(
                str(artifact_key),
                step_run=step_run,
                optional=True,
                dynamic=False,
                family=None,
            )

        for family in contract["dynamic_output_families"]:
            resolved_family = _format_dynamic_family(str(family), step_run)
            self._create_step_output_artifact(
                resolved_family,
                step_run=step_run,
                optional=False,
                dynamic=True,
                family=str(family),
            )

    def _append_gap(self, step_run: PlannedStepRun, kind: str, message: str) -> None:
        self.plan.contract_gaps.append(
            ContractGap(
                id=self._gap_id(),
                step_run_id=step_run.id,
                kind=kind,
                message=message,
            )
        )

    def _add_contract_gaps(
        self,
        step_run: PlannedStepRun,
        contract: Mapping[str, Any],
    ) -> None:
        spec = workflow_step_spec_for_step_name(step_run.step_name)

        has_declared_inputs = bool(
            contract["input_keys"]
            or contract["optional_input_keys"]
            or contract["dynamic_input_families"]
        )
        has_declared_outputs = bool(
            contract["output_keys"]
            or contract["optional_output_keys"]
            or contract["dynamic_output_families"]
        )

        if (
            not has_declared_inputs
            and (
                contract["depends_on"]
                or contract["upstream_step_inputs"]
                or step_run.step_name == "postprocessing"
            )
        ):
            message = "No declared input contract is available for this step."
            if step_run.step_name == "postprocessing":
                message = (
                    "Postprocessing currently reads workspace outputs directly; inbound artifact contract is not declared."
                )
            self._append_gap(step_run, "underdeclared_inputs", message)

        if not has_declared_outputs and spec is not None and spec.outputs_class is not None:
            self._append_gap(
                step_run,
                "underdeclared_outputs",
                "No declared output contract is available for this step.",
            )


def build_static_execution_plan(
    settings: PilatesConfig,
    *,
    config_path: Optional[str] = None,
    include_postprocessing: Optional[bool] = None,
    surface: "EnabledWorkflowSurface",
) -> StaticExecutionPlan:
    include_post = (
        bool(getattr(settings, "postprocessing", None))
        if include_postprocessing is None
        else bool(include_postprocessing)
    )
    builder = _PlanBuilder(
        settings=settings,
        surface=surface,
        config_path=config_path,
        include_postprocessing=include_post,
    )
    return builder.build()


def build_static_execution_plan_from_file(
    config_path: str,
    *,
    include_postprocessing: Optional[bool] = None,
) -> StaticExecutionPlan:
    settings = load_settings_for_planning(config_path)
    from pilates.workflows.surface import build_enabled_workflow_surface

    surface = build_enabled_workflow_surface(settings)
    return build_static_execution_plan(
        settings,
        config_path=str(Path(config_path).resolve()),
        include_postprocessing=include_postprocessing,
        surface=surface,
    )
