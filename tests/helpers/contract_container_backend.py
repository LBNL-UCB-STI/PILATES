"""
Contract-Validating Container Backend for PILATES workflow integration tests.

Intercepts container execution via ``GenericRunner.run_container`` and Consist's
``DockerBackend`` without replacing ``Preprocessor``, ``Runner``, or ``Postprocessor``
classes and without mocking ``xarray.open_zarr``, ``openmatrix``, ``pandas.HDFStore``,
or ``pyarrow.parquet``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from consist.integrations.containers.backends import DockerBackend
from pilates.activitysim.runner import validate_activitysim_zarr_skims
from pilates.config import PilatesConfig
from pilates.generic.runner import GenericRunner
from pilates.utils import consist_runtime as cr
from tests.helpers.micro_world_fixture import (
    MICRO_HOUSEHOLD_IDS,
    MICRO_PERSON_IDS,
    MICRO_TIME_PERIODS,
    MICRO_ZONES,
    create_micro_world_tables,
    write_micro_skims_omx,
    write_micro_skims_zarr,
    write_urbansim_h5,
)


@dataclass(frozen=True)
class ContainerInvocationRecord:
    """Captured metadata for a single container invocation."""

    model_name: str
    image: str
    command: list[str]
    volumes: dict[str, dict[str, str]]
    working_dir: str | None
    environment: dict[str, str]
    input_artifacts: list[Any]
    output_paths: Any
    lineage_mode: str | None
    backend_executed: bool


class MountTable:
    """Translates container paths to host paths and checks mount permissions."""

    def __init__(self, volumes: Mapping[str, Mapping[str, str]] | Mapping[str, str]):
        # Normalize either Docker-style {host: {"bind": container, "mode": mode}}
        # or Consist-style {host: container}
        entries: list[tuple[str, Path, str]] = []
        for host_raw, spec in volumes.items():
            host_path = Path(host_raw).resolve()
            if isinstance(spec, Mapping):
                bind = str(spec["bind"]).rstrip("/") or "/"
                mode = str(spec.get("mode", "rw"))
            else:
                bind = str(spec).rstrip("/") or "/"
                mode = "rw"
            entries.append((bind, host_path, mode))
        # Sort longest container prefix first so nested mounts (e.g. /output/cache) win
        self.entries: list[tuple[str, Path, str]] = sorted(
            entries, key=lambda item: len(item[0]), reverse=True
        )

    def audit_mounts_exist(self) -> None:
        """Verify all mounted host paths exist before container execution."""
        for bind, host_path, mode in self.entries:
            if mode not in ("ro", "rw"):
                raise AssertionError(
                    f"Invalid mount mode {mode!r} for {bind} -> {host_path}"
                )
            if not host_path.exists():
                raise FileNotFoundError(
                    f"Mounted host path does not exist for container mount {bind}: {host_path}"
                )

    def resolve(self, container_path: str, *, require_writable: bool = False) -> Path:
        """Map a container path to its corresponding host path and check rw mode if required."""
        norm = container_path.rstrip("/") or "/"
        for bind, host_path, mode in self.entries:
            if norm == bind or norm.startswith(f"{bind}/"):
                if require_writable and mode != "rw":
                    raise PermissionError(
                        f"Container attempted to write to read-only mount {bind} ({container_path})"
                    )
                rel = norm[len(bind) :].lstrip("/")
                return (host_path / rel).resolve() if rel else host_path
        raise FileNotFoundError(
            f"Container path {container_path!r} is not covered by any mounted volume: "
            f"{[(b, str(h), m) for b, h, m in self.entries]}"
        )


@dataclass
class ContractValidatingContainerBackend:
    """
    Validates container launch contracts and writes schema-faithful outputs.

    Can be used as a context manager wrapping workflow execution.
    """

    delegate_through_consist: bool = True
    beam_skim_marker: float = 0.25
    invocations: list[ContainerInvocationRecord] = field(default_factory=list)
    _orig_generic_run_container: Callable[..., Any] | None = field(
        default=None, init=False, repr=False
    )
    _orig_docker_init: Callable[..., None] | None = field(
        default=None, init=False, repr=False
    )
    _orig_docker_run: Callable[..., bool] | None = field(
        default=None, init=False, repr=False
    )
    _orig_docker_digest: Callable[..., str] | None = field(
        default=None, init=False, repr=False
    )
    _active_volumes_stack: list[dict[str, dict[str, str]]] = field(
        default_factory=list, init=False, repr=False
    )
    _active_model_stack: list[str] = field(default_factory=list, init=False, repr=False)
    _backend_executed_flag: list[bool] = field(
        default_factory=list, init=False, repr=False
    )

    def __enter__(self) -> ContractValidatingContainerBackend:
        self._orig_generic_run_container = GenericRunner.__dict__["run_container"]
        self._orig_docker_init = DockerBackend.__init__
        self._orig_docker_run = DockerBackend.run
        self._orig_docker_digest = DockerBackend.resolve_image_digest

        backend_self = self

        def _spy_run_container(
            client: Any,
            settings: PilatesConfig,
            image: str,
            volumes: dict[str, dict[str, str]],
            command: str,
            model_name: str,
            working_dir: str | None = None,
            environment: dict[str, str] | None = None,
            args: Sequence[Any] | None = None,
            input_artifacts: list[Any] | None = None,
            output_paths: Any = None,
            lineage_mode: str | None = None,
        ) -> Any:
            full_cmd = GenericRunner._build_full_command(command, args)
            norm_volumes = {
                str(k): {"bind": str(v["bind"]), "mode": str(v.get("mode", "rw"))}
                for k, v in volumes.items()
            }
            backend_self._active_volumes_stack.append(norm_volumes)
            backend_self._active_model_stack.append(model_name)
            backend_self._backend_executed_flag.append(False)
            try:
                if (
                    backend_self.delegate_through_consist
                    and cr.current_tracker() is not None
                ):
                    orig_fn = backend_self._orig_generic_run_container
                    assert orig_fn is not None
                    raw_callable = orig_fn.__func__
                    result = raw_callable(
                        client=client,
                        settings=settings,
                        image=image,
                        volumes=volumes,
                        command=command,
                        model_name=model_name,
                        working_dir=working_dir,
                        environment=environment,
                        args=args,
                        input_artifacts=input_artifacts,
                        output_paths=output_paths,
                        lineage_mode=lineage_mode,
                    )
                else:
                    backend_self._validate_and_populate(
                        model_name=model_name,
                        image=image,
                        command=full_cmd,
                        volumes=norm_volumes,
                        working_dir=working_dir,
                        environment=environment or {},
                    )
                    backend_self._backend_executed_flag[-1] = True
                    result = True

                executed = backend_self._backend_executed_flag[-1]
                backend_self.invocations.append(
                    ContainerInvocationRecord(
                        model_name=model_name,
                        image=image,
                        command=list(full_cmd),
                        volumes=norm_volumes,
                        working_dir=working_dir,
                        environment=dict(environment or {}),
                        input_artifacts=list(input_artifacts or []),
                        output_paths=output_paths,
                        lineage_mode=lineage_mode,
                        backend_executed=executed,
                    )
                )
                return result
            finally:
                backend_self._active_volumes_stack.pop()
                backend_self._active_model_stack.pop()
                backend_self._backend_executed_flag.pop()

        def _fake_resolve_digest(_docker_self: DockerBackend, image: str) -> str:
            return f"sha256:contract_test_{image.replace(':', '_').replace('/', '_')}"

        def _fake_docker_run(
            _docker_self: DockerBackend,
            image: str,
            command: str | list[str],
            volumes: dict[str, str],
            env: dict[str, str],
            working_dir: str | None = None,
            volume_modes: Mapping[str, str] | None = None,
        ) -> bool:
            cmd_list = (
                [command] if isinstance(command, str) else [str(x) for x in command]
            )
            if backend_self._active_volumes_stack:
                vol_spec = backend_self._active_volumes_stack[-1]
            else:
                modes = dict(volume_modes or {})
                vol_spec = {
                    host: {"bind": bind, "mode": str(modes.get(host, "rw"))}
                    for host, bind in volumes.items()
                }
            model_name = (
                backend_self._active_model_stack[-1]
                if backend_self._active_model_stack
                else backend_self._infer_model_name(image, cmd_list)
            )
            backend_self._validate_and_populate(
                model_name=model_name,
                image=image,
                command=cmd_list,
                volumes=vol_spec,
                working_dir=working_dir,
                environment=env,
            )
            if backend_self._backend_executed_flag:
                backend_self._backend_executed_flag[-1] = True
            return True

        def _fake_docker_init(
            docker_self: DockerBackend,
            client: Any = None,
            pull_latest: bool = False,
        ) -> None:
            docker_self.client = client if client is not None else object()
            docker_self.pull_latest = pull_latest

        GenericRunner.run_container = staticmethod(_spy_run_container)  # type: ignore[assignment]
        DockerBackend.__init__ = _fake_docker_init  # type: ignore[method-assign]
        DockerBackend.resolve_image_digest = _fake_resolve_digest  # type: ignore[method-assign]
        DockerBackend.run = _fake_docker_run  # type: ignore[method-assign]
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._orig_generic_run_container is not None:
            GenericRunner.run_container = self._orig_generic_run_container  # type: ignore[assignment]
        if self._orig_docker_init is not None:
            DockerBackend.__init__ = self._orig_docker_init  # type: ignore[method-assign]
        if self._orig_docker_digest is not None:
            DockerBackend.resolve_image_digest = self._orig_docker_digest  # type: ignore[method-assign]
        if self._orig_docker_run is not None:
            DockerBackend.run = self._orig_docker_run  # type: ignore[method-assign]

    @staticmethod
    def _infer_model_name(image: str, command: Sequence[str]) -> str:
        lowered = image.lower()
        if "urbansim" in lowered:
            return "urbansim"
        if "atlas" in lowered:
            return "atlas"
        if "activitysim" in lowered:
            return "activitysim"
        if "beam" in lowered:
            return "beam"
        joined = " ".join(command)
        if "--outyear" in joined:
            return "atlas"
        if "--config=" in joined:
            return "beam"
        return "unknown"

    def _validate_and_populate(
        self,
        *,
        model_name: str,
        image: str,
        command: list[str],
        volumes: Mapping[str, Mapping[str, str]],
        working_dir: str | None,
        environment: Mapping[str, str],
    ) -> None:
        mounts = MountTable(volumes)
        mounts.audit_mounts_exist()

        if model_name == "urbansim":
            self._handle_urbansim(command, mounts)
        elif model_name == "atlas":
            self._handle_atlas(command, mounts)
        elif model_name in ("activitysim", "activitysim_numba_warmup"):
            self._handle_activitysim(
                model_name=model_name,
                command=command,
                mounts=mounts,
                environment=environment,
            )
        elif model_name in (
            "beam",
            "beam_run",
            "beam_Router",
            "beam_FullSkim",
            "beam_full_skim",
        ):
            self._handle_beam(
                model_name=model_name,
                command=command,
                mounts=mounts,
            )
        else:
            raise AssertionError(f"Unsupported container model_name: {model_name!r}")

    @staticmethod
    def _parse_flag(command: Sequence[str], flag: str) -> str:
        for idx, token in enumerate(command):
            if token == flag and idx + 1 < len(command):
                return command[idx + 1]
            if token.startswith(f"{flag}="):
                return token.split("=", 1)[1]
        raise AssertionError(
            f"Expected flag {flag!r} in container command {list(command)}"
        )

    @staticmethod
    def _parse_multi_flag(command: Sequence[str], flag: str) -> list[str]:
        values: list[str] = []
        for idx, token in enumerate(command):
            if token == flag and idx + 1 < len(command):
                values.append(command[idx + 1])
            elif token.startswith(f"{flag}="):
                values.append(token.split("=", 1)[1])
        return values

    def _handle_urbansim(self, command: list[str], mounts: MountTable) -> None:
        if "-c" not in command:
            raise AssertionError(f"UrbanSim command missing '-c' flag: {command}")
        region_id = self._parse_flag(command, "-r")
        in_year = int(self._parse_flag(command, "-i"))
        forecast_year = int(self._parse_flag(command, "-y"))
        freq = int(self._parse_flag(command, "-f"))
        if forecast_year < in_year or freq <= 0:
            raise AssertionError(
                f"Invalid UrbanSim years/freq: in_year={in_year}, forecast_year={forecast_year}, freq={freq}"
            )

        data_dir = mounts.resolve("/usim/data", require_writable=True)
        base_h5 = data_dir / f"custom_mpo_{region_id}_model_data.h5"
        skims_omx = data_dir / f"skims_mpo_{region_id}.omx"
        if not base_h5.exists():
            raise FileNotFoundError(
                f"UrbanSim input HDF5 missing in container mount: {base_h5}"
            )
        if not skims_omx.exists():
            raise FileNotFoundError(
                f"UrbanSim skims OMX missing in container mount: {skims_omx}"
            )

        # Read actual input tables from base_h5 to preserve referential integrity
        tables = create_micro_world_tables(base_year=in_year)
        with pd.HDFStore(str(base_h5), mode="r") as store:
            keys = set(store.keys())
            for name in ("households", "persons", "blocks", "jobs"):
                year_key = f"/{in_year}/{name}"
                root_key = f"/{name}"
                if year_key in keys:
                    tables[name] = store[year_key]
                elif root_key in keys:
                    tables[name] = store[root_key]
                else:
                    raise AssertionError(
                        f"UrbanSim base HDF5 {base_h5} missing required table {name!r} (keys={sorted(keys)})"
                    )

        output_h5 = data_dir / f"model_data_{forecast_year}.h5"
        write_urbansim_h5(output_h5, tables, years=(in_year, forecast_year))

    def _handle_atlas(self, command: list[str], mounts: MountTable) -> None:
        freq = int(self._parse_flag(command, "--freq"))
        out_year = int(self._parse_flag(command, "--outyear"))
        beamac = int(self._parse_flag(command, "--beamac"))
        in_year = out_year - freq

        input_root = mounts.resolve("/atlas/input", require_writable=True)
        output_root = mounts.resolve("/atlas/output", require_writable=True)

        # AtlasPreprocessor stages inputs under year{current_year}. Check out_year and in_year.
        candidate_dirs = [
            input_root / f"year{out_year}",
            input_root / f"year{in_year}",
        ]
        staged_dir = next(
            (d for d in candidate_dirs if (d / "households.csv").exists()), None
        )
        if staged_dir is None:
            raise FileNotFoundError(
                f"ATLAS staged households.csv not found in any candidate dir: {candidate_dirs}"
            )

        for req_file in (
            "households.csv",
            "blocks.csv",
            "persons.csv",
            "residential.csv",
            "jobs.csv",
        ):
            fpath = staged_dir / req_file
            if not fpath.exists() or fpath.stat().st_size == 0:
                raise FileNotFoundError(
                    f"ATLAS required input missing or empty: {fpath}"
                )

        if beamac > 0:
            acc_candidates = list(staged_dir.glob("accessibility_*_tract.csv"))
            if not acc_candidates:
                raise FileNotFoundError(
                    f"ATLAS beamac={beamac} requires accessibility_<year>_tract.csv in {staged_dir}"
                )

        hh_df = pd.read_csv(staged_dir / "households.csv")
        if "household_id" in hh_df.columns:
            hh_ids = hh_df["household_id"].astype(int).tolist()
        else:
            hh_ids = list(MICRO_HOUSEHOLD_IDS)

        output_root.mkdir(parents=True, exist_ok=True)
        hhv_df = pd.DataFrame(
            {
                "household_id": hh_ids,
                "nvehicles": [1 if idx % 2 == 0 else 2 for idx in range(len(hh_ids))],
            }
        )
        hhv_df.to_csv(output_root / f"householdv_{out_year}.csv", index=False)

        veh_rows: list[dict[str, Any]] = []
        for idx, hh_id in enumerate(hh_ids):
            n_veh = int(hhv_df.loc[idx, "nvehicles"])
            for v_id in range(1, n_veh + 1):
                veh_rows.append(
                    {
                        "household_id": int(hh_id),
                        "vehicle_id": v_id,
                        "modelyear": 2018 + (idx % 3),
                        "bodytype": "car" if v_id == 1 else "suv",
                        "adopt_fuel": "conv" if idx % 2 == 0 else "ev",
                        "pred_power": "ICE" if idx % 2 == 0 else "BEV",
                        "ownlease": 1,
                        "MainDriver_id": int(
                            MICRO_PERSON_IDS[idx % len(MICRO_PERSON_IDS)]
                        ),
                        "year": out_year,
                    }
                )
        pd.DataFrame(veh_rows).to_csv(
            output_root / f"vehicles_{out_year}.csv", index=False
        )

        # Write continuation .RData files in /atlas/input/year{out_year}
        out_year_dir = input_root / f"year{out_year}"
        out_year_dir.mkdir(parents=True, exist_ok=True)
        (out_year_dir / "vehicles_output.RData").write_bytes(b"RDATA_VEHICLES_STUB")
        (out_year_dir / "households_output.RData").write_bytes(b"RDATA_HOUSEHOLDS_STUB")

    def _handle_activitysim(
        self,
        *,
        model_name: str,
        command: list[str],
        mounts: MountTable,
        environment: Mapping[str, str],
    ) -> None:
        data_container_dir = self._parse_flag(command, "-d")
        output_container_dir = self._parse_flag(command, "-o")
        config_container_dirs = self._parse_multi_flag(command, "-c")
        persist_sharrow_flag = "--persist-sharrow-cache" in command

        data_dir = mounts.resolve(data_container_dir, require_writable=False)
        output_dir = mounts.resolve(output_container_dir, require_writable=True)

        # Inspect staged settings.yaml across -c config directories
        merged_settings: dict[str, Any] = {}
        for cfg_cdir in config_container_dirs:
            if not cfg_cdir.startswith("/"):
                continue
            cfg_hdir = mounts.resolve(cfg_cdir, require_writable=False)
            settings_file = cfg_hdir / "settings.yaml"
            if settings_file.exists():
                with settings_file.open("r", encoding="utf-8") as fh:
                    loaded = yaml.safe_load(fh) or {}
                if isinstance(loaded, dict):
                    merged_settings.update(loaded)

        skim_file_type = str(merged_settings.get("skim_file_type", "omx")).lower()

        # Verify required input tables & skims in -d (may be .parquet or .csv)
        def _read_asim_input_table(stem: str) -> pd.DataFrame:
            pq_path = data_dir / f"{stem}.parquet"
            csv_path = data_dir / f"{stem}.csv"
            if pq_path.exists():
                return pd.read_parquet(pq_path)
            if csv_path.exists():
                return pd.read_csv(csv_path)
            raise FileNotFoundError(
                f"ActivitySim required input table missing in {data_container_dir}: "
                f"expected {pq_path} or {csv_path}"
            )

        hh_df = _read_asim_input_table("households")
        persons_df = _read_asim_input_table("persons")
        lu_df = _read_asim_input_table("land_use")

        cache_zarr_cpath = f"{output_container_dir.rstrip('/')}/cache/skims.zarr"
        cache_zarr_hpath = mounts.resolve(cache_zarr_cpath, require_writable=False)
        data_zarr_hpath = data_dir / "skims.zarr"
        omx_path = data_dir / "skims.omx"

        if cache_zarr_hpath.exists() or data_zarr_hpath.exists():
            active_zarr = (
                cache_zarr_hpath if cache_zarr_hpath.exists() else data_zarr_hpath
            )
            rejection = validate_activitysim_zarr_skims(active_zarr)
            if rejection is not None:
                raise AssertionError(
                    f"ActivitySim input skims.zarr failed validation: {rejection}"
                )
        elif skim_file_type == "zarr":
            raise FileNotFoundError(
                f"ActivitySim skim_file_type='zarr' requires skims.zarr at {cache_zarr_hpath} or {data_zarr_hpath}"
            )
        elif not omx_path.exists():
            raise FileNotFoundError(
                f"ActivitySim requires skims.omx at {omx_path} or skims.zarr at {cache_zarr_hpath}"
            )

        # If Numba cache is mounted and sharrow is active, populate numba compiled file
        numba_cache_cdir = environment.get("XDG_CACHE_HOME", "/app/numba_cache")
        try:
            numba_cache_hdir = mounts.resolve(numba_cache_cdir, require_writable=True)
            numba_subdir = numba_cache_hdir / "numba"
            numba_subdir.mkdir(parents=True, exist_ok=True)
            (numba_subdir / "compiled.nbc").write_bytes(b"NUMBA_COMPILED_CACHE_STUB")
        except FileNotFoundError:
            pass

        # CRITICAL CONTRACT: Only write output/cache/skims.zarr when --persist-sharrow-cache is in command!
        if persist_sharrow_flag:
            cache_cpath = f"{output_container_dir.rstrip('/')}/cache"
            cache_hdir = mounts.resolve(cache_cpath, require_writable=True)
            cache_hdir.mkdir(parents=True, exist_ok=True)
            zarr_out = cache_hdir / "skims.zarr"
            if not zarr_out.exists():
                write_micro_skims_zarr(zarr_out)

        if model_name == "activitysim_numba_warmup":
            return

        hh_ids = (
            hh_df.index.astype(int).tolist()
            if hh_df.index.name == "household_id"
            else hh_df["household_id"].astype(int).tolist()
        )
        person_ids = (
            persons_df.index.astype(int).tolist()
            if persons_df.index.name == "person_id"
            else persons_df["person_id"].astype(int).tolist()
        )
        zones = (
            lu_df.index.astype(int).tolist()
            if lu_df.index.name in ("TAZ", "zone_id")
            else list(MICRO_ZONES)
        )

        final_pipeline_dir = output_dir / "final_pipeline"
        final_pipeline_dir.mkdir(parents=True, exist_ok=True)

        def _write_table(table_name: str, df: pd.DataFrame) -> None:
            tdir = final_pipeline_dir / table_name
            tdir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(tdir / "final.parquet", index=True)

        # Ensure households and persons preserve columns needed by ActivitysimPostprocessor
        hh_out = hh_df.copy()
        if "household_id" not in hh_out.columns:
            hh_out = hh_out.reset_index()
        if "TAZ" not in hh_out.columns:
            hh_out["TAZ"] = [zones[i % len(zones)] for i in range(len(hh_out))]
        if "hhsize" not in hh_out.columns:
            hh_out["hhsize"] = 2
        if "num_workers" not in hh_out.columns:
            hh_out["num_workers"] = 1
        if "auto_ownership" not in hh_out.columns:
            hh_out["auto_ownership"] = 1

        persons_out = persons_df.copy()
        if "person_id" not in persons_out.columns:
            persons_out = persons_out.reset_index()
        if "household_id" not in persons_out.columns:
            persons_out["household_id"] = [
                hh_ids[i % len(hh_ids)] for i in range(len(persons_out))
            ]
        if "PNUM" not in persons_out.columns:
            persons_out["PNUM"] = 1
        if "workplace_taz" not in persons_out.columns:
            persons_out["workplace_taz"] = [
                zones[(i + 1) % len(zones)] for i in range(len(persons_out))
            ]
        if "school_taz" not in persons_out.columns:
            persons_out["school_taz"] = -1

        lu_out = lu_df.copy()
        if lu_out.index.name is None and "TAZ" in lu_out.columns:
            lu_out = lu_out.set_index("TAZ")

        tours_df = pd.DataFrame(
            {
                "tour_id": [1, 2, 3],
                "person_id": person_ids[:3],
                "household_id": hh_ids[:3],
                "tour_type": ["work", "work", "shop"],
                "tour_category": ["mandatory", "mandatory", "non_mandatory"],
                "origin": [zones[0], zones[0], zones[1]],
                "destination": [zones[1], zones[2], zones[2]],
                "start": [8, 8, 13],
                "end": [17, 17, 15],
                "tour_mode": ["DRIVEALONEFREE", "WALK_LOC", "SHARED2FREE"],
            }
        ).set_index("tour_id")

        trips_df = pd.DataFrame(
            {
                "trip_id": [1, 2, 3, 4, 5, 6],
                "tour_id": [1, 1, 2, 2, 3, 3],
                "person_id": [
                    person_ids[0],
                    person_ids[0],
                    person_ids[1],
                    person_ids[1],
                    person_ids[2],
                    person_ids[2],
                ],
                "household_id": [
                    hh_ids[0],
                    hh_ids[0],
                    hh_ids[1],
                    hh_ids[1],
                    hh_ids[2],
                    hh_ids[2],
                ],
                "primary_purpose": ["work", "work", "work", "work", "shop", "shop"],
                "purpose": ["work", "Home", "work", "Home", "shop", "Home"],
                "origin": [zones[0], zones[1], zones[0], zones[2], zones[1], zones[2]],
                "destination": [
                    zones[1],
                    zones[0],
                    zones[2],
                    zones[0],
                    zones[2],
                    zones[1],
                ],
                "depart": [8.0, 17.0, 8.5, 17.5, 13.0, 15.0],
                "trip_mode": [
                    "DRIVEALONEFREE",
                    "DRIVEALONEFREE",
                    "WALK_LOC",
                    "WALK_LOC",
                    "SHARED2FREE",
                    "SHARED2FREE",
                ],
                "outbound": [True, False, True, False, True, False],
            }
        ).set_index("trip_id")

        jtp_df = pd.DataFrame(
            {
                "participant_id": [1],
                "tour_id": [3],
                "household_id": [hh_ids[0]],
                "person_id": [person_ids[0]],
                "participant_num": [1],
            }
        ).set_index("participant_id")

        plans_df = pd.DataFrame(
            {
                "PlanElementIndex": [1, 2, 3, 1, 2, 3],
                "person_id": [
                    person_ids[0],
                    person_ids[0],
                    person_ids[0],
                    person_ids[1],
                    person_ids[1],
                    person_ids[1],
                ],
                "trip_id": [np.nan, 1.0, np.nan, np.nan, 3.0, np.nan],
                "tour_id": [np.nan, 1.0, 1.0, np.nan, 2.0, 2.0],
                "ActivityElement": [
                    "activity",
                    "leg",
                    "activity",
                    "activity",
                    "leg",
                    "activity",
                ],
                "ActivityType": ["Home", None, "work", "Home", None, "work"],
                "x": [-122.40, None, -122.35, -122.40, None, -122.30],
                "y": [37.78, None, 37.80, 37.78, None, 37.82],
                "departure_time": [8.0, 8.0, 17.0, 8.5, 8.5, 17.5],
                "leg_mode": [None, "DRIVEALONEFREE", None, None, "WALK_LOC", None],
            }
        )

        vehicles_df = pd.DataFrame(
            {
                "vehicle_id": [1, 2, 3],
                "household_id": hh_ids[:3],
                "vehicle_type": ["car", "suv", "car"],
            }
        ).set_index("vehicle_id")

        checkpoints_df = pd.DataFrame(
            {"checkpoint_name": ["final"], "timestamp": ["2026-01-01T00:00:00"]}
        )

        _write_table("households", hh_out)
        _write_table("persons", persons_out)
        _write_table("land_use", lu_out)
        _write_table("tours", tours_df)
        _write_table("trips", trips_df)
        _write_table("joint_tour_participants", jtp_df)
        _write_table("plans", plans_df)
        _write_table("beam_plans", plans_df)
        _write_table("vehicles", vehicles_df)
        _write_table("checkpoints", checkpoints_df)

        with pd.HDFStore(str(output_dir / "pipeline.h5"), mode="w") as store:
            store.put("/households", hh_out, format="fixed")
            store.put("/persons", persons_out, format="fixed")

    def _handle_beam(
        self,
        *,
        model_name: str,
        command: list[str],
        mounts: MountTable,
    ) -> None:
        config_cpath = self._parse_flag(command, "--config")
        config_hpath = mounts.resolve(config_cpath, require_writable=False)
        if not config_hpath.exists():
            raise FileNotFoundError(
                f"BEAM config file missing at container path {config_cpath}: {config_hpath}"
            )

        conf_text = config_hpath.read_text(encoding="utf-8")

        def _extract_conf_val(key: str, default: str = "") -> str:
            pattern = rf"^\s*{re.escape(key)}\s*=\s*(.+?)\s*$"
            matches = re.findall(pattern, conf_text, flags=re.MULTILINE)
            if not matches:
                return default
            raw = matches[-1].strip()
            return raw.strip('"')

        input_cdir = _extract_conf_val("beam.inputDirectory", "/app/input/sfbay")
        output_cdir = _extract_conf_val(
            "beam.outputs.baseOutputDirectory", "/app/output/sfbay"
        )
        sim_name = _extract_conf_val("beam.agentsim.simulationName", "micro-sim")
        skim_fmt = _extract_conf_val(
            "beam.Calibration.activitySimSkimmer.fileFormat", "zarr"
        ).lower()
        skim_base = _extract_conf_val(
            "beam.Calibration.activitySimSkimmer.fileBaseName", "skimsActivitySimOD"
        )

        def _resolve_interpolated_cpath(expr: str) -> str:
            val = expr.replace("${beam.inputDirectory}", input_cdir)
            r5_dir = f"{input_cdir.rstrip('/')}/r5-simple-no-local-tr"
            val = val.replace("${beam.routing.r5.directory}", r5_dir)
            val = val.replace('"', "").strip()
            return val

        # Audit key input files referenced in beam.conf
        for conf_key in (
            "beam.routing.r5.osmFile",
            "beam.physsim.inputNetworkFilePath",
            "beam.agentsim.agents.vehicles.vehicleTypesFilePath",
            "beam.agentsim.agents.vehicles.vehiclesFilePath",
            "beam.exchange.scenario.folder",
            "beam.Calibration.activitySimSkimmer.tazFilePath",
        ):
            raw_val = _extract_conf_val(conf_key, "")
            if raw_val:
                cpath = _resolve_interpolated_cpath(raw_val)
                if cpath.startswith("/"):
                    hpath = mounts.resolve(cpath, require_writable=False)
                    if not hpath.exists():
                        raise FileNotFoundError(
                            f"BEAM config key {conf_key} points to missing path {cpath} -> {hpath}"
                        )

        warmstart_type = _extract_conf_val("beam.warmStart.type", "disabled")
        warmstart_path = _extract_conf_val("beam.warmStart.path", "")
        if warmstart_type != "disabled" and warmstart_path.startswith("/"):
            ws_hpath = mounts.resolve(warmstart_path, require_writable=False)
            if not ws_hpath.exists():
                raise FileNotFoundError(
                    f"BEAM warmStart.path does not exist: {warmstart_path} -> {ws_hpath}"
                )

        if model_name == "beam_Router":
            return

        output_hroot = mounts.resolve(output_cdir, require_writable=True)
        run_dir = output_hroot / f"{sim_name}__2026-01-01_00-00-00"
        it0_dir = run_dir / "ITERS" / "it.0"
        it0_dir.mkdir(parents=True, exist_ok=True)

        # 1. Write 0.events.parquet with multiple event types ("ModeChoice", "PathTraversal")
        events_df = pd.DataFrame(
            {
                "time": [28800.0, 28900.0, 29500.0, 61200.0],
                "type": ["ModeChoice", "PathTraversal", "ModeChoice", "PathTraversal"],
                "person": [1001, 1001, 1002, 1001],
                "vehicle": ["1", "1", "body-1002", "1"],
                "mode": ["car", "car", "walk", "car"],
                "length": [2500.0, 2500.0, 600.0, 2500.0],
                "links": ["1,2", "1,2", "3", "2,1"],
                "linkTravelTime": ["30.0,45.0", "30.0,45.0", "60.0", "40.0,35.0"],
            }
        )
        events_df.to_parquet(it0_dir / "0.events.parquet", index=False)

        # 2. Write 0.linkstats.csv.gz
        linkstats_df = pd.DataFrame(
            {
                "link": [1, 2, 3],
                "hour": [8.0, 8.0, 17.0],
                "length": [1000.0, 1500.0, 1200.0],
                "freespeed": [15.0, 20.0, 15.0],
                "capacity": [1800.0, 2000.0, 1800.0],
                "stat": ["AVG", "AVG", "AVG"],
                "volume": [120.0, 150.0, 95.0],
                "traveltime": [75.0, 85.0, 90.0],
            }
        )
        linkstats_df.to_csv(
            it0_dir / "0.linkstats.csv.gz", index=False, compression="gzip"
        )

        # 3. Write 0.plans.csv.gz (required by BeamPostprocessor.merge_Select_beam_plan)
        beam_plans_df = pd.DataFrame(
            {
                "personId": [1001, 1001, 1001, 1002, 1002, 1002],
                "planIndex": [0, 0, 0, 0, 0, 0],
                "planScore": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "planSelected": [True, True, True, True, True, True],
                "planElementType": [
                    "activity",
                    "leg",
                    "activity",
                    "activity",
                    "leg",
                    "activity",
                ],
                "planElementIndex": [1, 2, 3, 1, 2, 3],
                "activityType": ["Home", None, "work", "Home", None, "work"],
                "activityLocationX": [-122.40, None, -122.35, -122.40, None, -122.30],
                "activityLocationY": [37.78, None, 37.80, 37.78, None, 37.82],
                "activityEndTime": [28800.0, None, 61200.0, 30600.0, None, 63000.0],
                "legMode": [None, "car", None, None, "walk_transit", None],
                "legDepartureTime": [None, 28800.0, None, None, 30600.0, None],
                "legTravelTime": [None, 600.0, None, None, 900.0, None],
                "tripId": [np.nan, 1.0, np.nan, np.nan, 3.0, np.nan],
                "tourId": [np.nan, 1.0, 1.0, np.nan, 2.0, 2.0],
            }
        )
        beam_plans_df.to_csv(
            it0_dir / "0.plans.csv.gz", index=False, compression="gzip"
        )

        # 4. Write top-level warmstart/scenario outputs in run_dir
        beam_plans_df.to_csv(run_dir / "plans.csv.gz", index=False, compression="gzip")
        for top_name in (
            "vehicles.csv.gz",
            "households.csv.gz",
            "output_persons.csv.gz",
            "population.csv.gz",
            "network.csv.gz",
        ):
            pd.DataFrame({"id": [1, 2, 3]}).to_csv(
                run_dir / top_name, index=False, compression="gzip"
            )

        # 5. Write ActivitySim OD skims according to configured format
        zarr_skim_path = it0_dir / f"0.{skim_base}_current.zarr"
        omx_skim_path = it0_dir / f"0.{skim_base}_current.omx"
        csv_skim_path = it0_dir / f"0.{skim_base}_current.csv.gz"

        if skim_fmt == "zarr":
            write_micro_skims_zarr(
                zarr_skim_path,
                zones=MICRO_ZONES,
                time_periods=MICRO_TIME_PERIODS,
                marker_value=self.beam_skim_marker,
            )
            write_micro_skims_omx(omx_skim_path, zones=MICRO_ZONES)
        elif skim_fmt == "omx":
            write_micro_skims_omx(omx_skim_path, zones=MICRO_ZONES)
        else:
            pd.DataFrame(
                {
                    "timePeriod": ["AM", "PM"],
                    "pathType": ["SOV", "SOV"],
                    "origin": ["1", "1"],
                    "destination": ["2", "2"],
                    "TIME_minutes": [10.0, 12.0],
                    "DIST_miles": [2.0, 2.0],
                    "VTOLL_cost": [0.5, 0.5],
                    "IVT_minutes": [0.0, 0.0],
                }
            ).to_csv(csv_skim_path, index=False, compression="gzip")
            write_micro_skims_omx(omx_skim_path, zones=MICRO_ZONES)

        # Ridehail skims
        pd.DataFrame(
            {
                "tazId": ["1", "2", "3"],
                "hour": [8, 8, 8],
                "waitTime": [180.0, 200.0, 220.0],
                "costPerMile": [1.5, 1.5, 1.5],
            }
        ).to_csv(it0_dir / "0.skimsRidehail.csv.gz", index=False, compression="gzip")
