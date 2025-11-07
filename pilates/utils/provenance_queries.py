"""
PHASE 2 IMPROVEMENT #6: Provenance Query API

Provides convenient query methods for provenance data, making it easier to:
- Find outputs of specific models
- Trace file lineage (ancestors and descendants)
- Find files by year range
- Detect broken provenance chains
- Query cross-model data flows

This API provides standard query patterns that don't require custom JSON parsing.
"""

import logging
from typing import List, Dict, Any, Optional, Set
from pilates.generic.records import PilatesRunInfo, FileRecord, ModelRunInfo

logger = logging.getLogger(__name__)


class ProvenanceQuery:
    """Query API for provenance data."""

    def __init__(self, run_info: PilatesRunInfo):
        """
        Initialize query API with provenance data.

        Args:
            run_info: PilatesRunInfo object containing all provenance records
        """
        self.run_info = run_info

    def find_outputs_of_model(
        self, model_name: str, year: int = None, iteration: int = None
    ) -> List[FileRecord]:
        """
        Find all outputs produced by a specific model.

        Args:
            model_name: Name of the model (e.g., "atlas", "activitysim")
            year: Optional filter by year
            iteration: Optional filter by iteration

        Returns:
            List of FileRecords produced by the model

        Example:
            >>> query = ProvenanceQuery(tracker.run_info)
            >>> atlas_outputs = query.find_outputs_of_model("atlas", year=2017)
            >>> for output in atlas_outputs:
            ...     print(f"  - {output.short_name}: {output.description}")
        """
        model_name = model_name.lower() if model_name else model_name

        # Find all matching model runs
        matching_runs = [
            run
            for run in self.run_info.model_runs.values()
            if run.model == model_name
            and (year is None or run.year == year)
            and (iteration is None or run.iteration == iteration)
        ]

        # Collect all output file hashes from these runs
        output_hashes = set()
        for run in matching_runs:
            output_hashes.update(run.output_record_hashes)

        # Return FileRecords
        return [
            self.run_info.file_records[h]
            for h in output_hashes
            if h in self.run_info.file_records
        ]

    def find_inputs_of_model(
        self, model_name: str, year: int = None, iteration: int = None
    ) -> List[FileRecord]:
        """
        Find all inputs consumed by a specific model.

        Args:
            model_name: Name of the model
            year: Optional filter by year
            iteration: Optional filter by iteration

        Returns:
            List of FileRecords consumed by the model
        """
        model_name = model_name.lower() if model_name else model_name

        matching_runs = [
            run
            for run in self.run_info.model_runs.values()
            if run.model == model_name
            and (year is None or run.year == year)
            and (iteration is None or run.iteration == iteration)
        ]

        input_hashes = set()
        for run in matching_runs:
            input_hashes.update(run.input_record_hashes)

        return [
            self.run_info.file_records[h]
            for h in input_hashes
            if h in self.run_info.file_records
        ]

    def trace_file_lineage(
        self, file_path: str = None, file_hash: str = None, short_name: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Trace complete lineage of a file (all ancestors and descendants).

        Args:
            file_path: Path to the file (can be relative)
            file_hash: Unique hash of the file
            short_name: Short name of the file

        Returns:
            Dict with 'file', 'ancestors', 'descendants', and 'lineage_depth' keys
            Returns None if file not found

        Example:
            >>> lineage = query.trace_file_lineage(short_name="usim_h5_updated")
            >>> print(f"File has {lineage['lineage_depth']} ancestor levels")
            >>> print(f"Ancestors: {[a.short_name for a in lineage['ancestors']]}")
        """
        # Find the file record
        target_hash = file_hash
        if not target_hash:
            target_hash = self._find_file_hash(file_path, short_name)

        if not target_hash or target_hash not in self.run_info.file_records:
            logger.warning(f"File not found: {file_path or short_name}")
            return None

        file_record = self.run_info.file_records[target_hash]

        # Find ancestors (files that contributed to this file)
        ancestors = self._find_ancestors(target_hash)

        # Find descendants (files that were created from this file)
        descendants = self._find_descendants(target_hash)

        return {
            "file": file_record,
            "ancestors": ancestors,
            "descendants": descendants,
            "lineage_depth": len(self._get_lineage_levels(target_hash)),
        }

    def _find_file_hash(
        self, file_path: str = None, short_name: str = None
    ) -> Optional[str]:
        """Find file hash by path or short_name."""
        for file_hash, file_rec in self.run_info.file_records.items():
            if short_name and file_rec.short_name == short_name:
                return file_hash
            if file_path and file_path == file_rec.file_path:
                return file_hash
        return None

    def _find_ancestors(
        self, file_hash: str, visited: Set[str] = None
    ) -> List[FileRecord]:
        """Recursively find all ancestor files."""
        if visited is None:
            visited = set()

        if file_hash in visited or file_hash not in self.run_info.file_records:
            return []

        visited.add(file_hash)
        file_record = self.run_info.file_records[file_hash]
        ancestors = []

        # Find direct ancestors via source_file_paths
        if file_record.source_file_paths:
            for source_path in file_record.source_file_paths:
                source_hash = self._find_file_hash(file_path=source_path)
                if source_hash and source_hash not in visited:
                    source_rec = self.run_info.file_records[source_hash]
                    ancestors.append(source_rec)
                    # Recursively find ancestors of ancestors
                    ancestors.extend(self._find_ancestors(source_hash, visited))

        return ancestors

    def _find_descendants(
        self, file_hash: str, visited: Set[str] = None
    ) -> List[FileRecord]:
        """Recursively find all descendant files."""
        if visited is None:
            visited = set()

        if file_hash in visited or file_hash not in self.run_info.file_records:
            return []

        visited.add(file_hash)
        file_record = self.run_info.file_records[file_hash]
        descendants = []

        # Find files that list this file as a source
        for other_hash, other_rec in self.run_info.file_records.items():
            if other_hash in visited:
                continue

            if (
                other_rec.source_file_paths
                and file_record.file_path in other_rec.source_file_paths
            ):
                descendants.append(other_rec)
                # Recursively find descendants of descendants
                descendants.extend(self._find_descendants(other_hash, visited))

        return descendants

    def _get_lineage_levels(self, file_hash: str) -> List[Set[str]]:
        """Get lineage organized by depth levels."""
        levels = []
        current_level = {file_hash}
        visited = set()

        while current_level:
            levels.append(current_level)
            visited.update(current_level)

            next_level = set()
            for fh in current_level:
                if fh not in self.run_info.file_records:
                    continue

                file_rec = self.run_info.file_records[fh]
                if file_rec.source_file_paths:
                    for source_path in file_rec.source_file_paths:
                        source_hash = self._find_file_hash(file_path=source_path)
                        if source_hash and source_hash not in visited:
                            next_level.add(source_hash)

            current_level = next_level

        return levels[1:]  # Exclude the file itself

    def find_files_modified_between_years(
        self, start_year: int, end_year: int
    ) -> List[FileRecord]:
        """
        Find all files created/modified within year range.

        Args:
            start_year: Start of year range (inclusive)
            end_year: End of year range (inclusive)

        Returns:
            List of FileRecords with years in the specified range
        """
        return [
            rec
            for rec in self.run_info.file_records.values()
            if rec.year and start_year <= rec.year <= end_year
        ]

    def find_broken_source_paths(self) -> List[Dict[str, Any]]:
        """
        Find files with source_file_paths that don't exist in file_records.

        Returns:
            List of dicts with 'file' and 'missing_source' keys

        Example:
            >>> broken = query.find_broken_source_paths()
            >>> if broken:
            ...     print(f"Warning: {len(broken)} broken source path references")
            ...     for item in broken:
            ...         print(f"  {item['file']}: missing {item['missing_source']}")
        """
        broken = []
        for file_hash, file_rec in self.run_info.file_records.items():
            if file_rec.source_file_paths:
                for source_path in file_rec.source_file_paths:
                    if not self._source_exists(source_path):
                        broken.append(
                            {"file": file_rec.short_name, "missing_source": source_path}
                        )
        return broken

    def _source_exists(self, source_path: str) -> bool:
        """Check if a source path exists in file_records."""
        return any(
            rec.file_path == source_path or source_path in rec.file_path
            for rec in self.run_info.file_records.values()
        )

    def find_model_run_chain(
        self, start_model: str = None, end_model: str = None
    ) -> List[ModelRunInfo]:
        """
        Find the chain of model runs from start_model to end_model.

        Args:
            start_model: Starting model name (e.g., "urbansim")
            end_model: Ending model name (e.g., "beam")

        Returns:
            Ordered list of ModelRunInfo objects showing the execution chain

        Example:
            >>> chain = query.find_model_run_chain("urbansim", "beam")
            >>> for run in chain:
            ...     print(f"{run.model} (year {run.year})")
        """
        # Get all model runs sorted by creation time
        sorted_runs = sorted(
            self.run_info.model_runs.values(), key=lambda r: r.created_at or ""
        )

        # Filter to range if specified
        if start_model or end_model:
            start_idx = 0
            end_idx = len(sorted_runs)

            if start_model:
                for i, run in enumerate(sorted_runs):
                    if run.model.lower() == start_model.lower():
                        start_idx = i
                        break

            if end_model:
                for i, run in enumerate(sorted_runs):
                    if run.model.lower() == end_model.lower():
                        end_idx = i + 1

            sorted_runs = sorted_runs[start_idx:end_idx]

        return sorted_runs

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics about the provenance data.

        Returns:
            Dict with counts and statistics about files, models, and runs
        """
        total_files = len(self.run_info.file_records)
        total_runs = len(self.run_info.model_runs)

        # Count files by model
        files_by_model = {}
        for file_rec in self.run_info.file_records.values():
            for model in file_rec.models:
                files_by_model[model] = files_by_model.get(model, 0) + 1

        # Count runs by model
        runs_by_model = {}
        for run in self.run_info.model_runs.values():
            runs_by_model[run.model] = runs_by_model.get(run.model, 0) + 1

        # Count files with/without source_file_paths
        files_with_sources = sum(
            1 for rec in self.run_info.file_records.values() if rec.source_file_paths
        )

        return {
            "total_files": total_files,
            "total_model_runs": total_runs,
            "models_used": list(runs_by_model.keys()),
            "files_by_model": files_by_model,
            "runs_by_model": runs_by_model,
            "files_with_source_paths": files_with_sources,
            "files_without_source_paths": total_files - files_with_sources,
            "years_covered": sorted(
                set(rec.year for rec in self.run_info.file_records.values() if rec.year)
            ),
        }
