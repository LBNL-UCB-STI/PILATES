import uuid
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class RunContext:
    """
    Manages the context for a single model run, including its unique ID
    and methods for recording provenance information.
    """
    def __init__(self, run_id: str = None, parameters: dict = None, code_version: str = None, hostname: str = None):
        """
        Initializes the RunContext.

        Args:
            run_id (str, optional): A pre-defined run ID. If None, a new UUID is generated.
            parameters (dict, optional): Dictionary of run parameters.
            code_version (str, optional): Identifier for the code version (e.g., git hash).
            hostname (str, optional): Hostname where the run is executed.
        """
        self.run_id = run_id if run_id else str(uuid.uuid4())
        self.start_time = None
        self.end_time = None
        self.status = 'initialized'
        self.parameters = parameters
        self.code_version = code_version
        self.hostname = hostname
        logger.info(f"RunContext initialized with ID: {self.run_id}")

    def record_run_start(self):
        """Records the start time and updates the status."""
        self.start_time = datetime.now()
        self.status = 'running'
        logger.info(f"Run {self.run_id} started at {self.start_time}")
        # TODO: Add database interaction to record run start in ModelRuns table

    def record_run_end(self, status: str = 'completed'):
        """Records the end time and final status."""
        self.end_time = datetime.now()
        self.status = status
        logger.info(f"Run {self.run_id} ended at {self.end_time} with status: {self.status}")
        # TODO: Add database interaction to update run end time and status in ModelRuns table

    def record_input(self, source_run_id: str, file_path: str, input_type: str = 'unknown'):
        """
        Records an input file consumed by the current run.

        Args:
            source_run_id (str): The run ID that produced this input file.
            file_path (str): The path to the input file.
            input_type (str, optional): A description of the input (e.g., 'ActivitySim Plans').
        """
        logger.info(f"Run {self.run_id} consumed input: Type='{input_type}', Path='{file_path}', SourceRun='{source_run_id}'")
        # TODO: Add database interaction to record input in ModelInputs table.
        # This might involve looking up the source_output_id from ModelOutputs
        # based on source_run_id and file_path.

    def record_output(self, output_type: str, file_path: str):
        """
        Records an output file produced by the current run.

        Args:
            output_type (str): A description of the output (e.g., 'BEAM Plans GZ').
            file_path (str): The path where the output file was saved.
        """
        logger.info(f"Run {self.run_id} produced output: Type='{output_type}', Path='{file_path}'")
        # TODO: Add database interaction to record output in ModelOutputs table.
        # The output_id would be generated here or in the database.
