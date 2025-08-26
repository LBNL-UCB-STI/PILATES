<p align="center"><img src="logo_multi.png" width="700" alt="PILATES Logo"></p>

**PILATES** (**P**latform for **I**ntegrated **L**anduse **A**nd **T**ransportation **E**xperiments and **S**imulation) is a framework for orchestrating containerized microsimulation applications to model the co-evolution of land use and transportation systems. It is designed for long-term regional forecasting, enabling researchers and planners to simulate complex urban dynamics over time by linking specialized models that operate at different time scales.

Rather than tightly coupling models within a single software process, PILATES orchestrates them in a containerized environment. This modular structure allows it to leverage the behavioral sophistication of existing models with minimal modification.

-----

## Key Features

  * **Integrated Microsimulation**: Orchestrate multiple containerized models like UrbanSim, ActivitySim, and BEAM in a cohesive workflow.
  * **Flexible Workflows**: Configure various combinations of land use and transportation models and set their execution frequency to study different feedback loops.
  * **State Management**: Reliably track and persist the simulation state across different model runs and time steps.
  * **Provenance Tracking**: Record a detailed lineage of all data transformations and model executions for reproducibility.
  * **HPC Ready**: Designed to run efficiently on high-performance computing (HPC) environments.
  * **Dual Storage Database System**: Store and query simulation data in analytical databases for improved performance and data provenance.

-----

## Database Integration & Dual Storage

PILATES includes a powerful dual storage system that enables analytical database backends for improved performance, scalability, and data provenance. This system is particularly valuable for large-scale simulations and cloud deployments.

### Key Database Features

  * **Dual Storage Architecture**: Automatically stores both raw UrbanSim data and processed ActivitySim inputs, preserving the expensive preprocessing results while maintaining access to original data.
  * **Database Input Mode**: ActivitySim can read input data directly from the database instead of H5 files, eliminating preprocessing overhead and enabling faster startup times.
  * **Parallel Uploads**: Efficiently upload large datasets using parallel table operations for improved performance.
  * **OpenLineage Integration**: Complete data lineage tracking with OpenLineage metadata for full reproducibility.
  * **Multiple Backends**: Currently supports DuckDB with extensible architecture for other analytical databases.

### Supported Workflows

1. **H5 to Database Extraction**: Convert existing UrbanSim H5 files to database format with dual storage
   ```bash
   python pilates/utils/h5_to_database.py --h5-file /path/to/urbansim_data.h5 --settings settings.yaml
   ```

2. **Run Upload from run_info.json**: Upload completed simulation results including ActivitySim CSV inputs
   ```bash
   python pilates/utils/upload_runs.py --run-info /path/to/run_info.json --settings settings.yaml
   ```

3. **Database Input Mode**: Configure ActivitySim to read from database instead of H5 files
   ```yaml
   # settings.yaml
   database:
     enabled: true
     type: duckdb
     path: pilates/database/region_data.duckdb
   
   activitysim_database:
     enabled: true
     use_processed_data: true
     year: 2017
   ```

### Performance Benefits

- **Faster ActivitySim Startup**: Skip expensive H5 preprocessing by reading pre-processed data from database
- **Parallel Processing**: Multiple tables uploaded simultaneously for improved throughput
- **Cloud Ready**: Database backends enable scalable cloud deployments
- **Memory Efficiency**: Query only needed data subsets instead of loading entire H5 files

For detailed database setup and usage instructions, see [docs/database-setup.md](docs/database-setup.md).

-----

## Integrated Simulation Models

PILATES integrates several leading simulation models to create a comprehensive forecasting tool. Each model handles a different aspect of the urban system.

  * **[UrbanSim](https://github.com/UDST/urbansim)**: A microsimulation platform for modeling the long-term evolution of metropolitan areas. It simulates the choices of households and businesses regarding location, as well as the decisions of real estate developers, to predict changes in land use, demographics, and economic conditions over periods of years or decades.

  * **[ATLAS](https://doi.org/10.1080/03081060.2024.2353784)**: A household vehicle fleet microsimulation model that focuses on fleet dynamics, including vehicle purchase, replacement, and technology choice (e.g., electric vs. internal combustion engine vehicles). ATLAS provides detailed insights into fleet turnover and technology adoption over time.

  * **[ActivitySim](https://github.com/ActivitySim/activitysim)**: An agent-based travel demand model that simulates the daily activities and travel patterns of a synthetic population. It generates individual travel plans, including the purpose, destination, time of day, and mode of travel for each trip, which serve as the demand input for the transport network model.

  * **[BEAM](https://github.com/LBNL-UCB-STI/beam)**: The **B**ehavior, **E**nergy, **A**utonomy, and **M**obility modeling framework. BEAM is an agent-based transportation simulation model that executes the travel plans generated by ActivitySim on a detailed road network. It simulates traffic congestion, transit operations, and the use of emerging modes like ride-hail, producing network performance metrics (skims) that are fed back to the demand models.

-----

## Simulation Scenarios

PILATES can be configured to run various simulation scenarios, from simple network analysis to fully integrated long-term forecasting.

1.  **BEAM Only**: This configuration is useful for detailed analysis of network performance using a fixed set of travel plans. It allows for studying the impacts of new infrastructure, operational strategies, or mobility services without the complexity of a dynamic demand model.

2.  **ActivitySim + BEAM**: This represents a standard agent-based travel demand model. ActivitySim generates daily travel demand, which BEAM then simulates on the network. The resulting network travel times are fed back to ActivitySim in a loop until a stable equilibrium is reached, providing a detailed snapshot of a typical day.

3.  **UrbanSim + ActivitySim + BEAM**: This workflow enables the study of feedback between land use and transportation. UrbanSim simulates long-term changes (e.g., over a 5-year period), the results of which are used to generate daily travel demand in ActivitySim/BEAM. The resulting changes in accessibility from the transport model can then inform the next UrbanSim simulation period.

4.  **UrbanSim + ATLAS + ActivitySim + BEAM**: This is the most comprehensive configuration, capturing the co-evolution of land use, transport, and vehicle technology. It adds ATLAS to the simulation loop to model how changes in the transportation system and household location influence the types of vehicles people own, which in turn affects energy consumption and emissions.

-----

## Getting Started

Follow these steps to get PILATES up and running on your local machine.

### 1\. Prerequisites

  * A container runtime: **Docker** or **Singularity**
  * **Anaconda** or **Miniconda** to manage the Python environment

### 2\. Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/LBNL-UCB-STI/PILATES.git
    cd PILATES
    ```

2.  **Create and activate the conda environment:**

    ```bash
    conda env create -f environment.yml
    conda activate pilates
    ```
3.  **Download the input data:**

 Various types of raw input data are required for the different models. See [lawrencium-setup.md](lawrencium-setup.md#download-data) for instructions on how to download and prepare the input data.
### 3\. Configuration

1.  Open the `settings.yaml` file.
2.  Set your container preference: `container_manager: "docker"` or `"singularity"`.
3.  Update region-specific parameters and computational settings (`num_processors`, `chunk_size`, etc.) as needed.
4.  (Optional) Configure database integration for improved performance:
    ```yaml
    database:
      enabled: true
      type: duckdb
      path: pilates/database/region_data.duckdb
    ```

### 4\. Running a Simulation

Execute the main script from the root directory. Use the `-p` flag to pull the latest container images before the first run.

```bash
python run.py -v -p
```

-----

## Advanced Usage

### Database Workflows

Convert existing UrbanSim H5 files for faster ActivitySim processing:

```bash
# Extract H5 to database with dual storage
python pilates/utils/h5_to_database.py --h5-file /path/to/urbansim_data.h5 --settings settings.yaml

# Upload completed runs to database
python pilates/utils/upload_runs.py --run-dir /path/to/run_output --settings settings.yaml
```

### Background Process

To run PILATES as a background process, use `nohup`:

```bash
nohup python run.py -v &
```

The output will be saved to `nohup.out`.

### HPC Execution

For HPC environments, specialized scripts are available in the `hpc/` directory.

```bash
cd hpc
./job_runner.sh [options]
```

For detailed instructions on setting up PILATES on the Lawrencium cluster, see [`lawrencium-setup.md`](https://www.google.com/search?q=lawrencium-setup.md).

-----

## Preprocessor/Runner/Postprocessor Pattern

PILATES uses a consistent preprocessor/runner/postprocessor pattern to implement model execution. This pattern provides a structured approach to handling data preparation, model execution, and output processing for various simulation models.

### Pattern Overview

The pattern consists of three main components:

1. **Preprocessor**: Prepares input data for the model by:
   - Copying data to mutable locations
   - Recording provenance of input files
   - Performing any necessary data transformations

2. **Runner**: Executes the core model functionality by:
   - Running containerized simulations (Docker or Singularity)
   - Managing input/output data flow
   - Tracking model execution progress

3. **Postprocessor**: Processes the raw outputs from the model by:
   - Validating results
   - Transforming data into standard formats
   - Recording provenance of output files

### Workflow Diagram

```
WorkflowState --> Initializes --> Preprocessor
Preprocessor --> Prepared Data --> Runner
Runner --> Raw Outputs --> Postprocessor
Postprocessor --> Processed Data --> Workspace
Workspace --> Stores Data --> Provenance Tracker
Provenance Tracker --> Next Stage --> WorkflowState
```

### Implementing for a New Model

To implement the pattern for a new model, you need to create three classes that extend the generic abstract classes:

1. **Preprocessor Implementation**:
   - Extend `GenericPreprocessor`
   - Implement `copy_data_to_mutable_location()` to copy model-specific data
   - Implement `preprocess()` to prepare data for the model run

2. **Runner Implementation**:
   - Extend `GenericRunner`
   - Implement `run()` to execute the model using containers
   - Handle model-specific container configuration

3. **Postprocessor Implementation**:
   - Extend `GenericPostprocessor`
   - Implement `postprocess()` to process raw outputs
   - Validate and transform data into standard formats

### Example Implementation

For a new model called "my_model", you would create:

```python
# my_model/preprocessor.py
from pilates.generic.preprocessor import GenericPreprocessor

class MyModelPreprocessor(GenericPreprocessor):
    def copy_data_to_mutable_location(self, settings, output_dir):
        # Implement data copying logic
        pass

    def preprocess(self, workspace, previous_records=RecordStore()):
        # Implement preprocessing logic
        pass

# my_model/runner.py
from pilates.generic.runner import GenericRunner

class MyModelRunner(GenericRunner):
    def run(self, store, workspace):
        # Implement model execution logic
        pass

# my_model/postprocessor.py
from pilates.generic.postprocessor import GenericPostprocessor

class MyModelPostprocessor(GenericPostprocessor):
    def postprocess(self, raw_outputs, runInfo, workspace, model_run_hash):
        # Implement postprocessing logic
        pass
```

### Registering the Model

Register the model classes in the `ModelFactory`:

```python
# pilates/generic/model_factory.py
class ModelFactory:
    _registry = {
        # Other models...
        "my_model": {
            "preprocessor": MyModelPreprocessor,
            "runner": MyModelRunner,
            "postprocessor": MyModelPostprocessor,
        }
    }
```

### Component Interaction

1. **Preprocessor-Workspace Interaction**:
   - The preprocessor uses the workspace to get paths for mutable data directories
   - It copies input data to these locations and records the files in the provenance tracker

2. **Runner-Workspace Interaction**:
   - The runner uses the workspace to access prepared input data
   - It executes the model in a container environment
   - Outputs are stored back in the workspace

3. **Postprocessor-Workspace Interaction**:
   - The postprocessor reads raw outputs from the workspace
   - Processes them into standardized formats
   - Records the processed outputs in the provenance tracker

### Provenance Tracking

Provenance tracking is integrated throughout the pattern:

- Preprocessors record input file locations and metadata
- Runners track execution parameters and container information
- Postprocessors document output file generation and transformations

This comprehensive tracking enables full reproducibility of simulations.

### Benefits of the Pattern

1. **Modularity**: Each component can be developed and tested independently
2. **Reusability**: Components can be reused across different models
3. **Maintainability**: Clear separation of concerns improves code organization
4. **Provenance**: Integrated tracking supports reproducibility

This pattern is a key enabler of PILATES' flexibility and extensibility, allowing it to integrate diverse simulation models while maintaining a consistent framework for execution and data management.

-----

## Contributing and Support

We welcome contributions! For bug reports, feature requests, and support questions, please use the [GitHub issue tracker](https://www.google.com/search?q=https://github.com/LBNL-UCB-STI/PILATES/issues).

Please see our (forthcoming) `CONTRIBUTING.md` for guidelines on making changes and our code style.

### How to Contribute

1.  **Fork** the repository.
2.  Create a new feature branch (`git checkout -b feature/my-amazing-feature`).
3.  Commit your changes (`git commit -m 'Add amazing feature'`).
4.  Push to the branch (`git push origin feature/my-amazing-feature`).
5.  Open a **Pull Request**.

-----

## Citation

If you use PILATES in your research, please cite the software release:

```bibtex
@misc{pilates_2024,
  author       = {Needell, Zachary and Waddell, Paul and Caicedo, Juan and Laarabi, Haitam and Wang, Yuhan and Poliziani, Cristian and Lazarus, Jessica and Openkov, Dmitrii and Gardner, Max and Rezaei, Nazanin and others},
  title        = {Platform for Integrated Land use And Transportation Experiments and Simulation (PILATES) v1.0},
  doi          = {10.11578/dc.20240613.2},
  url          = {https://www.osti.gov/biblio/2373117},
  place        = {United States},
  year         = {2024},
  month        = {05}
}
```

-----

## License

This project is licensed under an MIT. See the [LICENSE](LICENSE) file for details.
