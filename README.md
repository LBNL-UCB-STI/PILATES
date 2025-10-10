<p align="center"><img src="logo_multi.png" width="700" alt="PILATES Logo"></p>

**PILATES** (**P**latform for **I**ntegrated **L**anduse **A**nd **T**ransportation **E**xperiments and **S**imulation) is a framework for orchestrating containerized microsimulation models to study the co-evolution of land use and transportation systems. Designed for long-term regional forecasting, PILATES enables researchers and planners to simulate complex urban dynamics over multi-decade periods by linking specialized models that operate at different time scales.

Rather than tightly coupling models within a single software process, PILATES orchestrates them in a containerized environment. This modular architecture allows it to leverage the behavioral sophistication of existing specialized models with minimal modification, while providing robust state management and reproducibility.

-----

## Integrated Simulation Models

PILATES integrates several leading microsimulation models to create a comprehensive forecasting platform:

  * **[UrbanSim](https://github.com/UDST/urbansim)**: Models long-term metropolitan evolution by simulating household and business location choices, real estate development, and land use changes over periods of years or decades.

  * **[ATLAS](https://doi.org/10.1080/03081060.2024.2353784)**: Simulates household vehicle fleet dynamics, including vehicle purchase, replacement, and technology adoption (e.g., electric vs. internal combustion vehicles).

  * **[ActivitySim](https://github.com/ActivitySim/activitysim)**: Generates daily activity-based travel demand by simulating individual travel decisions including trip purpose, destination, time of day, and mode choice for a synthetic population.

  * **[BEAM](https://github.com/LBNL-UCB-STI/beam)**: The **B**ehavior, **E**nergy, **A**utonomy, and **M**obility framework simulates agent-based transportation on detailed road networks, modeling traffic congestion, transit operations, and emerging mobility services to produce network performance metrics.

-----

## Simulation Scenarios

PILATES supports various simulation configurations to match different research and planning needs:

1.  **BEAM Only**: Detailed network performance analysis using fixed travel plans. Ideal for studying infrastructure impacts, operational strategies, or new mobility services.

2.  **ActivitySim + BEAM**: Agent-based travel demand modeling with network feedback. ActivitySim generates daily travel demand, BEAM simulates it on the network, and the resulting travel times feed back until equilibrium is reached.

3.  **UrbanSim + ActivitySim + BEAM**: Long-term land use and transportation co-evolution. UrbanSim simulates multi-year changes, which inform ActivitySim/BEAM travel demand modeling. Changes in accessibility from the transport model inform the next UrbanSim period.

4.  **UrbanSim + ATLAS + ActivitySim + BEAM**: Comprehensive modeling of land use, transportation, and vehicle technology co-evolution, including how system changes influence household vehicle choices and their effects on energy consumption and emissions.

-----

## Key Features

  * **Modular Architecture**: Containerized models (Docker/Singularity) can be mixed and matched to create custom simulation workflows
  * **Flexible Temporal Coupling**: Configure execution frequency of different models to study various feedback loops and time horizons
  * **State Management**: Automated tracking and persistence of simulation state across model runs and time steps
  * **Reproducibility**: Complete provenance tracking of data transformations and model executions
  * **High-Performance Computing**: Optimized for HPC environments with parallel processing support
  * **Database Integration**: Optional analytical database backend for improved performance and data management

-----

## Getting Started

### Prerequisites

  * Container runtime: **Docker** or **Singularity**
  * **Anaconda** or **Miniconda** for Python environment management

### Installation

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

3.  **Download input data:**

    Various raw input data files are required for the different models. See [lawrencium-setup.md](lawrencium-setup.md#download-data) for instructions.

### Configuration

1.  Edit `settings.yaml` to configure your simulation:
    - Set container preference: `container_manager: "docker"` or `"singularity"`
    - Update region-specific parameters
    - Adjust computational settings (`num_processors`, `chunk_size`, etc.)

### Running a Simulation

Execute the main script from the root directory. Use the `-p` flag to pull the latest container images before the first run:

```bash
python run.py -v -p
```

## Advanced Features

### Database Integration

PILATES includes an optional database backend for improved performance and scalability. The database system supports:

- Direct database input to ActivitySim, eliminating expensive H5 preprocessing
- Dual storage architecture preserving both raw and processed data
- Parallel data uploads for large-scale simulations
- Complete data lineage tracking with OpenLineage metadata
- Automatic schema documentation and data quality validation

For detailed setup and usage instructions, see [docs/database-setup.md](docs/database-setup.md) and [docs/database_documentation_guide.md](docs/database_documentation_guide.md).

### HPC Execution

For high-performance computing environments, specialized scripts are available in the `hpc/` directory. For detailed instructions on setting up PILATES on the Lawrencium cluster, see [lawrencium-setup.md](lawrencium-setup.md).

```bash
cd hpc
./job_runner.sh [options]
```

### Background Execution

To run PILATES as a background process:

```bash
nohup python run.py -v &
```

Output will be saved to `nohup.out`.

-----

## Architecture

PILATES uses a consistent preprocessor/runner/postprocessor pattern for model execution, providing a structured approach to data preparation, model execution, and output processing. This pattern enables:

- Modular development and testing of individual components
- Reusability across different simulation models
- Clear separation of concerns for maintainability
- Integrated provenance tracking for reproducibility

For implementation details and guidance on integrating new models, see [docs/architecture.md](docs/architecture.md).

-----

## Contributing

We welcome contributions! Please use the [GitHub issue tracker](https://github.com/LBNL-UCB-STI/PILATES/issues) for bug reports, feature requests, and support questions.

### How to Contribute

1.  Fork the repository
2.  Create a feature branch (`git checkout -b feature/my-feature`)
3.  Commit your changes (`git commit -m 'Add new feature'`)
4.  Push to the branch (`git push origin feature/my-feature`)
5.  Open a Pull Request

Please see our (forthcoming) `CONTRIBUTING.md` for detailed guidelines.

-----

## Citation

If you use PILATES in your research, please cite:

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

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
