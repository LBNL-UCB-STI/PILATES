# PILATES

PILATES is a workflow runtime for coupled regional simulation. It coordinates UrbanSim, ATLAS,
ActivitySim, and BEAM across a shared scenario lifecycle, preserves explicit
handoffs at the workflow boundaries, and uses [Consist](https://github.com/LBNL-UCB-STI/consist) 
to keep data lineage and archived outputs queryable for run comparison and replay-aware analysis.

Minimal path:

```bash
python run.py -c <settings.yaml>
python examples/consist/restart_replay_inspection.py <archive-run-dir>
```

For a deeper Consist-facing walkthrough, start with
[docs/workflow/consist_in_pilates.md](docs/workflow/consist_in_pilates.md) and
[docs/analysis/consist_in_action.md](docs/analysis/consist_in_action.md).

The full documentation now lives in the static docs site:

- Public docs site: [LBNL-UCB-STI.github.io/PILATES](https://LBNL-UCB-STI.github.io/PILATES/)
- Local docs entrypoint in this repo: [docs/index.md](docs/index.md)

## Quick Start

```bash
git clone https://github.com/LBNL-UCB-STI/PILATES.git
cd PILATES
conda env create -f environment.yml
conda activate pilates
python run.py -c <settings.yaml>
```

Use the site for the real paths through the project:

- First local run: [docs/start-here/getting_started.md](docs/start-here/getting_started.md)
- Runtime CLI and config: [docs/run/cli.md](docs/run/cli.md) and [docs/run/configuration_reference.md](docs/run/configuration_reference.md)
- Workflow semantics: [docs/workflow/workflow_primer.md](docs/workflow/workflow_primer.md)
- Model requirements and handoffs: [docs/reference/model_boundaries.md](docs/reference/model_boundaries.md)
- Adding a model: [docs/extend/adding_a_model.md](docs/extend/adding_a_model.md)
- Archived-run analysis: [docs/analysis/overview.md](docs/analysis/overview.md)
- Consist showcase scripts: [docs/analysis/consist_in_action.md](docs/analysis/consist_in_action.md)
- Lawrencium: [docs/run/lawrencium.md](docs/run/lawrencium.md)

## Citation

If you use PILATES in research, cite:

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

## License

MIT. See [LICENSE](LICENSE).
