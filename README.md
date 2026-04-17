# PILATES

PILATES is a workflow runtime for coupled regional simulation. It coordinates
UrbanSim, ATLAS, ActivitySim, and BEAM across a shared scenario lifecycle with
explicit handoffs, Consist-backed replay and provenance, and post-run analysis
over archived artifacts. The launcher initializes runtime flags once, builds a
single enabled workflow surface for the active run shape, and then drives
planning, binding, restart, and stage execution from that shared projection.

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
