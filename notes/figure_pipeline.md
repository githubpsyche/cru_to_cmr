# Figure Pipeline

## Stage 1: Generate figures

- `analyses/render_free_recall.ipynb` dispatches `templates/free_recall_fitting.ipynb` via papermill for each model config → generates free recall figures (tif + png) into `figures/`
- `analyses/render_serial_recall.ipynb` dispatches `templates/serial_recall_fitting.ipynb` → generates serial recall figures into `figures/`
- `analyses/render_parameter_shifting.ipynb` dispatches `templates/parameter_shifting.ipynb` → generates parameter shifting figures into `figures/shifting/`
- All templates load pre-computed fits from `fits/` and data from `data/`
- Set `redo_fits = False` in orchestrator shared_params to skip fitting and only regenerate figures

## Stage 2: Map figures to manuscript numbering

- `analyses/figure_paths.md` lists tif paths organized under `# Figure N` headers
- The order of images under each header determines the subplot letter (a, b, c, ...)

## Stage 3: Rename for submission

- `analyses/rename_figures.py` reads `figure_paths.md` and copies tifs into `submission/` as `Figure1a.tif`, `Figure1b.tif`, etc.
- Pure copy operation, re-runnable if source tifs exist

## Validation

- To validate: re-run Stage 1 + Stage 3, then diff `submission/` against originals in `jaxcmr/projects/cru_to_cmr/indexed_figures/`
