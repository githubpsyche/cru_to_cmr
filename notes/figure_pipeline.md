# Figure Pipeline

## Stage 1: Generate figures

- `analyses/render_free_recall.ipynb` dispatches `templates/free_recall_fitting.ipynb` via papermill → PNGs to `figures/png/`, TIFs to `figures/tif/`
- `analyses/render_serial_recall.ipynb` dispatches `templates/serial_recall_fitting.ipynb` → PNGs to `figures/png/`, TIFs to `figures/tif/`
- `analyses/render_parameter_shifting.ipynb` dispatches `templates/parameter_shifting.ipynb` → PNGs to `figures/shifting/`
- All templates load pre-computed fits from `fits/` and data from `data/`
- Set `redo_fits = False` in orchestrator shared_params to skip fitting and only regenerate figures

## Stage 2: Map figures to manuscript numbering

- `analyses/figure_paths.md` lists tif paths organized under `# Figure N` headers
- The order of images under each header determines the subplot letter (a, b, c, ...)

## Stage 3: Rename for submission

- `analyses/rename_figures.py` reads `figure_paths.md` and copies tifs into `submission/` as `Figure1a.tif`, `Figure1b.tif`, etc.
- Pure copy operation, re-runnable if source tifs exist

## Stage 4: Render manuscript

```bash
quarto render --to html,apaquarto-docx,apaquarto-pdf
```

Output to `docs/`. HTML includes "Other Formats" links to PDF and DOCX.
