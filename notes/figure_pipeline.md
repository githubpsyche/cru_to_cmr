# Figure Pipeline

## Stage 1: Generate figures

- `factorial_comparison.py` generates free recall figures (tif) into `figures/`
- `serial_factorial_comparison.py` generates serial recall figures (tif) into `figures/`
- Both scripts also save PNG versions used by the manuscript

## Stage 2: Map figures to manuscript numbering

- `figure_paths.md` lists tif paths organized under `# Figure N` headers
- The order of images under each header determines the subplot letter (a, b, c, ...)

## Stage 3: Rename for submission

- `rename_figures.py` reads `figure_paths.md` and copies tifs into `submission/` as `Figure1a.tif`, `Figure1b.tif`, etc.
- Pure copy operation, re-runnable if source tifs exist

## Current state

- Source tifs in `figures/` were cleaned up — only PNGs remain
- Originals live in `jaxcmr/projects/cru_to_cmr/indexed_figures/` — do NOT copy yet
- To validate code: re-run analysis scripts (Stage 1), then `rename_figures.py` (Stage 3), then diff output against `indexed_figures/`