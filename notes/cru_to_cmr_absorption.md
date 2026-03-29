# `cru_to_cmr` Absorption Notes

## Goal

Identify what would need to change in `cru_to_cmr` itself for its fitting
workflow to be absorbed into the current central `jaxcmr` fitting template,
with as little new central-template work as possible.

This note is about project-side changes first. It does not assume that
`jaxcmr/templates/fitting.ipynb` should be expanded to preserve every current
`cru_to_cmr` behavior.

## Short answer

The free-recall side of `cru_to_cmr` is already close to the `jaxcmr` fitting
template. The serial/confusable side is also closer than it first appears, but
it currently mixes generic fitting concerns with project-specific figure
production and stale import paths.

The main changes needed in `cru_to_cmr` are:

- stop treating `factory_type`, `bounds`, and `analysis_paths` as the primary
  notebook interface
- generate central-template-ready parameter bundles instead
- remove project-specific output/export assumptions from the fitting template
- move serial list-length figure fanout out of the fitting template
- clean up the serial/confusable import paths so they refer to project-local
  modules explicitly

## What is already compatible enough

These parts do not appear to require major `jaxcmr` changes:

- `cru_to_cmr` model factories already expose `make_factory(...)` functions with
  the same broad pattern expected by the central template.
- The serial/confusable path already has a project-local likelihood generator
  and simulation utilities in:
  - `cru_to_cmr/confusable_likelihood.py`
  - `cru_to_cmr/confusable_simulation.py`
- The central `jaxcmr` fitting template already supports:
  - configurable `make_factory_path`
  - configurable `loss_fn_path`
  - configurable `sim_alg_path`
  - configurable free/fixed parameter dictionaries
  - configurable comparison and single-analysis configs

That means most of the absorption work is about reshaping `cru_to_cmr` inputs
and expectations, not redesigning the central fitting template from scratch.

## Changes needed in `cru_to_cmr`

### 1. Replace the current notebook parameter schema

The current `cru_to_cmr` fitting templates are organized around:

- `factory_type`
- `bounds`
- `base_params`
- `analysis_paths`

To fit the central template cleanly, `cru_to_cmr` should instead generate
template inputs in the `jaxcmr` schema:

- `make_factory_path`
- `loss_fn_path`
- `sim_alg_path`
- `parameters = {"fixed": ..., "free": ...}`
- `comparison_analysis_configs`
- standard run/data metadata such as `data_tag`, `data_path`, `trial_query`,
  `base_run_tag`, `experiment_count`, and `seed`

Concrete project-side change:

- keep the factorial config generation in `cru_to_cmr/config.py`
- change it so each model variant can emit a full central-template-ready job
  spec instead of only `factory_type` plus `bounds`

### 2. Make base vs compterm explicit in config, not implicit in template logic

Right now the free and serial templates switch factories internally using
`factory_type == "base" | "compterm"`.

For absorption, `cru_to_cmr` should stop relying on template-internal branching
and instead emit the exact factory import path in the render/config layer.

Concrete project-side change:

- free recall:
  - base: `cru_to_cmr.models.cmr_compare.make_factory`
  - compterm: `cru_to_cmr.models.cmr_compterm.make_factory`
- serial recall:
  - base: `cru_to_cmr.models.omnibus_cru_cmr.make_factory`
  - compterm: `cru_to_cmr.models.compterm_omnibus_cru_cmr.make_factory`

This removes one project-specific branch from the fitting template interface.

### 3. Stop fabricating dummy `connections` / feature matrices in the fitting templates

The current free and serial fitting notebooks construct zero-valued
`connections` matrices and pass them through fitting and simulation.

That does not look like a real model requirement for absorption:

- the free-recall factories already accept optional `features` and do not use a
  semantic matrix in the same way the central template does
- the serial omnibus/confusable factories internalize their letter-similarity
  distances rather than needing them from the notebook

Concrete project-side change:

- remove notebook-level construction of zero `connections` matrices
- treat these models as not requiring external feature matrices
- rely on the central template’s `embedding_path` / `emotion_feature_path`
  mechanism only when a model genuinely needs external features

This makes `cru_to_cmr` align better with the central template’s assumption
that optional features are either real inputs or absent.

### 4. Correct the serial/confusable import contract

The serial fitting template currently imports:

- `jaxcmr.experimental.confusable_likelihood`
- `jaxcmr.experimental.confusable_simulation`

But the confusable modules that currently exist in the workspace are project
local under `cru_to_cmr`.

Concrete project-side change:

- treat the project-local modules as the authoritative import paths:
  - `cru_to_cmr.confusable_likelihood.MemorySearchLikelihoodFnGenerator`
  - `cru_to_cmr.confusable_simulation.simulate_study_and_free_recall`
- generate those paths directly into `loss_fn_path` and `sim_alg_path`

This is both a cleanup step and a prerequisite for using the central template’s
string-based import pattern.

### 5. Move serial list-length figure fanout out of the fitting template

The largest remaining mismatch is not fitting itself. It is the serial
template’s downstream figure logic:

- it loops over `list_lengths = [5, 6, 7]`
- it regenerates the same benchmark analyses once per list length
- it writes separate figure variants for each list length

That behavior is project-specific analysis orchestration, not generic fitting.

Concrete project-side change:

- stop requiring the fitting template to emit per-list-length figure sets
- move that fanout into project-local render/orchestration code or separate
  analysis notebooks that run after the generic fit/simulate step

This is the main change needed if `cru_to_cmr` is to fit into the current
central fitting template without making the central template much more complex.

### 6. Align output conventions with the central workflow

The current `cru_to_cmr` fitting templates assume:

- `fits/`
- `figures/png/`
- `figures/tif/`
- `simulations/`
- color and black-and-white exports are both produced by the fitting notebook

The central template assumes a simpler structure:

- `fits/`
- `figures/fitting/`
- `simulations/`
- one figure naming/export workflow driven by `figure_dir` and `figure_str`

Concrete project-side change:

- stop treating `png`/`tif` split output directories as part of the fitting
  template contract
- adopt the central `target_directory` + product-subdirectory convention
- if TIFF and black-and-white exports are still required, move that concern to
  project-local postprocessing rather than keeping it inside the fitting
  template

### 7. Express benchmark analyses as analysis configs, not plain import lists

`cru_to_cmr` currently passes a simple `analysis_paths` list. The central
template expects richer config objects that can carry labels, suffixes, kwargs,
and plot metadata.

Concrete project-side change:

- replace `analysis_paths` with `comparison_analysis_configs`
- for simple cases, each config can still be minimal:
  - `{"target": "jaxcmr.analyses.spc.plot_spc"}`
  - `{"target": "jaxcmr.analyses.crp.plot_crp"}`
  - `{"target": "jaxcmr.analyses.pnr.plot_pnr"}`
- keep project-specific labeling and special plot treatment in the render layer
  only where needed

This keeps the actual fitting notebook closer to the central interface.

## What this means for the two current fitting templates

### Free recall

The free-recall template can likely be absorbed after:

- replacing `factory_type` with explicit `make_factory_path`
- replacing `bounds` + `base_params` with `parameters["free"]` and
  `parameters["fixed"]`
- replacing `analysis_paths` with `comparison_analysis_configs`
- dropping dummy `connections`
- aligning output directories

This is mostly a config/render refactor.

### Serial recall

The serial-recall template likely can be absorbed only after:

- switching to project-local `confusable_*` import paths in generated config
- emitting explicit `make_factory_path`, `loss_fn_path`, and `sim_alg_path`
- dropping dummy `connections`
- removing list-length fanout from the fitting template itself
- aligning output directories and export expectations

The serial path is therefore absorbable, but only if `cru_to_cmr` stops asking
the fitting template to also behave like a project-specific figure orchestrator.

## Practical migration order inside `cru_to_cmr`

1. Normalize config generation so free and serial variants emit central-template
   inputs directly.
2. Update the serial path to use project-local confusable import paths.
3. Remove dummy `connections` handling from the project notebooks/config.
4. Move list-length-specific figure fanout out of the serial fitting template.
5. Align output naming and directory expectations with the central workflow.
6. Once free and serial fits can run through the same central template
   contract, retire the project-local fitting templates.

## Bottom line

If the question is "what needs to change about `cru_to_cmr` so it can be
absorbed?", the answer is mostly:

- normalize `cru_to_cmr` to the central template interface
- stop embedding project-specific orchestration inside the fitting templates
- make the serial/confusable path explicit and project-local through import
  strings rather than through special notebook code

The free-recall path is already close. The serial path is the real forcing
case, but it still looks solvable primarily through `cru_to_cmr` cleanup rather
than a large redesign of the central fitting template.
