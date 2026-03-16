# Gunn, J. B., & Polyn, S. M. (2025). Bridging CRU and CMR in free and serial recall: A factorial comparison of retrieved-context models. The American Journal of Psychology, 138(2), 203-230.

Retrieved-context theory posits that episodic retrieval is driven by a representation that evolves over time, tying each item to the contextual features present during encoding and later accessing those associations to guide retrieval. Although the Context Maintenance and Retrieval (CMR) model offers a flexible and well-tested retrieved-context implementation — addressing both free and serial recall — it is relatively complex, incorporating mechanisms absent in some other retrieved-context models. In contrast, the Context Retrieval and Updating (CRU) model provides a simpler, more streamlined specification of context-driven retrieval shown to excel in strictly ordered memory tasks like serial recall. However, it remains unclear whether CRU's leaner architecture extends easily to unconstrained retrieval dynamics in free recall. It is similarly unknown whether CMR's added mechanisms confer meaningful advantages over leaner CRU in serial recall. To investigate the gap between CRU and CMR, we systematically compare them, presenting them side by side and exploring how each can be viewed as a parameterized variant of the same foundational ideas. Using a factorial model selection approach, we selectively incorporate CMR-like features into CRU and compare each hybrid variant to standard CMR on free and serial recall data. We find that selectively incorporating CMR-like features substantially improves CRU's fit to free recall and that CRU's item-confusion and recall termination mechanisms can help CMR capture serial recall data.

[Read the manuscript](https://githubpsyche.github.io/cru_to_cmr/)

## Reproducing the Manuscript

The codebase comes with a `cru_to_cmr` package hosting a flexible implementations of CRU and CMR along with code for fitting and simulating the models and analyzing the output. 

Requires Python 3.12+.

```bash
pip install -e .              # or: uv sync
pip install -e ".[dev]"       # include dev tools (papermill, vulture, ruff)
```

### 1. Generate figures

Run the orchestrator notebooks in `analyses/`:

- `render_free_recall.ipynb` — free recall model fits (HealeyKahana2014)
- `render_serial_recall.ipynb` — serial recall model fits (Gordon2021)
- `render_parameter_shifting.ipynb` — parameter sensitivity analyses

Each orchestrator dispatches a template notebook via [papermill](https://papermill.readthedocs.io/) for every model configuration. Pre-computed fits in `fits/` are loaded by default (`redo_fits = False`).

Figures are saved to `figures/png/` and `figures/tif/`.

### 2. Generate tables

Run the model comparison notebooks:

- `analyses/HealeyKahana2014_cru_to_cmr_Model_Comparison.ipynb`
- `analyses/Logan2021_cru_to_cmr_Model_Comparison.ipynb`

These produce markdown tables (BIC, AIC, AICw, winner ratios, parameter summaries) in `tables/`.

### 3. Prepare submission figures

```bash
cd analyses && python rename_figures.py
```

Reads `figure_paths.md` and copies TIFs into `submission/` with journal-standard names (`Figure1a.tif`, `Figure1b.tif`, ...).

### 4. Render manuscript

```bash
quarto render --to html,apaquarto-docx,apaquarto-pdf
```

Output appears in `docs/` (HTML with links to PDF and DOCX downloads).

## Project Structure

```
cru_to_cmr/
├── index.qmd                 # manuscript source
├── _quarto.yml               # quarto project config
├── references.bib            # bibliography
├── _extensions/              # quarto extensions (apaquarto, abstract-section)
├── cru_to_cmr/               # python package (vendored from jaxcmr)
│   ├── models/               # CRU/CMR model variants
│   ├── analyses/             # behavioral analysis functions (SPC, CRP, PNR, ...)
│   ├── components/           # model components (context, memory, termination)
│   ├── experimental/         # confusable likelihood/simulation
│   └── ...                   # fitting, simulation, helpers, config
├── analyses/                 # research notebooks
│   ├── templates/            # papermill templates (one model config per run)
│   ├── render_*.ipynb        # orchestrators (dispatch templates)
│   ├── *_Model_Comparison.ipynb
│   ├── check_unused.py       # dead code checker
│   ├── rename_figures.py     # submission figure renamer
│   └── figure_paths.md       # figure-to-submission mapping
├── figures/
│   ├── fig*.md               # figure layout includes (referenced by index.qmd)
│   ├── png/                  # generated PNGs (used by HTML render)
│   ├── tif/                  # generated TIFs (used by PDF render + submission)
│   └── shifting/             # parameter shifting figures
├── fits/                     # pre-computed model fits (JSON)
├── data/                     # datasets (HDF5) + embeddings (NPY)
├── docs/                     # rendered manuscript output
└── notes/                    # project notes
```
