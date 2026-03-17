# SLURM Workflow for Parameterized Notebook Execution

Guide for submitting analysis notebooks as SLURM jobs on a university HPC cluster.
Works for any project structured like cru_to_cmr: render_* orchestrators that
generate parameterized template notebooks via papermill, each independently executable.

## Overview

```
Generate          Transfer          Submit            Collect
─────────         ──────────        ──────────        ──────────
render_*     →    get project  →    sbatch job   →    get results
notebooks         on cluster        array             back locally
(prepare_only)
```

Each render_* notebook produces parameterized notebooks in `analyses/rendered/`.
Each rendered notebook is a self-contained analysis (one model fit, one parameter
sweep, etc.) that can run independently. SLURM job arrays execute them in parallel.

## Getting the Project on the Cluster

### Option A: git clone (recommended)

Data files, rendered notebooks, and fit results are all tracked in git.
A clone gives you everything.

```bash
git clone https://github.com/githubpsyche/<project>.git
cd <project>
```

To update later: `git pull`

### Option B: rsync

If git is unavailable on the cluster, or you need to sync untracked changes:

```bash
# From your local machine:
rsync -avz --exclude='.venv' <project>/ <user>@<cluster>:<project>/
```

### Install dependencies

```bash
# Install uv if not available (no root needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install all dependencies (reads pyproject.toml + uv.lock)
uv sync

# Install jaxcmr from GitHub (pin to a tag or commit)
uv add git+https://github.com/githubpsyche/jaxcmr.git@main
# or for a specific version:
# uv add git+https://github.com/githubpsyche/jaxcmr.git@v0.1.0
```

### Verify the environment

```bash
uv run python -c "import jaxcmr; import jax; print(jax.devices())"
```

## Generating Parameterized Notebooks

Run the render_* orchestrator notebooks with `prepare_only = True` (the default).
This injects parameters into template notebooks without executing them.

You can do this locally or on the cluster — it's fast (seconds, no fitting):

```bash
uv run jupyter execute analyses/render_free_recall.ipynb
uv run jupyter execute analyses/render_serial_recall.ipynb
uv run jupyter execute analyses/render_parameter_shifting.ipynb
```

Output appears in `analyses/rendered/`. Each notebook is ready to execute
independently.

## Submitting Jobs

### All notebooks (job array)

```bash
./scripts/submit_all.sh
```

This creates a manifest file and submits a single SLURM job array. One array
task per notebook, all managed as a single job.

### Filtered submission

```bash
./scripts/submit_all.sh "fitting_*"     # only fitting notebooks
./scripts/submit_all.sh "shifting_*"    # only shifting notebooks
```

### Single notebook

```bash
sbatch scripts/run_notebook.sbatch analyses/rendered/shifting_learning_rate.ipynb
```

## Monitoring Jobs

```bash
# All your jobs
squeue -u $USER

# Detailed view of a job array
squeue -j <jobid>

# After completion: runtime, memory, exit status
sacct --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode -j <jobid>

# Read a specific task's output (task 5 of array job 12345)
cat logs/nb_12345_5.out
cat logs/nb_12345_5.err

# Cancel all tasks in a job array
scancel <jobid>

# Cancel a single task
scancel <jobid>_<taskid>
```

## Collecting Results

Fit results and figures are written to `fits/` and `figures/` within the
project directory on the cluster.

### Option A: git (recommended)

```bash
# On the cluster:
git add fits/ figures/
git commit -m "Add fit results from cluster run"
git push

# Locally:
git pull
```

### Option B: rsync

```bash
# From your local machine:
rsync -avz <user>@<cluster>:<project>/fits/ fits/
rsync -avz <user>@<cluster>:<project>/figures/ figures/
```

## Troubleshooting

### Job failed with OOM (out of memory)

Increase `--mem` in `run_notebook.sbatch` or request fewer concurrent tasks:

```bash
sbatch --array=0-46%4 scripts/run_notebook.sbatch  # max 4 concurrent
```

### Job timed out

Increase `--time` in `run_notebook.sbatch`. Check how long a single fit takes
by running one interactively:

```bash
srun --time=08:00:00 --mem=16G --pty bash
uv run jupyter execute analyses/rendered/<notebook>.ipynb
```

### ModuleNotFoundError

The venv is probably not set up correctly, or jaxcmr isn't installed. SSH to
the cluster and verify:

```bash
uv run python -c "import jaxcmr; import cru_to_cmr"
```

### find_project_root fails

The sbatch script uses `cd "$SLURM_SUBMIT_DIR"` to set the working directory.
Make sure you submit from the project root:

```bash
cd /path/to/<project>
./scripts/submit_all.sh
```

### Re-running failed tasks

Check which tasks failed, then resubmit just those:

```bash
# Find failed task IDs
sacct -j <jobid> --format=JobID,State | grep FAILED

# Resubmit specific tasks (e.g., tasks 3, 7, 12)
sbatch --array=3,7,12 scripts/run_notebook.sbatch
```

## JAX Configuration

The sbatch script sets JAX defaults. Edit `scripts/run_notebook.sbatch` to
change these:

```bash
# CPU (default)
export JAX_PLATFORM_NAME=cpu

# GPU (uncomment in sbatch script; also uncomment module load cuda)
export JAX_PLATFORM_NAME=gpu

# Limit GPU memory preallocation (useful on shared GPU nodes)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

# Disable JAX compilation cache warnings
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
```

## Customizing Resources

Edit the `#SBATCH` directives in `scripts/run_notebook.sbatch`:

| Directive | Default | Description |
|---|---|---|
| `--time` | 04:00:00 | Wall time limit |
| `--mem` | 16G | Memory per task |
| `--cpus-per-task` | 4 | CPU cores |
| `--gres=gpu:1` | (not set) | Request GPU |
| `--partition` | (not set) | Cluster partition |
| `--array=0-N%K` | (no limit) | Max K concurrent tasks |

For heterogeneous jobs (some need GPU, some don't), copy `run_notebook.sbatch`
to a variant and adjust.

## Executing Locally Without SLURM

Set `prepare_only = False` in the render_* notebook's config cell and re-run
to execute all configs sequentially. Or run individual rendered notebooks:

```bash
MPLBACKEND=Agg jupyter execute analyses/rendered/shifting_learning_rate.ipynb
```
