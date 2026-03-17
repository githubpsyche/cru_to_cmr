# Deploy cru_to_cmr fits to CSD3

## Context
cru_to_cmr has 83 parameterized notebooks in `analyses/rendered/` ready to execute. These need to run on Cambridge's CSD3 cluster (CPU only). Both `cru_to_cmr` and `jaxcmr` must be rsync'd since cru_to_cmr depends on jaxcmr via editable install (`../jaxcmr` in pyproject.toml).

## Steps

### 1. Update sbatch script for CSD3
- Add `#SBATCH --partition=cclake` (CSD3 CPU partition)
- Add `module load python/3.12` (or whichever is available)
- Existing script already handles job arrays, manifest, and environment

### 2. Update slurm_guide.md for rsync workflow
Replace generic `<user>@<cluster>` placeholders with CSD3 specifics:
- Host: `login.hpc.cam.ac.uk`
- User: `jg2204`
- Both repos rsync'd to `~/workspace/` on cluster to preserve sibling structure

### 3. Local prep (already done)
83 notebooks exist in `analyses/rendered/`. No local action needed.

### 4. rsync to CSD3
```bash
rsync -avz --exclude='.venv' --exclude='.quarto' --exclude='docs/' \
  ~/workspace/jaxcmr/ jg2204@login.hpc.cam.ac.uk:~/workspace/jaxcmr/

rsync -avz --exclude='.venv' --exclude='.quarto' --exclude='docs/' \
  ~/workspace/cru_to_cmr/ jg2204@login.hpc.cam.ac.uk:~/workspace/cru_to_cmr/
```

jaxcmr first — cru_to_cmr's `uv sync` needs it present at `../jaxcmr`.

### 5. Cluster setup (SSH to CSD3)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv not available
cd ~/workspace/cru_to_cmr
uv sync
uv run python -c "import jaxcmr; import cru_to_cmr; import jax; print(jax.devices())"
mkdir -p logs
```

### 6. Submit
```bash
cd ~/workspace/cru_to_cmr
./scripts/submit_all.sh              # all 83
# or:
./scripts/submit_all.sh "fitting_*"  # 64 fitting jobs
./scripts/submit_all.sh "shifting_*" # 19 shifting jobs
```

### 7. Collect results
```bash
# From local machine:
rsync -avz jg2204@login.hpc.cam.ac.uk:~/workspace/cru_to_cmr/fits/ \
  ~/workspace/cru_to_cmr/fits/
rsync -avz jg2204@login.hpc.cam.ac.uk:~/workspace/cru_to_cmr/figures/ \
  ~/workspace/cru_to_cmr/figures/
```

## Files to modify
- `scripts/run_notebook.sbatch` — add CSD3 partition, module load
- `notes/slurm_guide.md` — replace placeholders with CSD3 specifics

## Verification
- Test single notebook first: `sbatch scripts/run_notebook.sbatch analyses/rendered/shifting_learning_rate.ipynb`
- Check output: `cat logs/nb_*.out`
- If it succeeds, submit the full array
