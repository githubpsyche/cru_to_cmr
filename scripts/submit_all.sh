#!/bin/bash
# Submit rendered notebooks as a SLURM job array.
#
# Usage:
#   ./scripts/submit_all.sh                # submit all rendered notebooks
#   ./scripts/submit_all.sh "fitting_*"    # submit only fitting notebooks
#   ./scripts/submit_all.sh "shifting_*"   # submit only shifting notebooks
#
# This creates a manifest file listing the matching notebooks, then submits
# a single job array where each task executes one notebook.

set -euo pipefail

PATTERN="${1:-*}"
RENDERED_DIR="analyses/rendered"
MANIFEST="$RENDERED_DIR/manifest.txt"

# Build manifest
: > "$MANIFEST"
for nb in $RENDERED_DIR/$PATTERN.ipynb; do
    [ -f "$nb" ] || continue
    echo "$nb" >> "$MANIFEST"
done

COUNT=$(wc -l < "$MANIFEST" | tr -d ' ')

if [ "$COUNT" -eq 0 ]; then
    echo "No notebooks matched pattern: $PATTERN"
    rm -f "$MANIFEST"
    exit 1
fi

mkdir -p logs

echo "Submitting $COUNT notebooks as a job array:"
head -5 "$MANIFEST"
[ "$COUNT" -gt 5 ] && echo "  ... and $((COUNT - 5)) more"

sbatch --array=0-$((COUNT - 1)) scripts/run_notebook.sbatch
