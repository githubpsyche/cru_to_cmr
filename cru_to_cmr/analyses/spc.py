"""Serial position curve analyses.

Computes recall probability at each study position, producing the
classic U-shaped serial position curve with primacy and recency
effects. Supports both unique-item and repeated-item lists.

"""

from __future__ import annotations

from typing import Optional, Sequence

import jax.numpy as jnp
from jax import jit, vmap
from matplotlib.axes import Axes

from ..helpers import apply_by_subject, find_max_list_length
from ..plotting import plot_data, set_plot_labels, prepare_plot_inputs
from ..repetition import all_study_positions
from ..typing import Array, Bool, Float, Integer, RecallDataset

__all__ = ["spc", "plot_spc"]


def spc(
    dataset: RecallDataset,
    size: int = 3,
) -> Float[Array, " study_positions"]:
    """Compute recall rate by study position, handling repeated items.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset containing ``recalls`` and ``pres_itemnos``.
    size : int, optional
        Maximum study positions an item can occupy (default: 3).

    Returns
    -------
    Float[Array, " study_positions"]
        Recall probability at each serial position.

    """
    recalls = dataset["recalls"]
    presentations = dataset["pres_itemnos"]
    list_length = presentations.shape[1]

    expanded_recalls = vmap(
        vmap(all_study_positions, in_axes=(0, None, None)), in_axes=(0, 0, None)
    )(recalls, presentations, size)

    counts = jnp.bincount(expanded_recalls.flatten(), length=list_length + 1)[1:]
    return counts / len(expanded_recalls)


def plot_spc(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    size: int = 3,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot serial position curves with confidence intervals.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets. Each is paired with the mask at the same index.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials. Length must match ``datasets``.
    color_cycle : list[str], optional
        Colors for each curve.
    labels : Sequence[str], optional
        Legend labels for each curve.
    contrast_name : str, optional
        Legend title.
    axis : Axes, optional
        Existing Axes to plot on.
    size : int, optional
        Maximum study positions an item can occupy (default: 3).
    confidence_level : float, optional
        Confidence level for error bounds (default: 0.95).

    Returns
    -------
    Axes
        Matplotlib Axes with serial position curves.

    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )

    if labels is None:
        labels = [""] * len(datasets)

    max_list_length = find_max_list_length(datasets, trial_masks)
    for data_index, data in enumerate(datasets):
        subject_values = jnp.vstack(
            apply_by_subject(
                data,
                trial_masks[data_index],
                jit(spc, static_argnames=("size",)),
                size=size,
            )
        )

        color = color_cycle[data_index % len(color_cycle)]
        plot_data(
            axis,
            jnp.arange(max_list_length, dtype=int) + 1,
            subject_values,
            labels[data_index],
            color,
            confidence_level=confidence_level,
        )

    set_plot_labels(axis, "Study Position", "Recall Rate", contrast_name)
    return axis
