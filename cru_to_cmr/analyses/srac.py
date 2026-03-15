"""Serial Recall Accuracy Curve (SRAC).

Computes the probability that the item recalled at each output
position matches the item studied at that position (strict serial
recall accuracy).

"""

from typing import Optional, Sequence

import jax.numpy as jnp
from jax import jit, vmap
from matplotlib.axes import Axes

from ..helpers import apply_by_subject, find_max_list_length
from ..plotting import plot_data, prepare_plot_inputs, set_plot_labels
from ..repetition import all_study_positions
from ..typing import Array, Bool, Float, Integer, RecallDataset

__all__ = ["trial_srac", "srac", "plot_srac"]


def trial_srac(
    recalls: Integer[Array, " recall_positions"],
    presentations: Integer[Array, " study_positions"],
    size: int = 3,
) -> Bool[Array, " study_positions"]:
    """Flag each study position as correctly recalled or not.

    Parameters
    ----------
    recalls : Integer[Array, " recall_positions"]
        1-indexed recalls for one trial; 0 for no recall.
    presentations : Integer[Array, " study_positions"]
        Item IDs in study order.
    size : int
        Max study positions an item can occupy.

    Returns
    -------
    Bool[Array, " study_positions"]
        True where the study position was correctly recalled.

    """
    list_length = presentations.shape[0]
    recalls = recalls[:list_length]
    expanded_recalls = vmap(all_study_positions, in_axes=(0, None, None))(
        recalls, presentations, size
    )
    study_positions = jnp.arange(1, list_length + 1)
    return vmap(lambda r, i: jnp.any(r == i))(expanded_recalls, study_positions)


def srac(
    dataset: RecallDataset,
    size: int = 3,
) -> Float[Array, " study_positions"]:
    """Proportion of correct recalls at each study position.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with ``recalls`` and ``pres_itemnos``.
    size : int
        Max study positions an item can occupy.

    Returns
    -------
    Float[Array, " study_positions"]
        Mean accuracy at each study position.

    """
    recalls = dataset["recalls"]
    presentations = dataset["pres_itemnos"]

    return vmap(trial_srac, in_axes=(0, 0, None))(
        recalls,
        presentations,
        size,
    ).mean(axis=0)


def plot_srac(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    size: int = 3,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot serial recall accuracy with confidence intervals.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials.
    color_cycle : list[str] or None
        Colors for each curve.
    labels : Sequence[str] or None
        Legend labels for each curve.
    contrast_name : str or None
        Legend title.
    axis : Axes or None
        Existing Axes to plot on.
    size : int
        Max study positions an item can occupy.
    confidence_level : float
        Confidence level for error bounds.

    Returns
    -------
    Axes
        Matplotlib Axes with the SRAC plot.

    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )

    if labels is None:
        labels = ["" for _ in datasets]

    # Identify the largest list length across datasets so we can plot consistently
    max_list_length = find_max_list_length(datasets, trial_masks)

    for data_index, data_dict in enumerate(datasets):
        subject_values = jnp.vstack(
            apply_by_subject(
                data_dict,
                trial_masks[data_index],
                jit(srac, static_argnames=("size",)),
                size=size,
            )
        )

        color = color_cycle[data_index % len(color_cycle)]
        xvals = jnp.arange(max_list_length) + 1
        subject_values = subject_values[:, :max_list_length]
        xvals = jnp.arange(max_list_length) + 1
        plot_data(
            axis,
            xvals,
            subject_values,
            labels[data_index],
            color,
            confidence_level=confidence_level,
        )

    set_plot_labels(axis, "Study Position", "Serial Recall Accuracy", contrast_name)
    return axis
