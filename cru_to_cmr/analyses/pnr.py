"""Probability of nth recall (PNR).

Computes the probability that each serial position is recalled at
each output position. Supports both unique-item and repeated-item
lists, with optional conditioning on a ``_should_tabulate`` mask.

"""

__all__ = [
    "available_recalls_with_repeats",
    "actual_recalls_with_repeats",
    "conditional_pnr_with_repeats",
    "plot_pnr",
]

from typing import Optional, Sequence

import jax.numpy as jnp
from jax import jit, lax, vmap
from matplotlib.axes import Axes

from ..helpers import apply_by_subject, find_max_list_length
from ..plotting import plot_data, prepare_plot_inputs, set_plot_labels
from ..repetition import all_study_positions
from ..typing import Array, Bool, Float, Integer, RecallDataset


def available_recalls_with_repeats(
    recalls: Integer[Array, " recall_positions"],
    presentations: Integer[Array, " list_length"],
    query_recall_position: int,
    list_length: int,
    size: int,
) -> Bool[Array, " list_length"]:
    """Mask of available positions when items may repeat.

    Parameters
    ----------
    recalls : Integer[Array, " recall_positions"]
        1-indexed recalls for a single trial.
    presentations : Integer[Array, " list_length"]
        Items presented at each study position.
    query_recall_position : int
        Index in the recall sequence to evaluate.
    list_length : int
        Number of items in the study list.
    size : int
        Max study positions an item can occupy.

    Returns
    -------
    Bool[Array, " list_length"]
        True for positions not yet recalled.

    """
    prior = vmap(all_study_positions, in_axes=(0, None, None))(
        recalls[:query_recall_position], presentations, size
    ).reshape(-1)

    init = jnp.ones(list_length + 1, dtype=bool)
    final_mask, _ = lax.scan(lambda m, p: (m.at[p].set(False), None), init, prior)

    return final_mask[1:]


def actual_recalls_with_repeats(
    recalls: Integer[Array, " recall_positions"],
    presentations: Integer[Array, " list_length"],
    query_recall_position: int,
    list_length: int,
    size: int,
) -> Bool[Array, " list_length"]:
    """Mask with study positions of the recalled item as True.

    Parameters
    ----------
    recalls : Integer[Array, " recall_positions"]
        1-indexed recalls for a single trial.
    presentations : Integer[Array, " list_length"]
        Items presented at each study position.
    query_recall_position : int
        Index in the recall sequence to evaluate.
    list_length : int
        Number of items in the study list.
    size : int
        Max study positions an item can occupy.

    Returns
    -------
    Bool[Array, " list_length"]
        True at all positions of the recalled item.

    """
    item = recalls[query_recall_position]
    current = all_study_positions(item, presentations, size)  # shape: (size,)

    init = jnp.zeros(list_length + 1, dtype=bool)
    final_mask, _ = lax.scan(lambda m, p: (m.at[p].set(True), None), init, current)
    return final_mask[1:]


def conditional_pnr_with_repeats(
    dataset: RecallDataset,
    size: int,
    query_recall_position: int,
) -> Float[Array, " list_length"]:
    """Conditional PNR when study items may repeat.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with ``recalls`` and ``pres_itemnos``.
    size : int
        Max study positions an item can occupy.
    query_recall_position : int
        0-based recall index to analyze.

    Returns
    -------
    Float[Array, " list_length"]
        Conditional probability at each study position.

    """
    recalls = dataset["recalls"]
    presentations = dataset["pres_itemnos"]
    list_length = presentations.shape[1]

    actual = vmap(actual_recalls_with_repeats, in_axes=(0, 0, None, None, None))(
        recalls, presentations, query_recall_position, list_length, size
    )
    available = vmap(available_recalls_with_repeats, in_axes=(0, 0, None, None, None))(
        recalls, presentations, query_recall_position, list_length, size
    )

    numerator = actual.sum(axis=0)
    denominator = available.sum(axis=0)
    return numerator / denominator


def plot_pnr(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    query_recall_position: int = 0,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    size: int = 3,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot probability of nth recall with confidence intervals.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials.
    query_recall_position : int
        0-based recall index to plot.
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
        Matplotlib Axes with the PNR plot.

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
                jit(
                    conditional_pnr_with_repeats,
                    static_argnames=("size", "query_recall_position"),
                ),
                size=size,
                query_recall_position=query_recall_position,
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

    set_plot_labels(axis, "Study Position", "Probability of Nth Recall", contrast_name)
    return axis
