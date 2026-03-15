"""Order error rate analysis.

Computes the proportion of recall attempts where the recalled item
appears at the wrong serial position (transposition errors) as a
function of output position.

"""

__all__ = ["trial_order_error_rate", "order_error_rate", "plot_order_error_rate"]

from typing import Optional, Sequence

import jax.numpy as jnp
from jax import jit, vmap
from matplotlib.axes import Axes

from ..helpers import apply_by_subject, find_max_list_length
from ..repetition import all_study_positions
from ..plotting import plot_data, prepare_plot_inputs, set_plot_labels
from ..typing import Array, Bool, Float, Integer, RecallDataset


def trial_order_error_rate(
    recalls: Integer[Array, " recall_positions"],
    presentations: Integer[Array, " study_positions"],
    size: int = 3,
) -> Bool[Array, " study_positions"]:
    """Flag order errors at each study position for one trial.

    Parameters
    ----------
    recalls : Integer[Array, " recall_positions"]
        Recall sequence for a trial. 1-indexed; 0 pads.
    presentations : Integer[Array, " study_positions"]
        Presented item IDs for the trial. 1-indexed; 0 pads.
    size : int
        Max study positions an item can occupy.

    Returns
    -------
    Bool[Array, " study_positions"]
        True where a list item was recalled in the wrong position.

    """
    list_len = presentations.shape[0]
    study_pos_1 = jnp.arange(1, list_len + 1)  # 1-based indexes

    # 1. Token produced at each output slot i (pad with 0 if recall shorter)
    recall_tokens = jnp.where(
        study_pos_1 - 1 < recalls.shape[0],  # convert to 0-based for slicing
        recalls[:list_len],
        0,  # no response -> treat as 0
    )

    # 2. Is that token on today's list (and non-zero)?
    #    token k is "on the list" if 1 <= k <= list_len and presentations[k-1] != 0
    token_on_list = (
        (recall_tokens > 0)
        & (recall_tokens <= list_len)
        & (presentations[recall_tokens - 1] != 0)
    )

    # 3. For each output slot, does its token actually belong here?
    #    Use all_study_positions to map token -> all study slots of that item.
    expanded = vmap(all_study_positions, in_axes=(0, None, None))(
        recall_tokens, presentations, size
    )  # shape: (list_len, size)

    correct_here = jnp.any(expanded == study_pos_1[:, None], axis=1)

    # 4. Order-error logic, ignoring padded study slots (presentations == 0)
    return token_on_list & (~correct_here) & (presentations != 0)


def order_error_rate(
    dataset: RecallDataset,
    size: int = 3,
) -> Float[Array, " study_positions"]:
    """Return position-specific order error rate.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with ``recalls`` and ``pres_itemnos``.
    size : int
        Max study positions an item can occupy.

    Returns
    -------
    Float[Array, " study_positions"]
        Mean order error rate at each study position.

    """
    recalls = dataset["recalls"]
    presentations = dataset["pres_itemnos"]

    return vmap(trial_order_error_rate, in_axes=(0, 0, None))(
        recalls,
        presentations,
        size,
    ).mean(axis=0)


def plot_order_error_rate(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    size: int = 3,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot order error rate curves with confidence intervals.

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
        Matplotlib Axes with order error rate curves.

    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )

    if labels is None:
        labels = ["" for _ in datasets]

    # Identify the largest list length across datasets, so we can plot consistently
    max_list_length = find_max_list_length(datasets, trial_masks)

    for data_index, data_dict in enumerate(datasets):
        subject_values = jnp.vstack(
            apply_by_subject(
                data_dict,
                trial_masks[data_index],
                jit(order_error_rate, static_argnames=("size",)),
                size=size,
            )
        )
        color = color_cycle[data_index % len(color_cycle)]
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

    set_plot_labels(axis, "Study Position", "Order Error Rate", contrast_name)
    return axis
