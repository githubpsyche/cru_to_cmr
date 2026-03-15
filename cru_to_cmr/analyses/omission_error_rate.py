"""Omission error rate analysis.

Computes the proportion of study positions that are not recalled
(omissions) as a function of serial position.

"""

__all__ = [
    "trial_omission_error_rate",
    "omission_error_rate",
    "plot_omission_error_rate",
]

from typing import Optional, Sequence

import jax.numpy as jnp
from jax import jit, vmap
from matplotlib.axes import Axes

from ..helpers import apply_by_subject, find_max_list_length
from ..repetition import all_study_positions
from ..plotting import plot_data, prepare_plot_inputs, set_plot_labels
from ..typing import Array, Bool, Float, Integer, RecallDataset


def trial_omission_error_rate(
    recalls: Integer[Array, " recall_positions"],
    presentations: Integer[Array, " study_positions"],
    size: int = 3,
) -> Bool[Array, " study_positions"]:
    """Flag omission errors at each study position for one trial.

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
        True where the study position was never recalled.

    """
    # Expand each recall token into every study position it could refer to
    list_length = presentations.shape[0]
    recalls = recalls[:list_length]
    expanded_recalls = vmap(all_study_positions, in_axes=(0, None, None))(
        recalls, presentations, size
    )

    study_positions = jnp.arange(1, list_length + 1)

    # For each study position: did it ever appear in any expanded recall list?
    position_was_recalled = vmap(
        lambda pos: jnp.any(expanded_recalls == pos)
    )(study_positions)

    # Omission = never recalled AND study slot isn't padding (presentation != 0)
    return (~position_was_recalled) & (presentations != 0)


def omission_error_rate(
    dataset: RecallDataset,
    size: int = 3,
) -> Float[Array, " study_positions"]:
    """Return position-specific omission error rate.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with ``recalls`` and ``pres_itemnos``.
    size : int
        Max study positions an item can occupy.

    Returns
    -------
    Float[Array, " study_positions"]
        Mean omission rate at each study position.

    """
    recalls = dataset["recalls"]
    presentations = dataset["pres_itemnos"]
    return vmap(trial_omission_error_rate, in_axes=(0, 0, None))(
        recalls,
        presentations,
        size,
    ).mean(axis=0)


def plot_omission_error_rate(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    size: int = 3,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot omission error rate curves with confidence intervals.

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
        Matplotlib Axes with omission error rate curves.

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
                jit(omission_error_rate, static_argnames=("size",)),
                size=size,
            )
        )
        color = color_cycle[data_index % len(color_cycle)]
        xvals = jnp.arange(max_list_length) + 1
        subject_values = subject_values[:, :max_list_length]
        plot_data(
            axis,
            xvals,
            subject_values,
            labels[data_index],
            color,
            confidence_level=confidence_level,
        )

    set_plot_labels(axis, "Study Position", "Omission Error Rate", contrast_name)
    return axis
