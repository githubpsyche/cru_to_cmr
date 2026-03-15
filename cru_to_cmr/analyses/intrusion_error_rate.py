"""Intrusion error rate analysis.

Computes the proportion of recall attempts that are extra-list
intrusions (items not on the current study list) as a function of
output position.

"""

__all__ = [
    "trial_intrusion_error_rate",
    "intrusion_error_rate",
    "plot_intrusion_error_rate",
]

from typing import Optional, Sequence

import jax.numpy as jnp
from jax import jit, vmap
from matplotlib.axes import Axes

from ..helpers import apply_by_subject, find_max_list_length
from ..plotting import plot_data, prepare_plot_inputs, set_plot_labels
from ..typing import Array, Bool, Float, Integer, RecallDataset


def trial_intrusion_error_rate(
    recalls: Integer[Array, " recall_positions"],
    presentations: Integer[Array, " study_positions"],
) -> Bool[Array, " study_positions"]:
    """Flag intrusion errors at each study position for one trial.

    Parameters
    ----------
    recalls : Integer[Array, " recall_positions"]
        Recall sequence for a trial. 1-indexed; 0 pads.
    presentations : Integer[Array, " study_positions"]
        Presented item IDs for the trial. 1-indexed.

    Returns
    -------
    Bool[Array, " study_positions"]
        True where the recalled item is not on the study list.

    """
    list_length = presentations.shape[0]
    study_position_indices = jnp.arange(list_length)
    recalled_items_at_position = recalls[study_position_indices]
    item_is_in_study_list = jnp.any(
        recalled_items_at_position[:, None] == presentations[None, :],
        axis=1,
    )
    is_intrusion = (recalled_items_at_position != 0) & (~item_is_in_study_list)
    return is_intrusion & (presentations != 0)


def intrusion_error_rate(dataset: RecallDataset) -> Float[Array, " study_positions"]:
    """Return position-specific intrusion error rate.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with ``recalls`` and ``pres_itemnos``.

    Returns
    -------
    Float[Array, " study_positions"]
        Mean intrusion rate at each study position.

    """
    recalls = dataset["recalls"]
    presentations = dataset["pres_itemnos"]

    return vmap(trial_intrusion_error_rate, in_axes=(0, 0))(
        recalls, presentations
    ).mean(axis=0)


def plot_intrusion_error_rate(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot intrusion error rate curves with confidence intervals.

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
    confidence_level : float
        Confidence level for error bounds.

    Returns
    -------
    Axes
        Matplotlib Axes with intrusion error rate curves.

    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )

    if labels is None:
        labels = ["" for _ in datasets]

    # Identify the largest list length across datasets, so we can plot consistently
    max_list_length = find_max_list_length(datasets, trial_masks)

    for data_index, data_dict in enumerate(datasets):
        # We'll apply the accurate_spc function to each subject, then stack
        subject_values = jnp.vstack(
            apply_by_subject(
                data_dict,
                trial_masks[data_index],
                jit(intrusion_error_rate),
            )
        )

        # Plot
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

    set_plot_labels(axis, "Study Position", "Intrusion Error Rate", contrast_name)
    return axis
