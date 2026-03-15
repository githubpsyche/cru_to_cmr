"""Shared plotting utilities for analysis modules.

Provides axis initialization, confidence-interval computation via
bootstrap, and standardized plot-data / label-setting helpers used
by all analysis plot functions.

"""

from collections.abc import Sequence
from typing import Optional, TypeVar, cast

import matplotlib.pyplot as plt
from jax import numpy as jnp
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from scipy.stats import bootstrap

from cru_to_cmr.typing import Array, Bool, Real, RecallDataset

__all__ = [
    "init_plot",
    "prepare_plot_inputs",
    "plot_data",
    "calculate_errors",
    "plot_with_error_bars",
    "plot_without_error_bars",
    "set_plot_labels",
]


T = TypeVar("T")


def _normalize_to_list(value: Sequence[T] | T) -> list[T]:
    """Returns value as a list.

    Args:
        value: Value to normalize into a list.
    """
    if isinstance(value, dict):
        return [cast(T, value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(cast(Sequence[T], value))
    return [cast(T, value)]


def prepare_plot_inputs(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    color_cycle: Optional[list[str]],
    axis: Optional[Axes],
) -> tuple[
    Axes,
    list[RecallDataset],
    list[Bool[Array, " trial_count"]],
    list[str],
]:
    """Returns normalized plot inputs for analysis plotters.

    Args:
        datasets: Datasets to plot.
        trial_masks: Masks selecting trials for each dataset.
        color_cycle: Colors for plotting each dataset.
        axis: Existing axis to plot on.

    Returns:
        (axis, datasets, trial_masks, color_cycle): Normalized plot inputs.
    """
    axis = init_plot(axis)
    datasets_list = _normalize_to_list(datasets)
    trial_masks_list = _normalize_to_list(trial_masks)
    if color_cycle is None:
        color_cycle = [each["color"] for each in rcParams["axes.prop_cycle"]]
    return axis, datasets_list, trial_masks_list, color_cycle


def segment_by_nan(vector: jnp.ndarray) -> list[tuple[int, int]]:
    """Return index ranges split by NaN entries.

    Args:
        vector: Vector to segment.

    Returns:
        List of (start, end) index pairs.
    """
    segments = []
    start_idx = 0
    for i in range(len(vector)):
        if jnp.isnan(vector[i]):
            segments.append((start_idx, i))
            start_idx = i + 1
    if start_idx < len(vector):
        segments.append((start_idx, len(vector)))
    return segments


def init_plot(axis: Optional[Axes] = None) -> Axes:
    """Return an initialized plotting axis.

    Args:
        axis: An existing axis to use for the plot.
    """
    if axis is None:
        plt.figure()
        axis = plt.gca()
    return axis


def calculate_errors(
    y_values: jnp.ndarray,
    y_mean: jnp.ndarray,
    *,
    confidence_level: float = 0.95,
) -> jnp.ndarray:
    """Return error bars from bootstrapped confidence intervals.

    Args:
        y_values: Array of y values.
        y_mean: Array of y mean values.
        confidence_level: Confidence level for the interval.
    """
    errors = jnp.zeros((2, len(y_mean)))
    bootstrapped_confidence_intervals = bootstrap(
        (y_values,), jnp.nanmean, confidence_level=confidence_level
    ).confidence_interval

    return (
        errors.at[0]
        .set(y_mean - bootstrapped_confidence_intervals.low)
        .at[1]
        .set(bootstrapped_confidence_intervals.high - y_mean)
    )


def plot_with_error_bars(
    axis: Axes,
    x_values: Real[Array, " trial_count values"],
    y_values: Real[Array, " trial_count values"],
    y_mean: Real[Array, " values"],
    segments: list[tuple[int, int]],
    label: str,
    color: str,
    *,
    confidence_level: float = 0.95,
):
    """Returns a plot of a line graph with error bars on the specified axis.

    Args:
        axis: The axis object to plot on.
        x_values: The x-axis values.
        y_values: The y-axis values.
        y_mean: The mean y-axis values.
        segments: A list of tuples indicating start and end indices for segments.
        label: The label for the line graph.
        color: The color of the line graph.
        confidence_level: Confidence level for the bootstrapped intervals.
    """
    errors = calculate_errors(y_values, y_mean, confidence_level=confidence_level)

    for index, (start, end) in enumerate(segments):
        axis.errorbar(
            x_values[start:end],
            y_mean[start:end],
            errors[:, start:end],
            label=label if index == 0 else None,
            color=color,
            capsize=3,
            marker="o",
            markersize=5,
            linewidth=1.5,
        )


def plot_without_error_bars(
    axis: Axes,
    x_values: Real[Array, " trial_count values"],
    y_mean: Real[Array, " values"],
    segments: list[tuple[int, int]],
    label: str,
    color: str,
):
    """Returns a plot of a line graph without error bars on the specified axis.

    Args:
        axis: The axis object to plot on.
        x_values: The x-axis values.
        y_values: The y-axis values.
        y_mean: The mean y-axis values.
        segments: A list of tuples indicating start and end indices for segments.
        label: The label for the line graph.
        color: The color of the line graph.
    """
    for index, (start, end) in enumerate(segments):
        axis.plot(
            x_values[start:end],
            y_mean[start:end],
            label=label if index == 0 else None,
            color=color,
            marker="o",
            markersize=5,
            linewidth=1.5,
        )


def plot_data(
    axis: Axes,
    x_values: Real[Array, " trial_count values"],
    y_values: Real[Array, " trial_count values"],
    label: str,
    color: str,
    *,
    draw_ci: bool = True,
    confidence_level: float = 0.95,
) -> Axes:
    """Returns axis plotting data as line segment(s) on the provided axis.

    Args:
        axis: the axis to plot on; if None, a new figure is created.
        x_values: Vector of x values to plot.
        y_values: vector or matrix of y values to plot.
        label: name for the plotted data.
        color: color for the plotted data.
        draw_ci: Whether to draw confidence intervals.
        confidence_level: Confidence level for the bootstrapped intervals.
    """
    if y_values.ndim == 1:
        y_values = y_values[jnp.newaxis, :]
    y_mean = jnp.nanmean(y_values, axis=0)
    segments = segment_by_nan(y_mean)

    if len(y_values) > 1 and draw_ci:
        plot_with_error_bars(
            axis,
            x_values,
            y_values,
            y_mean,
            segments,
            label,
            color,
            confidence_level=confidence_level,
        )
    else:
        plot_without_error_bars(axis, x_values, y_mean, segments, label, color)

    x = jnp.asarray(x_values)
    x = x[jnp.isfinite(x)]
    if x.size and bool(jnp.allclose(x, jnp.round(x), atol=1e-9)):
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    return axis

def set_plot_labels(
    axis: Axes, xlabel: str, ylabel: str, contrast_name: Optional[str] = None
) -> Axes:
    """Returns an axis with modified labels and optional legend settings for a plot.

    Args:
        axis: The axis to modify.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        contrast_name: Name of the contrast for the legend, if applicable.
    """
    axis.set_xlabel(xlabel, fontsize=16)
    axis.set_ylabel(ylabel, fontsize=16)

    if contrast_name:
        axis.legend(title=contrast_name)

    axis.tick_params(labelsize=14)
    for loc in ("top", "right"):
        axis.spines[loc].set_visible(False)
    axis.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    return axis
