"""Parameter summary and model comparison utilities.

Provides functions for loading optimized parameters, computing
confidence intervals, generating t-test matrices, and computing
AIC/BIC model comparison statistics.

"""

from typing import Callable, Optional

import numpy as np
from scipy.stats import t



__all__ = [
    "calculate_ci",
    "add_summary_lines",
    "summarize_parameters",
]

def calculate_ci(data: list[float], confidence=0.95) -> float:
    """Returns the confidence interval for a list of values.

    Args:
        data (list[float]): Values to calculate the confidence interval for.
        confidence (float, optional): The confidence level for the interval. Defaults to 0.95.
    """
    assert len(data) > 1
    n = len(data)
    stderr = np.std(np.array(data), ddof=1) / np.sqrt(n)
    return stderr * t.ppf((1 + confidence) / 2.0, n - 1)


def _normalize_variant(values: list[float]) -> Optional[np.ndarray]:
    """Returns normalized variant values or None when no usable data is present.

    Args:
      values: Subject-level measurements for a single model variant.
    """
    variant_array = np.asarray(values, dtype=float)
    if variant_array.size == 0 or np.isnan(variant_array).all():
        return None
    return variant_array


def _format_row(parameter_label: str, statistic_label: str, values: list[str]) -> str:
    """Returns a Markdown table row for the summary output.

    Args:
      parameter_label: Name of the parameter or metric for the row.
      statistic_label: Statistic identifier (e.g., mean, std).
      values: Formatted statistic values for each model variant.
    """
    cells = [parameter_label, statistic_label, *values]
    sanitized = [cell or "" for cell in cells]
    return "| " + " | ".join(sanitized) + " |\n"


def _format_mean_cell(
    array: Optional[np.ndarray], raw_values: list[float], include_ci: bool
) -> str:
    """Returns the formatted mean cell text, optionally with confidence intervals.

    Args:
      array: Prepared numeric data for a model variant.
      raw_values: Original values for confidence interval calculation.
      include_ci: Whether to append the confidence interval.
    """
    if array is None:
        return ""
    mean_value = np.mean(array)
    if np.isnan(mean_value):
        return ""
    if include_ci:
        return f"{mean_value:.2f} +/- {calculate_ci(raw_values):.2f}"
    return f"{mean_value:.2f}"


def _format_stat_cell(
    array: Optional[np.ndarray], reducer: Callable[[np.ndarray], float]
) -> str:
    """Returns the formatted statistic cell using the provided reducer.

    Args:
      array: Prepared numeric data for a model variant.
      reducer: Function that computes the statistic of interest.
    """
    if array is None:
        return ""
    value = reducer(array)
    if np.isnan(value):
        return ""
    return f"{value:.2f}"


def add_summary_lines(
    md_table: str,
    errors: list[list[float]],
    label: str,
    include_std=False,
    include_ci=False,
) -> str:
    """Add summary statistics rows to a Markdown table segment.

    Args:
      md_table: Markdown table fragment to extend.
      errors: Values grouped by model variant.
      label: Parameter or metric name for the new rows.
      include_std: Whether to include standard deviation rows.
      include_ci: Whether to include confidence interval text for means.
    """
    display_label = label.replace("_", " ")
    variant_arrays = [_normalize_variant(values) for values in errors]

    mean_values = [
        _format_mean_cell(variant_array, raw_values, include_ci)
        for variant_array, raw_values in zip(variant_arrays, errors)
    ]
    md_table += _format_row(display_label, "mean", mean_values)

    if include_std:
        std_values = [
            _format_stat_cell(variant_array, np.std) for variant_array in variant_arrays
        ]
        md_table += _format_row("", "std", std_values)

    min_values = [
        _format_stat_cell(variant_array, np.nanmin) for variant_array in variant_arrays
    ]
    md_table += _format_row("", "min", min_values)

    max_values = [
        _format_stat_cell(variant_array, np.nanmax) for variant_array in variant_arrays
    ]
    md_table += _format_row("", "max", max_values)

    return md_table


def summarize_parameters(
    model_data: list[dict],
    query_parameters: Optional[list[str]] = None,
    include_std=False,
    include_ci=False,
) -> str:
    """Returns a Markdown table of parameter statistics across model variants.

    The table includes the mean (with optional confidence intervals), standard deviation,
    minimum, and maximum for each requested parameter.

    Args:
      model_data: Collection of model summaries containing `name`, `fitness`, and `fits`.
      query_parameters: Ordered parameter identifiers to surface; defaults to all found.
      include_std: Whether to include standard deviation rows.
      include_ci: Whether to append confidence intervals to mean rows.
    """
    # Extract model names in input order
    model_names = [variant["name"] for variant in model_data]

    # identify query parameters; by default, is all unique fixed params across model variants
    if query_parameters is None:
        query_parameters = []
        for entry in model_data:
            for param in entry["fixed"].keys():
                if param not in query_parameters:
                    query_parameters.append(param)

    header_names = [n.replace("_", " ") for n in model_names]
    md_table = "| Parameter | Statistic | " + " | ".join(header_names) + " |\n"
    md_table += "|---|---" + ("|---" * len(model_data)) + "|\n"

    # likelihood entry first
    values = [variant_data["fitness"] for variant_data in model_data]
    md_table = add_summary_lines(
        md_table, values, "fitness", include_std=include_std, include_ci=include_ci
    )

    # Compute the mean and confidence interval for params for each model variant
    for param in query_parameters:
        values = []
        for variant_data in model_data:
            subject_count = len(variant_data["fitness"])
            fallback = [np.nan] * subject_count
            values.append(variant_data["fits"].get(param, fallback))
        md_table = add_summary_lines(
            md_table, values, param, include_std=include_std, include_ci=include_ci
        )

    return md_table
