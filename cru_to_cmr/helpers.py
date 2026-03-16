"""Data loading and general-purpose helpers.

Provides HDF5 dataset loading, trial and recall mask generation,
subject-level analysis application, and assorted utility functions
used across the package.

"""

import importlib
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import h5py
import jax.numpy as jnp
import numpy as np  # preserved here in order to evaluate some trial queries

from cru_to_cmr.typing import Array, Bool, Float, Integer, Real, RecallDataset


__all__ = [
    "all_rows_identical",
    "find_project_root",
    "format_floats",
    "log_likelihood",
    "import_from_string",
    "generate_trial_mask",
    "load_data",
    "limit_to_first_subjects",
    "find_max_list_length",
    "apply_by_subject",
]


def find_project_root(marker: str = ".git") -> str:
    """Finds the project root by traversing upwards from cwd
    until a directory containing `marker` is found."""
    start = Path.cwd()
    for path in [start, *start.parents]:
        if (path / marker).exists():
            return str(path)
    raise FileNotFoundError(f"Could not find project root containing {marker}.")


def format_floats(iterable: Iterable[float], precision: int = 2) -> List[str]:
    """Formats a list of floats to a specified precision."""
    format_str = f"{{:.{precision}f}}"
    return [format_str.format(x) for x in iterable]


def all_rows_identical(arr: Real[Array, " x y"]) -> bool:
    """Return whether all rows in the 2D array are identical."""
    return jnp.all(arr == arr[0]).item()


def log_likelihood(likelihoods: Float[Array, "trial_count ..."]) -> Float[Array, ""]:
    """Return the summed log likelihood over specified likelihoods."""
    return -jnp.sum(jnp.log(likelihoods))


def import_from_string(import_string):
    """
    Import a module or function from a string.

    Args:
        import_string: A string in the format 'module.submodule.ClassName' or 'module.function_name'.

    Returns:
        The imported module or function.

    Raises:
        ImportError: If the import string is not valid.
    """
    module_name, function_name = import_string.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def generate_trial_mask(
    data: RecallDataset, trial_query: Optional[str]
) -> Bool[Array, " trial_count"]:
    """Returns a boolean mask for selecting trials based on a specified query condition.

    Args:
        data: dict containing trial data arrays, including a "recalls" key with an array.
        trial_query: condition to evaluate, which should return a boolean array.
        If None, returns a mask that selects all trials.
    """
    if trial_query is None:
        return jnp.ones(data["recalls"].shape[0], dtype=bool)
    return eval(trial_query).flatten()


def load_data(data_path: str, max_subjects: int = 0) -> RecallDataset:
    """
    Loads and processes an HDF5 dataset from the specified file.

    This function opens the HDF5 file at `data_path`, extracts all datasets stored
    under the "/data" group, transposes each array, and converts them into jnp
    arrays for further processing.

    Args:
        data_path: Path to the HDF5 file containing the dataset.

    Returns:
        A dictionary where each key corresponds to a dataset name and each value is
        a jax.numpy array containing the transposed data.
    """
    with h5py.File(data_path, "r") as f:
        result = {key: jnp.array(f["/data"][key][()].T) for key in f["/data"].keys()}  # type: ignore

    if max_subjects == 0:
        return result  # type: ignore
    else:
        return limit_to_first_subjects(result, max_subjects)  # type: ignore


def limit_to_first_subjects(
    data: RecallDataset,
    max_subjects: int,
) -> RecallDataset:
    """Returns dataset restricted to the first `max_subjects` unique subjects.

    Args:
      data: Trial-indexed arrays with a `subject` column shaped [trial_count, 1].
      max_subjects: Maximum number of subjects to retain, preserving encounter order.
    """
    subject_ids = data["subject"].reshape(-1)
    unique_subjects = np.unique(np.asarray(subject_ids))
    cutoff_index = min(max_subjects, unique_subjects.size) - 1
    subject_cutoff = unique_subjects[cutoff_index]
    include_mask = subject_ids <= subject_cutoff
    return {key: value[include_mask] for key, value in data.items()}  # type: ignore


def find_max_list_length(
    datasets: Sequence[RecallDataset],
    trial_masks: Sequence[Bool[Array, " trial_count"]],
) -> int:
    """Returns highest list length across multiple datasets, given trial masks.

    Args:
        datasets: dataset dicts, each with a key "listLength" mapping to a numpy array.
        trial_masks: Boolean numpy arrays used as masks to filter each dataset.
    """
    return max(
        jnp.max(data["listLength"][trial_mask]).item()
        for data, trial_mask in zip(datasets, trial_masks)
    )


def apply_by_subject(
    data: RecallDataset,
    trial_mask: Bool[Array, " trial_count"],
    func: Callable,
    *args,
    **kwargs,
) -> list[jnp.ndarray]:
    """Apply ``func`` per subject using the entire masked dataset slice.

    Args:
        data: Dataset containing trial-indexed arrays (e.g., ``recalls``, ``pres_itemids``).
        trial_mask: Boolean mask selecting trials to include.
        func: Callable invoked as ``func(subject_data, *args, **kwargs)`` where
          ``subject_data`` is a masked `RecallDataset`.
        *args: Additional positional arguments forwarded to ``func``.
        **kwargs: Additional keyword arguments forwarded to ``func``.
    """

    subject_indices = data["subject"].flatten()
    results: list[jnp.ndarray] = []

    for subject in jnp.unique(data["subject"]):
        subject_mask = jnp.logical_and(subject_indices == subject, trial_mask)
        if jnp.sum(subject_mask) == 0:
            continue
        subject_dataset: RecallDataset = {
            key: value[subject_mask]  # type: ignore
            for key, value in data.items()
        }
        results.append(func(subject_dataset, *args, **kwargs))

    return results
