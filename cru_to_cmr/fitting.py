"""Model fitting utilities.

Provides differential-evolution parameter optimization for
memory-search models, including subject-level trial masking and
loss function evaluation.

"""

import time
from typing import Any, Mapping, Optional, Type

import numpy as np
from jax import numpy as jnp
from scipy.optimize import differential_evolution
from tqdm import trange

from cru_to_cmr.typing import (
    Array,
    Bool,
    FitResult,
    Float,
    Float_,
    Integer,
    LossFnGenerator,
    MemorySearchModelFactory,
    RecallDataset,
)


__all__ = [
    "make_subject_trial_masks",
    "ScipyDE",
]

def make_subject_trial_masks(
    trial_mask: Bool[Array, " trials"], subject_vector: Integer[Array, " trials"]
):
    """Returns a list of subject-specific masks and the list of unique subjects."""
    unique_subjects = np.unique(subject_vector)
    subject_masks = [
        (subject_vector == s) & trial_mask.astype(bool) for s in unique_subjects
    ]
    return subject_masks, unique_subjects


class ScipyDE:
    """A fitting class that uses SciPy's Differential Evolution algorithm."""

    def __init__(
        self,
        dataset: RecallDataset,
        features: Optional[Float[Array, "word_count features_count"]],
        base_params: Mapping[str, Float_],
        model_create_fn: Type[MemorySearchModelFactory],
        loss_fn_generator: Type[LossFnGenerator],
        hyperparams: Optional[dict[str, Any]] = None,
    ):
        """
        Configure the fitting algorithm.

        Args:
            dataset: The dataset containing trial data (including 'subject').
            features: Optional feature matrix aligned to the vocabulary.
            base_params: A dictionary of parameters that are held fixed.
            model_create_fn: Function to create a memory search model.
            loss_fn_generator: Class implementing LossFnGenerator.
            hyperparams: Optional dictionary of hyperparameters for the fitting routine.
                May include 'bounds' (dict[str, list[float]]) and other keys
                like 'num_steps', 'pop_size', etc.
        """
        # Store essential data
        self.dataset = dataset
        self.base_params = base_params
        self.subjects = dataset["subject"].flatten()

        # configure convenience features
        if hyperparams is None:
            hyperparams = {}
        self.progress_bar = hyperparams.get("progress_bar", True)
        self.display_iterations = hyperparams.get("display_iterations", False)

        # Extract bounds for free params; store all hyperparams for convenience
        self.free_parameter_bounds = hyperparams.get("bounds", {})
        self.bounds = np.array(list(self.free_parameter_bounds.values()))

        if hyperparams is None:
            hyperparams = {}
        self.all_hyperparams = {
            "bounds": self.free_parameter_bounds,
            "num_steps": hyperparams.get("num_steps", 1000),
            "pop_size": hyperparams.get("pop_size", 15),
            "relative_tolerance": hyperparams.get("relative_tolerance", 0.001),
            "cross_over_rate": hyperparams.get("cross_over_rate", 0.9),
            "diff_w": hyperparams.get("diff_w", 0.85),
            "best_of": hyperparams.get("best_of", 1),
        }

        self.loss_fn_generator = loss_fn_generator(model_create_fn, dataset, features)

    def _fit_single_mask(
        self, trial_mask: Bool[Array, " trials"], subject_id: int = -1
    ) -> FitResult:
        """Returns result of fitting the model to the trials specified by the mask."""
        # Convert the mask to an array of trial indices
        trial_indices = jnp.where(trial_mask)[0]

        # Build a scalar loss function based on these trial indices
        loss_fn = self.loss_fn_generator(
            trial_indices, self.base_params, self.free_parameter_bounds
        )

        # Run differential evolution
        best_fitness = np.inf
        best_fit_result = None
        for _ in range(self.all_hyperparams["best_of"]):
            fit_result = differential_evolution(
                loss_fn,
                self.bounds,
                maxiter=self.all_hyperparams["num_steps"],
                popsize=self.all_hyperparams["pop_size"],
                vectorized=True,
                disp=self.display_iterations,
                tol=self.all_hyperparams["relative_tolerance"],
                mutation=self.all_hyperparams["diff_w"],
                recombination=self.all_hyperparams["cross_over_rate"],
            )
            if fit_result.fun < best_fitness:
                best_fitness = fit_result.fun
                best_fit_result = fit_result

        assert best_fit_result is not None, "No fit result found"
        return {
            "fixed": {k: float(v) for k, v in self.base_params.items()},
            "free": {
                k: self.free_parameter_bounds[k] for k in self.free_parameter_bounds
            },
            "fitness": [float(best_fitness)],
            "fits": {
                # For each base param, we just repeat its original value
                **{k: [float(v)] for k, v in self.base_params.items()},
                # For each free param, we store the optimizer's best value
                **{
                    # Map each free param name to the best-fit value
                    param_name: [float(best_fit_result.x[idx])]
                    for idx, param_name in enumerate(self.free_parameter_bounds)
                },
                "subject": [subject_id],
            },
            # These keys will be added at the top-level fit call
            "hyperparameters": {},
            "fit_time": 0.0,
        }

    def fit(
        self, trial_mask: Bool[Array, " trials"], fit_to_subjects: bool = True
    ) -> FitResult:
        """Either fit the model to all subjects individually or do a single global fit."""
        t0 = time.perf_counter()

        # If not per-subject, just do one global fit
        if not fit_to_subjects:
            result = self._fit_single_mask(trial_mask)
            result["hyperparameters"] = self.all_hyperparams
            result["fit_time"] = time.perf_counter() - t0
            return result

        # Otherwise, gather per-subject masks
        subject_trial_masks, unique_subjects = make_subject_trial_masks(
            trial_mask, self.subjects
        )

        # Prepare a global result structure
        all_results: FitResult = {
            "fixed": {k: float(v) for k, v in self.base_params.items()},
            "free": {
                k: self.free_parameter_bounds[k] for k in self.free_parameter_bounds
            },
            "fitness": [],
            "fits": {
                **{k: [] for k in self.free_parameter_bounds},
                **{k: [] for k in self.base_params},
                "subject": [],
            },
            "hyperparameters": self.all_hyperparams,
            "fit_time": 0.0,
        }

        # Optionally show progress bar
        subject_range = (
            trange(len(unique_subjects))
            if self.progress_bar
            else range(len(unique_subjects))
        )
        for s in subject_range:
            # If no trials for this subject, skip
            if np.sum(subject_trial_masks[s]) == 0:
                continue

            # Single-fit on the subject-specific mask
            fit_result = self._fit_single_mask(
                subject_trial_masks[s], int(unique_subjects[s])
            )
            all_results["fitness"] += fit_result["fitness"]

            # Show in tqdm progress bar
            if self.progress_bar:
                subject_range.set_description(  # type: ignore
                    f"Subject={unique_subjects[s]}, Fitness={fit_result['fitness'][0]}"
                )

            # Accumulate fitted parameters
            all_results["fits"]["subject"].append(int(unique_subjects[s]))

            # Append param values
            for param_name, values_list in fit_result["fits"].items():
                # Skip 'subject' to avoid double-appending
                if param_name == "subject":
                    continue
                all_results["fits"][param_name] += values_list

            # Bail out if we got a non-finite fitness
            if not jnp.isfinite(fit_result["fitness"][0]):
                raise ValueError(
                    f"Non-finite fitness for subject {unique_subjects[s]}", fit_result
                )

        all_results["fit_time"] = time.perf_counter() - t0
        return all_results
