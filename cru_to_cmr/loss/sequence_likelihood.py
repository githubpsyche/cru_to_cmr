"""Likelihood-based loss function generator for memory-search models.

Provides utilities to simulate retrieval event likelihoods per trial and to
aggregate them into a negative log-likelihood objective for fitting.

"""

from typing import Callable, Iterable, Mapping, Optional, Type

import numpy as np
from jax import jit, lax, vmap
from jax import numpy as jnp

from cru_to_cmr.helpers import log_likelihood
from cru_to_cmr.typing import (
    Array,
    Float,
    Float_,
    Integer,
    RecallDataset,
    MemorySearchModelFactory,
)


__all__ = [
    "MemorySearchLikelihoodFnGenerator",
]

class MemorySearchLikelihoodFnGenerator:
    """Generates loss functions for a given dataset and model factory.

    Creates per-trial models, produces event likelihoods, and returns a callable
    that evaluates negative log-likelihood for parameter vectors.
    """

    def __init__(
        self,
        model_factory: Type[MemorySearchModelFactory],
        dataset: RecallDataset,
        features: Optional[Float[Array, " word_pool_items features_count"]],
    ) -> None:
        """Initialize with dataset and optional feature embeddings.

        Args:
          model_factory: Class implementing `MemorySearchModelFactory`.
          dataset: Trial-wise presentations and recalls.
          features: Optional feature matrix describing word-pool items.
        """
        factory = model_factory(dataset, features)
        self.present_lists = jnp.array(dataset["pres_itemnos"])
        self.create_model = factory.create_trial_model

        # Reindex the recalled items so they match the "present_lists" indexing
        trials = np.array(dataset["recalls"])
        for trial_index in range(trials.shape[0]):
            present = self.present_lists[trial_index]
            recall = trials[trial_index]
            reindexed = np.array(
                [(present[item - 1] if item else 0) for item in recall]
            )
            trials[trial_index] = reindexed

        self.trials = jnp.array(trials)

    def predict_trial(
        self,
        trial_index: Integer[Array, ""],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, " recall_events"]:
        """Returns recall-event probabilities for one trial.

        Args:
          trial_index: Index identifying which trial to simulate.
          parameters: Model parameters specific to the trial simulation.
        """
        present = self.present_lists[trial_index]
        model = self.create_model(trial_index, parameters)
        model = lax.fori_loop(
            0, present.size, lambda i, m: m.experience(present[i]), model
        )
        model = model.start_retrieving()
        return lax.scan(
            lambda m, c: (m.retrieve(c), m.outcome_probability(c)),
            model,
            self.trials[trial_index],
        )[1]

    def present_and_predict_trials_loss(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, ""]:
        """Returns negative log likelihood across specified trials."""

        raw = vmap(self.predict_trial, in_axes=(0, None))(trial_indices, parameters)
        return log_likelihood(raw)

    def __call__(
        self,
        trial_indices: Integer[Array, " trials"],
        base_params: Mapping[str, Float_],
        free_param_names: Iterable[str],
    ) -> Callable[[np.ndarray], Float[Array, ""]]:
        """Returns a loss function specialized to trials and free parameters.

        The returned function accepts either one parameter vector or a matrix of
        parameter vectors and returns corresponding negative log-likelihood values.

        Args:
          trial_indices: Trials to evaluate.
          base_params: Fixed parameters.
          free_param_names: Names and order of free parameters.
        """

        @jit
        def single_param_loss(x: jnp.ndarray) -> Float[Array, ""]:
            """Returns loss for one parameter vector."""
            param_dict = {key: x[i] for i, key in enumerate(free_param_names)}
            return self.present_and_predict_trials_loss(
                trial_indices, {**base_params, **param_dict}
            )

        @jit
        def multi_param_loss(x: jnp.ndarray) -> Float[Array, " n_samples"]:
            """Returns one loss per parameter vector."""

            def loss_for_one_sample(x_row: jnp.ndarray) -> Float[Array, ""]:
                param_dict = {key: x_row[i] for i, key in enumerate(free_param_names)}
                return self.present_and_predict_trials_loss(
                    trial_indices, {**base_params, **param_dict}
                )

            # vmap applies loss_for_one_sample across the leading dimension of x
            return vmap(loss_for_one_sample, in_axes=1)(x)

        # Return a function that checks the dimensionality of x at runtime
        return lambda x: multi_param_loss(x) if x.ndim > 1 else single_param_loss(x)
