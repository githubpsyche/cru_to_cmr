"""Confusable-item likelihood loss function.

Extends the standard sequence likelihood to support recall events
where the recalled item may be confused with a similar item,
computing likelihoods under an error-tolerant retrieval model.

"""

from typing import Iterable, Mapping, Optional, Type

from jax import lax, vmap
from jax import numpy as jnp

from jaxcmr.helpers import log_likelihood
from jaxcmr.typing import (
    Array,
    Float,
    Float_,
    Integer,
    MemorySearch,
    MemorySearchModelFactory,
    RecallDataset,
)


__all__ = [
    "predict_and_simulate_recalls",
    "MemorySearchLikelihoodLoss",
]

def predict_and_simulate_recalls(
    model: MemorySearch, choices: Integer[Array, " recall_events"]
) -> tuple[MemorySearch, Float[Array, " recall_events"]]:
    """
    Return the updated model and the outcome probabilities of a chain of retrieval events.
    Args:
        model: the current memory search model.
        choices: the indices of the items to retrieve (1-indexed) or 0 to stop.
    """
    return lax.scan(
        lambda m, c: (m.retrieve(c), m.outcome_probability(c)), model, choices
    )


class MemorySearchLikelihoodLoss:
    def __init__(
        self,
        model_factory_cls: Type[MemorySearchModelFactory],
        dataset: RecallDataset,
        features: Optional[Float[Array, " word_pool_items features_count"]],
    ) -> None:
        """Initialize the factory with the specified trials and trial data."""
        
        assert "pres_itemids" in dataset
        assert "rec_itemids" in dataset

        self.factory = model_factory_cls(dataset, features)
        self.create_model = self.factory.create_trial_model
        self.present_lists = jnp.array(dataset["pres_itemids"])
        self.trials = jnp.array(dataset["rec_itemids"])

    def init_model_for_retrieval(
        self,
        trial_index: Integer[Array, ""],
        parameters: Mapping[str, Float_],
    ) -> MemorySearch:
        """
        Create and initialize a MemorySearch model for a given trial's presentation list.
        """
        present = self.present_lists[trial_index]
        model = self.create_model(trial_index, parameters)
        model = lax.fori_loop(
            0, present.size, lambda i, m: m.experience(present[i]), model
        )
        return model.start_retrieving()

    def base_predict_trials(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Integer[Array, " trials recall_events"]:
        """
        Predict outcomes for each trial using a single initial model (from trial 0),
        skipping re-experiencing items for each subsequent trial.
        Only valid if all present-lists match.
        """
        model = self.init_model_for_retrieval(trial_indices[0], parameters)
        return vmap(predict_and_simulate_recalls, in_axes=(None, 0))(
            model, self.trials[trial_indices]
        )[1]

    def present_and_predict_trials(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Integer[Array, " trials recall_events"]:
        """
        Predict outcomes for each trial by creating a new model for each trial
        (re-experiencing items per trial).
        """

        def present_and_predict_trial(i):
            model = self.init_model_for_retrieval(i, parameters)
            return predict_and_simulate_recalls(model, self.trials[i])[1]

        return vmap(present_and_predict_trial)(trial_indices)

    def base_predict_trials_loss(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, ""]:
        """Return negative log-likelihood for the 'base' approach."""
        return log_likelihood(self.base_predict_trials(trial_indices, parameters))

    def present_and_predict_trials_loss(
        self,
        trial_indices: Integer[Array, " trials"],
        parameters: Mapping[str, Float_],
    ) -> Float[Array, ""]:
        """Return negative log-likelihood for the 'present-and-predict' approach."""
        return log_likelihood(
            self.present_and_predict_trials(trial_indices, parameters)
        )

    def __call__(
        self,
        trial_indices: Integer[Array, " trials"],
        base_params: Mapping[str, Float_],
        free_param_names: Iterable[str],
        x: jnp.ndarray,
    ) -> Float[Array, " n_samples"]:
        """Returns one loss per parameter vector."""
        free_param_names = tuple(free_param_names)

        selected_lists = self.present_lists[trial_indices]
        use_base_loss = jnp.all(selected_lists == selected_lists[0])

        def loss_for_one_sample(x_row: jnp.ndarray) -> Float[Array, ""]:
            param_dict = {key: x_row[i] for i, key in enumerate(free_param_names)}
            params = {**base_params, **param_dict}
            return lax.cond(
                use_base_loss,
                lambda _: self.base_predict_trials_loss(trial_indices, params),
                lambda _: self.present_and_predict_trials_loss(trial_indices, params),
                operand=None,
            )

        return vmap(loss_for_one_sample, in_axes=1)(x)


# Compatibility aliases. Do not use in new code.
MemorySearchLikelihoodFnGenerator = MemorySearchLikelihoodLoss
