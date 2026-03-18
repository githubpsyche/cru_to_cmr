"""Simulation utilities for confusable-item models.

Extends the standard simulation pipeline to handle recall events
where the model may output a confused (similar) item instead of the
target, matching the error-tolerant retrieval in
``confusable_likelihood``.

"""

from typing import Mapping, Optional, Sequence, Type

import numpy as np
import jax.numpy as jnp
from jax import lax, random, vmap
from jax.tree_util import tree_map

from jaxcmr.typing import (
    Array,
    Bool,
    Float,
    Float_,
    Int_,
    Integer,
    MemorySearch,
    MemorySearchModelFactory,
    PRNGKeyArray,
    RecallDataset,
)


__all__ = [
    "item_to_study_positions",
    "simulate_free_recall",
    "simulate_study_and_free_recall",
    "MemorySearchSimulator",
    "preallocate_for_h5_dataset",
    "simulate_h5_from_h5",
    "parameter_shifted_simulate_h5_from_h5",
]

def item_to_study_positions(
    item: Int_,
    presentation: Integer[Array, " list_length"],
    size: int,
) -> Integer[Array, " size"]:
    """Return the one-indexed study positions of an item in a 1D presentation sequence.

    Args:
        item: the item index.
        presentation: the 1D presentation sequence.
        size: number of non-zero entries to return.
    """
    return lax.cond(
        item == 0,
        lambda: jnp.zeros(size, dtype=int),
        lambda: jnp.nonzero(presentation == item, size=size, fill_value=-1)[0] + 1,
    )


def _single_free_recall(
    model: MemorySearch, rng: PRNGKeyArray
) -> tuple[MemorySearch, Integer[Array, ""]]:
    """Return model state and choice after performing a free recall event.

    Args:
        model: the current memory search model, after starting retrieval.
        rng: key for random number generation.
    """
    p_all = model.outcome_probabilities()
    choice = random.choice(rng, p_all.shape[0], p=p_all)
    return model.retrieve(choice), choice


def _maybe_free_recall(
    model: MemorySearch, rng: PRNGKeyArray
) -> tuple[MemorySearch, Integer[Array, ""]]:
    """Return model state and choice after performing a free recall event if the model is active.

    Args:
        model: the current memory search model, after starting retrieval.
        rng: key for random number generation.
    """
    return lax.cond(
        model.is_active,
        _single_free_recall,
        lambda m, _: (m, 0),
        model,
        rng,
    )


def simulate_free_recall(
    model: MemorySearch, list_length: int, rng: PRNGKeyArray
) -> tuple[MemorySearch, Integer[Array, ""] | PRNGKeyArray]:
    """Return model state and choices after performing free recall events until termination.

    Args:
        model: the current memory search model, after starting retrieval.
        list_length: the length of the study and recall sequences.
        rng: key for random number generation.
    """
    return lax.scan(_maybe_free_recall, model, random.split(rng, list_length))


def simulate_study_and_free_recall(
    model: MemorySearch, present: Integer[Array, " study_events"], rng: PRNGKeyArray
) -> tuple[MemorySearch, Integer[Array, ""]]:
    """Return model state and choices after simulating a trial.

    Args:
        model: the current memory search model.
        present: the indices of the items to present (1-indexed).
        rng: key for random number generation.
    """
    model = lax.fori_loop(0, present.size, lambda i, m: m.experience(present[i]), model)
    model = model.start_retrieving()
    return simulate_free_recall(model, present.size, rng)


class MemorySearchSimulator:
    """Stateless wrapper that can be vmapped over **trials**."""

    def __init__(
        self,
        model_factory: Type[MemorySearchModelFactory],
        dataset: RecallDataset,
        features: Optional[Float[Array, " word_pool_items features_count"]],
    ) -> None:
        """Initialize the factory with the specified trials and trial data."""
        factory = model_factory(dataset, features)
        self.create_model = factory.create_trial_model
        self.present_lists = jnp.array(dataset["pres_itemids"])
        self.empty = jnp.zeros(dataset["recalls"].shape[-1], jnp.int32)

    def simulate_trial(
        self,
        trial_index: Integer[Array, ""],
        subject_index: Integer[Array, " subject_count"],
        parameters: Mapping[str, Float_],
        rng: PRNGKeyArray,
    ) -> Integer[Array, " recalls"]:
        model = self.create_model(
            trial_index, tree_map(lambda p: p[subject_index], parameters)
        )
        return simulate_study_and_free_recall(
            model, self.present_lists[trial_index], rng
        )[1]


def _parameter_indices_for_subjects(
    subject_ids: Integer[Array, " trial_count"],
    parameters: Mapping[str, Float_],
) -> Integer[Array, " trial_count"]:
    """Returns zero-based parameter indices aligned with subject identifiers.

    Args:
        subject_ids: Raw subject identifiers from the dataset.
        parameters: Mapping of per-subject parameter arrays. When a ``'subject'``
            entry is present, it is used to translate raw IDs to contiguous indices.
    """
    parameter_subjects = parameters.get("subject")
    if parameter_subjects is None:
        return subject_ids

    parameter_subjects = jnp.asarray(parameter_subjects).reshape(-1)
    sort_indices = jnp.argsort(parameter_subjects)
    sorted_subjects = parameter_subjects[sort_indices]
    lookup_positions = jnp.searchsorted(sorted_subjects, subject_ids)
    matched_subjects = sorted_subjects[lookup_positions]

    if not np.array_equal(np.array(matched_subjects), np.array(subject_ids)):
        raise ValueError(
            "Dataset subject IDs are not present in parameters['subject']."
        )

    return sort_indices[lookup_positions].astype(jnp.int32)


def preallocate_for_h5_dataset(
    data: RecallDataset, trial_mask: Bool[Array, " trial_count"], experiment_count: int
) -> RecallDataset:
    """Returns dict with same keys as `data`; each is an array replicated by `experiment_count`.

    Arrays are allocated for each key in the input data.
    For 'recalls', the array is initialized with zeros.

    Args:
        data: Dictionary containing dataset arrays.
        trial_mask: Boolean array to select trials.
        experiment_count: Number of times to replicate each array.
    """
    return tree_map(
        lambda x: jnp.repeat(x[trial_mask], experiment_count, axis=0),
        data,
    )


def simulate_h5_from_h5(
    model_factory: Type[MemorySearchModelFactory],
    dataset: RecallDataset,
    features: Optional[Float[Array, " word_pool_items features_count"]],
    parameters: dict[str, Float[Array, " subject_count"]],
    trial_mask: Bool[Array, " trial_count"],
    experiment_count: int,
    rng: PRNGKeyArray,
    size: int = 3,
) -> RecallDataset:
    """
    Simulates dataset from existing dataset using a memory search model parameterized by subject.

    Args:
        model_factory: Factory class for creating memory search model instances.
        dataset: Original H5 dataset containing trial data.
        features: Optional feature matrix describing word-pool items.
        parameters: Dictionary of simulation parameters, parameterized per subject.
        trial_mask: Boolean array specifying which trials to simulate.
        experiment_count: Number of simulation iterations per trial.
        rng: PRNGKeyArray for random number generation.
        size: Maximum number of study positions to return for each item during reindexing.
    """

    sim_h5 = preallocate_for_h5_dataset(dataset, trial_mask, experiment_count)
    simulator = MemorySearchSimulator(model_factory, sim_h5, features)

    # Flat trial + subject index vectors (static shapes)
    total_trials = sim_h5["subject"].size
    trial_indices = jnp.arange(total_trials, dtype=jnp.int32)
    subject_ids = sim_h5["subject"].flatten()
    parameter_indices = _parameter_indices_for_subjects(subject_ids, parameters)
    rngs = random.split(rng, total_trials)

    # One jit-compiled vmap over trials
    typed_ids = vmap(simulator.simulate_trial, in_axes=(0, 0, None, 0))(
        trial_indices, parameter_indices, parameters, rngs
    )
    sim_h5["rec_itemids"] = typed_ids

    # Reindex study positions
    recalls_zero = vmap(
        vmap(item_to_study_positions, in_axes=(0, None, None)), in_axes=(0, 0, None)
    )(typed_ids, sim_h5["pres_itemids"], size)[:, :, 0]

    # mark −1 only where a letter was typed but not found in the list
    intrusion_mask = (typed_ids > 0) & (recalls_zero == 0)
    sim_h5["recalls"] = jnp.where(intrusion_mask, -1, recalls_zero)

    return sim_h5


def parameter_shifted_simulate_h5_from_h5(
    model_factory: Type[MemorySearchModelFactory],
    dataset: RecallDataset,
    features: Optional[Float[Array, " word_pool_items features_count"]],
    parameters: dict[str, Float[Array, " subject_count"]],
    trial_mask: Bool[Array, " trial_count"],
    experiment_count: int,
    varied_parameter: str,
    parameter_values: Sequence[float],
    rng: PRNGKeyArray,
    size: int = 3,
) -> Sequence[RecallDataset]:
    """
    Simulates multiple H5 datasets by systematically varying a specified parameter.

    Args:
        model_factory: Factory class for creating memory search model instances.
        dataset: Original H5 dataset containing trial data.
        features: Optional feature matrix describing word-pool items.
        parameters: Dictionary of simulation parameters, parameterized per subject.
        trial_mask: Boolean array specifying which trials to simulate.
        experiment_count: Number of simulation iterations per trial.
        varied_parameter: The parameter key to be varied across simulations.
        parameter_values: Sequence of values to assign to the varied parameter.
        rng: PRNGKeyArray for random number generation.
        size: Maximum number of study positions to return for each item during reindexing.

    Returns:
        A list of H5-like datasets (dictionaries), each corresponding to simulation results generated
        with a different value for the varied parameter.
    """

    results: list[RecallDataset] = []
    for value in parameter_values:
        rng, this_rng = random.split(rng)
        swept_params = {
            **parameters,
            varied_parameter: jnp.full_like(parameters[varied_parameter], value),
        }
        sim_data = simulate_h5_from_h5(
            model_factory,
            dataset,
            features,
            swept_params,
            trial_mask,
            experiment_count,
            this_rng,
            size,
        )
        results.append(sim_data)

    return results
