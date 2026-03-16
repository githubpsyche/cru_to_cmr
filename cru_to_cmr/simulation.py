"""Simulation utilities for memory-search models.

Provides functions to run forward simulations of memory-search models,
converting model parameters and study lists into simulated recall
sequences. Supports single-trial, multi-trial (vmapped), and
dataset-level simulation with optional study-phase replay.

"""

from typing import Mapping, Optional, Sequence, Type

import jax.numpy as jnp
import numpy as np
from jax import lax, random, vmap
from jax.tree_util import tree_map

from cru_to_cmr.typing import (
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
    TrialSimulator,
)


__all__ = [
    "item_to_study_positions",
    "simulate_free_recall",
    "simulate_study_and_free_recall",
    "MemorySearchSimulator",
    "preallocate_for_h5_dataset",
    "simulate_h5_from_h5",
]

def item_to_study_positions(
    item: Int_,
    presentation: Integer[Array, " list_length"],
    size: int,
) -> Integer[Array, " size"]:
    """Map an item to its one-indexed study positions.

    Parameters
    ----------
    item : Int_
        Item index (0 means no item).
    presentation : Integer[Array, " list_length"]
        Presentation sequence for a single trial.
    size : int
        Number of positions to return.

    Returns
    -------
    Integer[Array, " size"]
        One-indexed study positions, zero-padded.

    """
    return lax.cond(
        item == 0,
        lambda: jnp.zeros(size, dtype=int),
        lambda: jnp.nonzero(presentation == item, size=size, fill_value=-1)[0] + 1,
    )


def _reindex_recalls(
    recalls: Integer[Array, " trial_count recall_events"],
    presentations: Integer[Array, " trial_count list_length"],
    size: int,
) -> Integer[Array, " trial_count recall_events"]:
    """Reindex recall indices as study positions.

    Parameters
    ----------
    recalls : Integer[Array, " trial_count recall_events"]
        Simulated recall indices per trial.
    presentations : Integer[Array, " trial_count list_length"]
        Study sequences aligned with each trial.
    size : int
        Number of positions to retain.

    Returns
    -------
    Integer[Array, " trial_count recall_events"]
        Recalls expressed as study positions.

    """
    return vmap(
        vmap(item_to_study_positions, in_axes=(0, None, None)), in_axes=(0, 0, None)
    )(recalls, presentations, size)[:, :, 0]


def _single_free_recall(
    model: MemorySearch, rng: PRNGKeyArray
) -> tuple[MemorySearch, Integer[Array, ""]]:
    """Execute one free-recall event.

    Parameters
    ----------
    model : MemorySearch
        Retrieval-ready memory search model.
    rng : PRNGKeyArray
        Random key.

    Returns
    -------
    tuple[MemorySearch, Integer[Array, ""]]
        Updated model and chosen item index.

    """
    p_all = model.outcome_probabilities()
    choice = random.choice(rng, p_all.shape[0], p=p_all)
    return model.retrieve(choice), choice


def _maybe_free_recall(
    model: MemorySearch, rng: PRNGKeyArray
) -> tuple[MemorySearch, Integer[Array, ""]]:
    """Perform one recall step if active; no-op otherwise."""
    return lax.cond(
        model.is_active,
        _single_free_recall,
        lambda m, _: (m, 0),
        model,
        rng,
    )


def simulate_free_recall(
    model: MemorySearch, list_length: int, rng: PRNGKeyArray
) -> tuple[MemorySearch, Integer[Array, " recall_events"]]:
    """Simulate repeated free-recall steps from a model.

    Parameters
    ----------
    model : MemorySearch
        Retrieval-ready memory search model.
    list_length : int
        Upper bound on recall steps to simulate.
    rng : PRNGKeyArray
        Random key.

    Returns
    -------
    tuple[MemorySearch, Integer[Array, " recall_events"]]
        Updated model and recalled item indices.

    """
    return lax.scan(_maybe_free_recall, model, random.split(rng, list_length))


def simulate_study_and_free_recall(
    model: MemorySearch,
    present: Integer[Array, " study_events"],
    trial: Integer[Array, " recalls"],
    rng: PRNGKeyArray,
) -> tuple[MemorySearch, Integer[Array, " recall_events"]]:
    """Simulate study then free recall for a single trial.

    Parameters
    ----------
    model : MemorySearch
        Memory search model in study mode.
    present : Integer[Array, " study_events"]
        One-indexed study sequence for the trial.
    trial : Integer[Array, " recalls"]
        Observed recall sequence for the trial.
    rng : PRNGKeyArray
        Random key.

    Returns
    -------
    tuple[MemorySearch, Integer[Array, " recall_events"]]
        Updated model and recalled item indices.

    """
    model = lax.fori_loop(0, present.size, lambda i, m: m.experience(present[i]), model)
    model = model.start_retrieving()
    return simulate_free_recall(model, present.size, rng)


class MemorySearchSimulator:
    """Stateless trial-level simulator usable with vmap."""

    def __init__(
        self,
        model_factory: Type[MemorySearchModelFactory],
        dataset: RecallDataset,
        features: Optional[Float[Array, " word_pool_items features_count"]],
        simulate_trial_fn: TrialSimulator = simulate_study_and_free_recall,
    ) -> None:
        """Initialize simulator from a dataset and model factory.

        Parameters
        ----------
        model_factory : Type[MemorySearchModelFactory]
            Factory class for memory search models.
        dataset : RecallDataset
            Dataset with presentation and recall sequences.
        features : Optional[Float[Array, " ..."]]
            Feature matrix for the word pool, or None.
        simulate_trial_fn : TrialSimulator
            Trial simulation function.

        """
        factory = model_factory(dataset, features)
        self.create_model = factory.create_trial_model
        self.present_lists = jnp.array(dataset["pres_itemnos"])
        self._simulate_trial_fn = simulate_trial_fn

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

    def simulate_trial(
        self,
        trial_index: Integer[Array, ""],
        subject_index: Integer[Array, " subject_count"],
        parameters: Mapping[str, Float_],
        rng: PRNGKeyArray,
    ) -> Integer[Array, " recall_events"]:
        """Simulate a single trial using per-subject parameters.

        Parameters
        ----------
        trial_index : Integer[Array, ""]
            Index of the trial to simulate.
        subject_index : Integer[Array, " subject_count"]
            Index into per-subject parameter arrays.
        parameters : Mapping[str, Float_]
            Per-subject model parameters.
        rng : PRNGKeyArray
            Random key.

        Returns
        -------
        Integer[Array, " recall_events"]
            Recalled item indices for the trial.

        """
        model = self.create_model(
            trial_index, tree_map(lambda p: p[subject_index], parameters)
        )
        return self._simulate_trial_fn(
            model, self.present_lists[trial_index], self.trials[trial_index], rng
        )[1]


def _parameter_indices_for_subjects(
    subject_ids: Integer[Array, " trial_count"],
    parameters: Mapping[str, Float_],
) -> Integer[Array, " trial_count"]:
    """Map subject IDs to zero-based parameter indices.

    Parameters
    ----------
    subject_ids : Integer[Array, " trial_count"]
        Raw subject identifiers from the dataset.
    parameters : Mapping[str, Float_]
        Per-subject parameter arrays.

    Returns
    -------
    Integer[Array, " trial_count"]
        Zero-based parameter indices.

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
    """Replicate selected trials for batch simulation.

    Parameters
    ----------
    data : RecallDataset
        Input dataset.
    trial_mask : Bool[Array, " trial_count"]
        Mask selecting trials to include.
    experiment_count : int
        Replication count per trial.

    Returns
    -------
    RecallDataset
        Dataset with selected trials replicated.

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
    simulate_trial_fn: TrialSimulator = simulate_study_and_free_recall,
) -> RecallDataset:
    """Simulate a dataset using per-subject parameters.

    Parameters
    ----------
    model_factory : Type[MemorySearchModelFactory]
        Factory class for memory search models.
    dataset : RecallDataset
        Original dataset containing trial data.
    features : Optional[Float[Array, " ..."]]
        Feature matrix for the word pool, or None.
    parameters : dict[str, Float[Array, " subject_count"]]
        Per-subject simulation parameters.
    trial_mask : Bool[Array, " trial_count"]
        Mask selecting trials to simulate.
    experiment_count : int
        Replications per selected trial.
    rng : PRNGKeyArray
        Random key.
    size : int
        Max study positions per item when reindexing.
    simulate_trial_fn : TrialSimulator
        Trial-level simulation function.

    Returns
    -------
    RecallDataset
        Simulated dataset with recall sequences.

    """

    sim_h5 = preallocate_for_h5_dataset(dataset, trial_mask, experiment_count)
    simulator = MemorySearchSimulator(
        model_factory, sim_h5, features, simulate_trial_fn
    )

    # Flat trial + subject index vectors (static shapes)
    total_trials = sim_h5["subject"].size
    trial_indices = jnp.arange(total_trials, dtype=jnp.int32)
    subject_ids = sim_h5["subject"].flatten()
    parameter_indices = _parameter_indices_for_subjects(subject_ids, parameters)
    rngs = random.split(rng, total_trials)

    # One jit-compiled vmap over trials
    recalls = vmap(simulator.simulate_trial, in_axes=(0, 0, None, 0))(
        trial_indices, parameter_indices, parameters, rngs
    )

    # Reindex study positions
    sim_h5["recalls"] = _reindex_recalls(recalls, sim_h5["pres_itemnos"], size)
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
    simulate_trial_fn: TrialSimulator = simulate_study_and_free_recall,
) -> Sequence[RecallDataset]:
    """Simulate multiple datasets by sweeping a single parameter."""
    template = preallocate_for_h5_dataset(dataset, trial_mask, experiment_count)
    simulator = MemorySearchSimulator(
        model_factory, template, features, simulate_trial_fn
    )
    total_trials = template["subject"].size
    trial_indices = jnp.arange(total_trials, dtype=jnp.int32)
    subject_ids = template["subject"].flatten()
    template_without_recalls = {
        key: value for key, value in template.items() if key != "recalls"
    }

    def run_for_parameters(
        param_map: Mapping[str, Float_], sweep_rng: PRNGKeyArray
    ) -> RecallDataset:
        rngs = random.split(sweep_rng, total_trials)
        parameter_indices = _parameter_indices_for_subjects(subject_ids, param_map)
        recalls = vmap(simulator.simulate_trial, in_axes=(0, 0, None, 0))(
            trial_indices, parameter_indices, param_map, rngs
        )
        return template_without_recalls | {
            "recalls": _reindex_recalls(recalls, template["pres_itemnos"], size)
        }

    results: list[RecallDataset] = []
    for value in parameter_values:
        rng, this_rng = random.split(rng)
        swept_params = {
            **parameters,
            varied_parameter: jnp.full_like(parameters[varied_parameter], value),
        }
        results.append(run_for_parameters(swept_params, this_rng))

    return results
