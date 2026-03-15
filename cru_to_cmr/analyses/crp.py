"""Lag-Conditional Response Probability (Lag-CRP).

Provides tabulators and CRP functions for computing lag-conditional
response probability curves from free recall data. Supports both
unique-item lists (``SimpleTabulation``) and lists with repeated
items (``Tabulation``).

Notes
-----
**Conventions**

- list_length (L): fixed study-list length within a call.
- trials (recalls): int array ``[n_trials, n_recall_events]`` of
  serial positions in ``1..L``; ``0`` pads after termination.
- presentations: int array ``[n_trials, L]`` of item IDs at study
  positions; required only when items may repeat.
- size: upper bound on how many distinct study positions a single
  item can occupy within a list.
- Lag axis: all CRP outputs are length ``2*L - 1``. Index *i*
  corresponds to lag ``i - (L - 1)``; the center is lag 0.

**Design decisions**

- Zeros (pads) and out-of-range recalls are ignored via guards in
  the tabulators.
- Lags with zero availability yield NaN.
- A recall of a repeated item is treated as recalling all of that
  item's study positions. Actual and available lags are computed as
  unique lag set unions (boolean one-hot union per lag) to avoid
  multiplicity inflation.
- The first non-zero recall marks that item's study positions as
  unavailable and defines the reference for the next transition.

**Tabulation** (with repeats): tracks ``previous_positions``,
``avail_recalls`` (boolean ``[L]``), and ``actual_lags`` /
``avail_lags`` (int ``[2*L - 1]``). On each valid recall executes
four ordered steps: (1) update previous positions, (2) update
availability, (3) accumulate actual lags as the union of unique lags
from previous to current positions, (4) accumulate available lags
from each previous position to all currently available positions.
Zeros in position arrays are padding; indices are 1-based for
positions, 0-based internally.

**JAX compilation**: all public functions are side-effect-free and
JIT-safe. Use ``jit(crp, static_argnames=("size",))``; keep shapes
(L, number of recall events) consistent within a compiled call to
avoid recompiles. Ensure ``size`` >= the true maximum per-item
repetition; compute it once outside JIT if needed.

"""

__all__ = [
    "set_false_at_index",
    "Tabulation",
    "tabulate_trial",
    "crp",
    "plot_crp",
]

from typing import Optional, Sequence

from jax import jit, lax, vmap
from jax import numpy as jnp
from matplotlib.axes import Axes
from simple_pytree import Pytree

from ..helpers import apply_by_subject
from ..plotting import plot_data, prepare_plot_inputs, set_plot_labels
from ..repetition import all_study_positions
from ..typing import Array, Bool, Float, Int_, Integer, RecallDataset


def set_false_at_index(
    vec: Bool[Array, " positions"], i: Int_
) -> tuple[Bool[Array, " positions"], None]:
    """Set ``vec[i - 1]`` to ``False`` using 1-based indexing.

    Parameters
    ----------
    vec : Bool[Array, " positions"]
        Boolean vector.
    i : Int_
        1-based index; 0 is a no-op sentinel.

    Returns
    -------
    tuple[Bool[Array, " positions"], None]
        Updated vector and ``None``.

    """

    should_update = (i > 0) & (i <= vec.size)
    return lax.cond(
        should_update, lambda: (vec.at[i - 1].set(False), None), lambda: (vec, None)
    )


class Tabulation(Pytree):
    """CRP tabulator supporting repeated items."""

    def __init__(
        self,
        presentation: Integer[Array, " study_events"],
        first_recall: Int_,
        size: int = 3,
    ):
        self.list_length = presentation.size
        self.lag_range = self.list_length - 1
        self.all_positions = jnp.arange(1, self.list_length + 1, dtype=int)
        self.base_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.size = size
        self.item_study_positions = lax.map(
            lambda i: all_study_positions(i, presentation, size),
            self.all_positions,
        )

        self.actual_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)
        self.avail_lags = jnp.zeros(self.lag_range * 2 + 1, dtype=int)

        self.previous_positions = self.item_study_positions[first_recall - 1]
        self.avail_recalls = jnp.ones(self.list_length, dtype=bool)
        self.avail_recalls = self.available_recalls_after(first_recall)

    # for updating avail_recalls: study positions still available for retrieval
    def available_recalls_after(self, recall: Int_) -> Bool[Array, " positions"]:
        """Clear availability for all study positions of a recall."""
        study_positions = self.item_study_positions[recall - 1]
        return lax.scan(set_false_at_index, self.avail_recalls, study_positions)[0]

    # for updating actual_lags: lag-transitions actually made from the previous item
    def lags_from_previous(self, recall_pos: Int_) -> Bool[Array, " positions"]:
        """Unique lags from previous positions to a recall position."""

        def f(prev):
            return lax.cond(
                (recall_pos * prev) == 0,
                lambda: self.base_lags,
                lambda: self.base_lags.at[recall_pos - prev + self.lag_range].add(1),
            )

        return lax.map(f, self.previous_positions).sum(0).astype(bool)

    def tabulate_actual_lags(self, recall: Int_) -> Integer[Array, " lags"]:
        "Tabulates the actual transition after a transition."
        recall_study_positions = self.item_study_positions[recall - 1]
        new_lags = (
            lax.map(self.lags_from_previous, recall_study_positions).sum(0).astype(bool)
        )
        return self.actual_lags + new_lags

    # for updating avail_lags: lag-transitions available from the previous item
    def available_lags_from(self, pos: Int_) -> Bool[Array, " lags"]:
        "Identifies recallable lag transitions from the specified study position."
        return lax.cond(
            pos == 0,
            lambda: self.base_lags,
            lambda: self.base_lags.at[self.all_positions - pos + self.lag_range].add(
                self.avail_recalls
            ),
        )

    def tabulate_available_lags(self) -> Integer[Array, " lags"]:
        "Union of lags from each previous position to all currently available positions."
        new_lags = (
            lax.map(self.available_lags_from, self.previous_positions)
            .sum(0)
            .astype(bool)
        )
        return self.avail_lags + new_lags

    # unifying tabulation of actual/avail lags, previous positions, and avail recalls
    def should_tabulate(self, recall: Int_) -> Bool:
        "Only consider transitions from items with study positions that have not been recalled yet."

        def _for_nonzero():
            recall_study_positions = self.item_study_positions[recall - 1]
            is_valid_study_position = recall_study_positions != 0
            is_available_study_position = self.avail_recalls[recall_study_positions - 1]
            return jnp.any(is_valid_study_position & is_available_study_position)

        return lax.cond(
            recall == 0,
            lambda: jnp.array(False),
            _for_nonzero,
        )

    def tabulate(self, recall: Int_) -> "Tabulation":
        "Tabulates actual and possible serial lags of current from previous item."
        return lax.cond(
            self.should_tabulate(recall),
            lambda: self.replace(
                previous_positions=self.item_study_positions[recall - 1],
                avail_recalls=self.available_recalls_after(recall),
                actual_lags=self.tabulate_actual_lags(recall),
                avail_lags=self.tabulate_available_lags(),
            ),
            lambda: self,
        )


def tabulate_trial(
    trial: Integer[Array, " recall_events"],
    presentation: Integer[Array, " study_events"],
    size: int = 3,
) -> tuple[Float[Array, " lags"], Float[Array, " lags"]]:
    """Tabulate actual and available lags for a single trial.

    Parameters
    ----------
    trial : Integer[Array, " recall_events"]
        Serial positions in 1..L with 0 pads.
    presentation : Integer[Array, " study_events"]
        Item IDs at each study position.
    size : int
        Max study positions per item.

    Returns
    -------
    tuple[Float[Array, " lags"], Float[Array, " lags"]]
        Actual and available lag counts.

    """

    init = Tabulation(presentation, trial[0], size)
    tab = lax.fori_loop(1, trial.size, lambda i, t: t.tabulate(trial[i]), init)
    return tab.actual_lags, tab.avail_lags


def crp(
    dataset: RecallDataset,
    size: int = 3,
) -> Float[Array, " lags"]:
    """Compute Lag-CRP with support for repeated items.

    Parameters
    ----------
    dataset : RecallDataset
        Recall dataset with ``recalls`` and ``pres_itemnos``.
    size : int
        Max study positions an item can occupy.

    Returns
    -------
    Float[Array, " lags"]
        CRP of length 2*L - 1; NaN where denominator is
        zero.

    """
    actual, possible = vmap(tabulate_trial, in_axes=(0, 0, None))(
        dataset["recalls"], dataset["pres_itemnos"], size
    )
    return actual.sum(0) / possible.sum(0)


def plot_crp(
    datasets: Sequence[RecallDataset] | RecallDataset,
    trial_masks: Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"],
    max_lag: int = 5,
    color_cycle: Optional[list[str]] = None,
    labels: Optional[Sequence[str]] = None,
    contrast_name: Optional[str] = None,
    axis: Optional[Axes] = None,
    size: int = 3,
    confidence_level: float = 0.95,
) -> Axes:
    """Plot subject-wise Lag-CRP with confidence intervals.

    Parameters
    ----------
    datasets : Sequence[RecallDataset] | RecallDataset
        One or more datasets to plot.
    trial_masks : Sequence[Bool[Array, " trial_count"]] | Bool[Array, " trial_count"]
        Boolean mask(s) selecting trials.
    max_lag : int
        Maximum lag to display.
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
        Matplotlib Axes with the Lag-CRP plot.

    """
    axis, datasets, trial_masks, color_cycle = prepare_plot_inputs(
        datasets, trial_masks, color_cycle, axis
    )
    if labels is None:
        labels = [""] * len(datasets)

    lag_interval = jnp.arange(-max_lag, max_lag + 1, dtype=int)

    for data_index, data in enumerate(datasets):
        lag_range = (jnp.max(data["listLength"]) - 1).item()
        subject_values = apply_by_subject(
            data,
            trial_masks[data_index],
            jit(crp, static_argnames=("size")),
            size=size,
        )
        subject_values = jnp.vstack(subject_values)
        subject_values = subject_values[
            :, lag_range - max_lag : lag_range + max_lag + 1
        ]
        color = color_cycle[data_index % len(color_cycle)]
        plot_data(
            axis,
            lag_interval,
            subject_values,
            labels[data_index],
            color,
            confidence_level=confidence_level,
        )

    set_plot_labels(axis, "Lag", "Conditional Resp. Prob.", contrast_name)
    return axis
