"""Termination policies for memory search models.

Provides functions that compute the probability of terminating
recall at each retrieval step, including fixed-rate and
activation-dependent termination rules.

"""

from __future__ import annotations

from typing import Mapping

from jax import lax
from jax import numpy as jnp

from cru_to_cmr.math import lb
from simple_pytree import Pytree
from cru_to_cmr.typing import Array, Float, Float_, MemorySearch

__all__ = [
    "PositionalTermination",
]


class PositionalTermination(Pytree):
    """Termination probability is an exponential function of recall position."""

    def __init__(
        self,
        list_length: int,
        parameters: Mapping[str, Float_],
    ) -> None:
        scale = parameters["stop_probability_scale"]
        growth = parameters["stop_probability_growth"]
        self._stop_probability = scale * jnp.exp(jnp.arange(list_length) * growth)

    def stop_probability(self, model: MemorySearch) -> Float[Array, ""]:
        total_recallable = jnp.sum(model.recallable)
        return lax.cond(
            jnp.logical_or(total_recallable == 0, ~model.is_active),
            lambda: jnp.array(1.0),
            lambda: jnp.minimum(
                1.0 - (lb * total_recallable),
                self._stop_probability[model.recall_total],
            ),
        )
