"""Linear associative memory operations for CMR.

Implements the ``LinearMemory`` state object that stores and
retrieves item-context associations via outer-product learning and
matrix-vector retrieval. Provides both MFC (memory-to-context) and
MCF (memory-to-feature/item) association matrices.

"""

from typing import Mapping

from jax import lax
from jax import numpy as jnp
from simple_pytree import Pytree

from cru_to_cmr.typing import Array, Context, Float, Float_, Int_

__all__ = ["LinearMemory"]


class LinearMemory(Pytree):
    """Associative memory storing linear item–context links."""

    def __init__(
        self,
        state: Float[Array, " input_size output_size"],
    ):
        """Initialize memory with an association matrix.

        Args:
            state: Association weights. Shape (input_size, output_size).
        """
        self.state = state
        self.input_size = self.state.shape[0]
        self.output_size = self.state.shape[1]

    def associate(
        self,
        in_pattern: Float[Array, " input_size"],
        out_pattern: Float[Array, " output_size"],
        learning_rate: Float_,
    ) -> "LinearMemory":
        """Returns memory after associating input and output patterns.

        Args:
            in_pattern: Input feature pattern.
            out_pattern: Output feature pattern.
            learning_rate: Scaling factor for the association.
        """
        return self.replace(
            state=self.state + (learning_rate * jnp.outer(in_pattern, out_pattern))
        )

    def probe(
        self,
        in_pattern: Float[Array, " input_size"],
    ) -> Float[Array, " output_size"]:
        """Returns output pattern associated with the input pattern.

        Args:
            in_pattern: Feature pattern to query.
        """
        return jnp.dot(in_pattern, self.state)

    def zero_out(
        self,
        index: Int_,
    ) -> "LinearMemory":
        """Returns memory with associations for an input index cleared.

        Args:
            index: Input index to clear.
        """
        return self.replace(state=self.state.at[index].set(0))

def init_mfc(
    list_length: int,
    parameters: Mapping[str, Float_],
    context: Context,
) -> "LinearMemory":
    """Returns linear item-to-context memory model.

    Initially, each item links to a unique context unit with weight ``1 - learning_rate``.
    To allow out-of-list contexts, set the context builder to include an additional unit.

    Args:
        list_length: Number of items.
        parameters: Model parameter mapping.
        context: Context instance providing feature dimension.
    """
    context_feature_count = context.size
    learning_rate = parameters["learning_rate"]
    item_feature_count = list_length
    return LinearMemory(
        jnp.eye(item_feature_count, context_feature_count, 1) * (1 - learning_rate),
    )

def init_mcf(
    list_length: int,
    parameters: Mapping[str, Float_],
    context: Context,
) -> "LinearMemory":
    """Returns linear context-to-item memory model.

    In-list context units associate with all items via ``shared_support`` and
    receive additional ``item_support`` for their corresponding item. Start-of-list
    and out-of-list units begin with zero associations.

    Args:
        list_length: Number of items.
        parameters: Model parameter mapping.
        context: Context instance providing feature dimension.
    """
    item_support = parameters["item_support"]
    shared_support = parameters["shared_support"]

    item_count = list_length
    context_feature_count = context.size

    base_memory = jnp.full((context_feature_count - 1, item_count), shared_support)
    base_memory = lax.fori_loop(
        0, item_count, lambda i, m: m.at[i, i].set(item_support), base_memory
    )
    start_list = jnp.zeros((1, item_count))
    return LinearMemory(jnp.vstack((start_list, base_memory)))
