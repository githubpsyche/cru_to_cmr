"""Utilities for handling item repetitions in study lists.

Provides helpers to map recalled items to all of their valid study
positions.

"""

from jax import lax
from jax import numpy as jnp

from .typing import Array, Int_, Integer


__all__ = [
    "item_to_study_positions",
    "all_study_positions",
]

def item_to_study_positions(
    item: Int_,
    presentation: Integer[Array, " list_length"],
    size: int,
) -> Integer[Array, " size"]:
    """Returns one-indexed positions where ``item`` appears in ``presentation``.

    Args:
      item: Item identifier; ``0`` is treated as no item.
      presentation: 1D sequence of item identifiers for the study list. Shape [list_length].
      size: Maximum number of positions to return; pads with 0 when fewer.
    """
    return lax.cond(
        item == 0,
        lambda: jnp.zeros(size, dtype=int),
        lambda: jnp.nonzero(presentation == item, size=size, fill_value=-1)[0] + 1,
    )


def all_study_positions(
    study_position: Int_,
    presentation: Integer[Array, " list_length"],
    size: int,
) -> Integer[Array, " size"]:
    """Returns study positions of the item shown at ``study_position``.

    Args:
      study_position: One-indexed study position; ``<=0`` returns all zeros.
      presentation: 1D sequence of item identifiers for the study list. Shape [list_length].
      size: Maximum number of positions to return; pads with 0 when fewer.
    """
    item = lax.cond(
        study_position > 0,
        lambda: presentation[study_position - 1],
        lambda: 0,
    )
    return item_to_study_positions(item, presentation, size)
