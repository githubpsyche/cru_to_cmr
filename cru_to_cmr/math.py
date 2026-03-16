"""Numerical primitives for memory-search models.

Provides power-scaling, normalization, primacy decay, and cosine
similarity functions used by model components and analyses.

"""

import jax.numpy as jnp
from jax import lax

from cru_to_cmr.typing import Array, Float, Float_, Int_

__all__ = [
    "lb",
    "power_scale",
    "exponential_primacy_decay",
    "normalize_magnitude",
]

lb = jnp.finfo(jnp.float32).eps


def power_scale(value: Float_, scale: Float_) -> Float:
    """Returns a log-stabilized power that preserves ordering, not magnitude."""
    log_activation = jnp.log(value)
    return lax.cond(
        jnp.logical_and(jnp.any(value != 0), scale != 1),
        lambda _: jnp.exp(scale * (log_activation - jnp.max(log_activation))),
        lambda _: value,
        None,
    )

def exponential_primacy_decay(
    study_index: Int_, primacy_scale: Float_, primacy_decay: Float_
):
    """Returns the exponential primacy weighting for the specified study event.

    Args:
        study_index: the index of the study event.
        primacy_scale: the scale factor for primacy effect.
        primacy_decay: the decay factor for primacy effect.
    """
    return primacy_scale * jnp.exp(-primacy_decay * study_index) + 1


def normalize_magnitude(
    vector: Float[Array, " features"],
) -> Float[Array, " features"]:
    """Return the input vector normalized to unit length."""
    return vector / jnp.sqrt(jnp.sum(vector**2) + lb)
