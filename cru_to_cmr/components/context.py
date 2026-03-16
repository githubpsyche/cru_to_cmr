"""Temporal context utilities for memory search models.

Implements the ``TemporalContext`` state object that maintains and
updates the drifting temporal context vector central to CMR-family
models. Supports standard integration, expanded representations,
and outlist integration for repeated items.

"""

__all__ = ["TemporalContext"]

import base64
import io

import matplotlib.pyplot as plt
from jax import numpy as jnp
from simple_pytree import Pytree

from ..math import normalize_magnitude
from ..state_analysis import matrix_heatmap
from ..typing import Array, Float, Float_


class TemporalContext(Pytree):
    """Drifting, unit-length context representation.

    The vector starts with a *start-of-list* unit (index ``0``) set to ``1`` and
    one unit per study item initialised to ``0``. On every call to
    :meth:`integrate`, the context drifts toward a normalised input vector while
    remaining unit-length. This initial state is preserved to enable drift back
    to the start-of-list context unit.

    An optional out-of-list context unit (index ``item_count + 1``) can be used
    to simulate post-study drift, but unless the drift rate is near ``1``, it
    does not affect behavior because CMR relies on relative differences between
    context units.
    """

    def __init__(self, item_count: int, size: int) -> None:
        """Create a new temporal context model.

        Args:
            item_count: Number of items in the context model.
            size: Size of the context representation.
        """
        self.size = size
        self.zeros = jnp.zeros(size)
        self.state = self.zeros.at[0].set(1)
        self.initial_state = self.zeros.at[0].set(1)
        self.next_outlist_unit = item_count + 1

    def integrate(
        self,
        context_input: Float[Array, " context_feature_units"],
        drift_rate: Float_,
    ) -> "TemporalContext":
        """Returns context after integrating input while preserving unit length.

        Args:
            context_input: Input representation to integrate.
            drift_rate: Drift rate parameter.
        """
        context_input = normalize_magnitude(context_input)
        rho = jnp.sqrt(
            1 + jnp.square(drift_rate) * (jnp.square(self.state * context_input) - 1)
        ) - (drift_rate * (self.state * context_input))
        return self.replace(
            state=normalize_magnitude((rho * self.state) + (drift_rate * context_input))
        )

    def _repr_markdown_(self) -> str:
        """Returns a markdown ``img`` tag of the current state."""
        fig, ax = matrix_heatmap(self.state, figsize=(6, 0.6))

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_yticks([])

        # Remove colorbar safely if desired:
        for coll in ax.collections:
            if hasattr(coll, "colorbar") and coll.colorbar:
                coll.colorbar.remove()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)

        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f'<img src="data:image/png;base64,{encoded}" />'

    @property
    def outlist_input(self) -> Float[Array, " context_feature_units"]:
        """Returns the out-of-list context input."""
        return self.zeros.at[self.next_outlist_unit].set(1)

    def integrate_with_outlist(
        self,
        inlist_input: Float[Array, " context_feature_units"],
        ratio: Float_,
        drift_rate: Float_,
    ) -> "TemporalContext":
        """Integrate in-list input mixed with out-of-list context.

        Will produce errors if no out-of-list context units are available.

        Args:
            inlist_input: Input representation to integrate.
            ratio: Ratio of out-of-list to in-list context.
            drift_rate: Drift rate parameter.
        """
        context_input = normalize_magnitude(
            (normalize_magnitude(inlist_input) * ratio)
            + (self.outlist_input * (1 - ratio))
        )
        rho = jnp.sqrt(
            1 + jnp.square(drift_rate) * (jnp.square(self.state * context_input) - 1)
        ) - (drift_rate * (self.state * context_input))
        return self.replace(
            state=normalize_magnitude(
                (rho * self.state) + (drift_rate * context_input)
            ),
            next_outlist_unit=self.next_outlist_unit + 1,
        )

    @classmethod
    def init_expanded(cls, item_count: int) -> "TemporalContext":
        """Returns context with additional out-of-list capacity.

        Args:
            item_count: Number of items in the context model.
        """
        return cls(item_count, item_count + item_count + 1)


def init(item_count: int) -> "TemporalContext":
    """Returns context sized for ``item_count`` items.

    Args:
        item_count: Number of items in the context model.
    """
    return TemporalContext(item_count, item_count + 1)
