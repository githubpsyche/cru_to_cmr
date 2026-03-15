"""Omnibus CRU-CMR hybrid model.

Implements a model that combines CRU (Context Retrieval and
Updating) and CMR retrieval mechanisms, allowing parameter-level
interpolation between the two architectures.

"""

from typing import Mapping, Optional

import numpy as np
from jax import lax, vmap
from jax import numpy as jnp
from simple_pytree import Pytree

from cru_to_cmr.math import (
    exponential_primacy_decay,
    exponential_stop_probability,
    lb,
    power_scale,
)
from cru_to_cmr.components.context import TemporalContext
from cru_to_cmr.components.linear_memory import LinearMemory
from cru_to_cmr.typing import (
    Array,
    Float,
    Float_,
    Int_,
    Integer,
    MemorySearch,
    RecallDataset,
)


__all__ = [
    "CMR",
    "BaseCMR",
    "BaseCMRFactory",
    "letter_similarities",
]

class CMR(Pytree):
    """The Context Maintenance and Retrieval (CMR) model of memory search."""

    def __init__(
        self,
        list_length: int,
        parameters: Mapping[str, Float_],
        distances: Integer[Array, " word_pool_items word_pool_items"],
    ):
        self.encoding_drift_rate_max = parameters["encoding_drift_rate"]
        self.start_drift_rate = parameters["start_drift_rate"]
        self.recall_drift_rate = parameters["recall_drift_rate"]
        self.shared_support = parameters["shared_support"]
        self.item_support = parameters["item_support"]
        self.primacy_scale = parameters["primacy_scale"]
        self.primacy_decay = parameters["primacy_decay"]
        self.mfc_learning_rate = parameters["learning_rate"]
        self.mcf_sensitivity = parameters["choice_sensitivity"]
        self.allow_repeated_recalls = parameters.get("allow_repeated_recalls", False)

        # specific to item-independent stop rule
        self.stop_probability_scale = parameters["stop_probability_scale"]
        self.stop_probability_growth = parameters["stop_probability_growth"]

        # specific to cru
        self.item_sensitivity_max = parameters["item_sensitivity_max"]
        self.item_sensitivity_decrease = parameters["item_sensitivity_decrease"]
        self.encoding_drift_decrease = parameters["encoding_drift_decrease"]

        LETTER_COUNT = 26
        self.item_count = LETTER_COUNT
        self.item_arange = jnp.arange(self.item_count+1)
        self.items = jnp.eye(self.item_count)
        self._stop_probability = exponential_stop_probability(
            self.stop_probability_scale,
            self.stop_probability_growth,
            jnp.arange(list_length),
        )
        self._mcf_learning_rate = exponential_primacy_decay(
            jnp.arange(list_length), self.primacy_scale, self.primacy_decay
        )
        self.context = TemporalContext.init(self.item_count)
        self.mfc = LinearMemory.init_mfc(
            self.item_count,
            self.context.size,
            parameters["learning_rate"],
        )
        self.mcf = LinearMemory.init_mcf(
            self.item_count,
            self.context.size,
            parameters["item_support"],
            parameters["shared_support"],
        )
        self.distances = distances
        self.slip_matrix = jnp.eye(self.item_count)
        self.recalls = jnp.zeros(self.item_count, dtype=int)
        self.recallable = jnp.ones(self.item_count, dtype=bool)
        self.is_active = jnp.array(True)
        self.recall_total = jnp.array(0, dtype=int)
        self.study_index = jnp.array(0, dtype=int)

    @property
    def mcf_learning_rate(self) -> Float[Array, ""]:
        """The learning rate for the MCF memory under its current state."""
        return self._mcf_learning_rate[self.study_index]

    @property
    def encoding_drift_rate(self) -> Float[Array, ""]:
        """The contextual drift rate while encoding items."""
        return self.encoding_drift_rate_max * (
            self.encoding_drift_decrease**self.study_index
        )

    def _update_slip_row(
        self, letter_idx: Int_
    ) -> Integer[Array, " word_pool_items word_pool_items"]:
        #! we update slip matrix based on encoded item and item sensitivity
        item_sensitivity = self.item_sensitivity_max * (
            self.item_sensitivity_decrease**self.study_index
        )
        letter_distances = self.distances[letter_idx]
        z_row = jnp.exp(-item_sensitivity * letter_distances)
        z_row = z_row / z_row.sum()
        return self.slip_matrix.at[letter_idx, :].set(z_row)

    def experience_item(self, item_index: Int_) -> "CMR":
        """Return the model after experiencing item with the specified index.

        Args:
            item_index: the index of the item to experience. 0-indexed.
        """
        item = self.items[item_index]

        context_input = self.mfc.probe(item)
        new_context = self.context.integrate(context_input, self.encoding_drift_rate)

        #! We associate with current context state instead of new_context in this implementation
        return self.replace(
            context=new_context,
            mfc=self.mfc.associate(
                item, self.context.state, self.mfc_learning_rate
            ),  #! updated
            mcf=self.mcf.associate(
                self.context.state, item, self.mcf_learning_rate
            ),  #! updated
            study_index=self.study_index + 1,
            slip_matrix=self._update_slip_row(item_index),
        )

    def experience(self, choice: Int_) -> "CMR":
        """Returns model after simulating the specified study event.

        Args:
            choice: the index of the item to experience (1-indexed). 0 is ignored.
        """
        return lax.cond(
            choice == 0,
            lambda: self,
            lambda: self.experience_item(choice - 1),
        )

    def start_retrieving(self) -> "CMR":
        """Returns model after transitioning from study to retrieval mode."""
        start_input = self.context.initial_state
        start_context = self.context.integrate(start_input, self.start_drift_rate)
        return self.replace(context=start_context)

    def retrieve_item(self, item_index: Int_) -> "CMR":
        """Return model after simulating retrieval of item with the specified index.

        Args:
            choice: the index of the item to retrieve (0-indexed)
        """
        item_activations = self.activations()
        p_mem = item_activations / jnp.sum(item_activations)
        p_slip = self.slip_matrix[:, item_index]
        joint = p_mem * p_slip
        best_i = jnp.argmax(joint)

        new_context = self.context.integrate(
            self.mfc.probe(self.items[best_i]),
            self.recall_drift_rate,
        )
        return self.replace(
            context=new_context,
            recalls=self.recalls.at[self.recall_total].set(best_i + 1),
            recallable=self.recallable.at[best_i].set(self.allow_repeated_recalls),
            recall_total=self.recall_total + 1,
        )

    def retrieve(self, choice: Int_) -> "CMR":
        """Return model after simulating the specified retrieval event.

        Args:
            choice: the index of the item to retrieve (1-indexed) or 0 to stop.
        """
        return lax.cond(
            choice == 0,
            lambda: self.replace(is_active=False),
            lambda: self.retrieve_item(choice - 1),
        )

    def activations(self) -> Float[Array, " item_count"]:
        """Returns relative support for retrieval of each item given model state"""
        _activations = self.mcf.probe(self.context.state) * self.recallable
        return (power_scale(_activations, self.mcf_sensitivity) + lb) * self.recallable

    def stop_probability(self) -> Float[Array, ""]:
        """Returns probability of stopping retrieval given model state"""
        total_recallable = jnp.sum(self.recallable)
        return lax.cond(
            jnp.logical_or(total_recallable == 0, ~self.is_active),
            true_fun=lambda: 1.0,
            false_fun=lambda: jnp.minimum(
                1.0 - (lb * total_recallable),
                self._stop_probability[self.recall_total],
            ),
        )

    def item_probability(self, item_index: Int_) -> Float[Array, ""]:
        """Return the probability of retrieval of an item at the specified index.

        Assumes that some items are recallable, with at least the minimum recall probability.

        Args:
            item_index: the index of the item to retrieve.
        """
        item_activations = self.activations()
        p_mem = item_activations / jnp.sum(item_activations)
        p_slip = self.slip_matrix[:, item_index]
        return jnp.vdot(p_mem, p_slip)

    def outcome_probability(self, choice: Int_) -> Float[Array, ""]:
        """Return probability of the specified retrieval event.

        Args:
            choice: the index of the item to retrieve (1-indexed) or 0 to stop.
        """
        p_stop = self.stop_probability()
        return lax.cond(
            choice == 0,
            lambda: p_stop,
            lambda: lax.cond(
                jnp.logical_or(p_stop == 1.0, ~self.recallable[choice - 1]),
                lambda: 0.0,
                lambda: (1 - p_stop) * self.item_probability(choice - 1),
            ),
        )

    def outcome_probabilities(self) -> Float[Array, " recall_outcomes"]:
        """Return the outcome probabilities of all recall events."""
        return vmap(self.outcome_probability)(self.item_arange)


def BaseCMR(
    list_length: int,
    parameters: Mapping[str, Float_],
    distances: Integer[Array, " word_pool_items word_pool_items"],
) -> CMR:
    """Creates a regular CMR model with linear associative $M^{FC}$ and $M^{CF}$ memories."""
    return CMR(list_length, parameters, distances)


class BaseCMRFactory:
    def __init__(
        self,
        dataset: RecallDataset,
        features: Optional[Float[Array, " word_pool_items features_count"]],
    ) -> None:
        """Initialize the factory with the specified trials and trial data."""
        self.max_list_length = np.max(dataset["listLength"]).item()
        self.distances = 1 / (letter_similarities + lb)

    def create_model(
        self,
        parameters: Mapping[str, Float_],
    ) -> MemorySearch:
        """Create a new memory search model with the specified parameters."""
        return BaseCMR(self.max_list_length, parameters, self.distances)

    def create_trial_model(
        self,
        trial_index: Int_,
        parameters: Mapping[str, Float_],
    ) -> MemorySearch:
        """Create a new memory search model with the specified parameters for the specified trial."""
        return BaseCMR(self.max_list_length, parameters, self.distances)


letter_similarities = jnp.array(
    [
        [
            1.0,
            0.48225309,
            0.49236829,
            0.48856752,
            0.50548451,
            0.48790008,
            0.45743562,
            0.47693995,
            0.49363215,
            0.48548403,
            0.4743608,
            0.50140393,
            0.49436425,
            0.49169043,
            0.49370526,
            0.48344211,
            0.47901897,
            0.48971596,
            0.50431187,
            0.49835543,
            0.48058439,
            0.48780488,
            0.50175615,
            0.47917964,
            0.45454545,
            0.48716325,
        ],
        [
            0.48225309,
            1.0,
            0.48313847,
            0.56731151,
            0.48083858,
            0.48959608,
            0.47103156,
            0.50456633,
            0.48574343,
            0.4893565,
            0.48527199,
            0.50459179,
            0.49701789,
            0.49166626,
            0.50289163,
            0.52592826,
            0.49024414,
            0.49509852,
            0.47140904,
            0.48969198,
            0.49156958,
            0.48732943,
            0.51395385,
            0.47585058,
            0.47236656,
            0.4799616,
        ],
        [
            0.49236829,
            0.48313847,
            1.0,
            0.48744821,
            0.50238634,
            0.48875855,
            0.45154881,
            0.47952431,
            0.49195651,
            0.48330192,
            0.45446282,
            0.50115265,
            0.48742445,
            0.47968533,
            0.50017506,
            0.47899602,
            0.48574343,
            0.50673964,
            0.4865233,
            0.49421765,
            0.50248731,
            0.47413589,
            0.50052555,
            0.47856049,
            0.46036277,
            0.48311513,
        ],
        [
            0.48856752,
            0.56731151,
            0.48744821,
            1.0,
            0.4803074,
            0.51767873,
            0.49534377,
            0.49229557,
            0.49487801,
            0.50522912,
            0.49072529,
            0.50932057,
            0.49987503,
            0.49468217,
            0.52037259,
            0.49763623,
            0.51132587,
            0.4810699,
            0.46598322,
            0.49019608,
            0.49190811,
            0.4725228,
            0.48875855,
            0.48576703,
            0.47330557,
            0.47671259,
        ],
        [
            0.50548451,
            0.48083858,
            0.50238634,
            0.4803074,
            1.0,
            0.48709206,
            0.45884188,
            0.47463097,
            0.49324258,
            0.50769153,
            0.47114252,
            0.51469453,
            0.50632911,
            0.47094283,
            0.49485352,
            0.51200655,
            0.46589638,
            0.48383975,
            0.49682035,
            0.49610557,
            0.48909322,
            0.48028433,
            0.50223495,
            0.47612246,
            0.48144047,
            0.48074612,
        ],
        [
            0.48790008,
            0.48959608,
            0.48875855,
            0.51767873,
            0.48709206,
            1.0,
            0.49024414,
            0.50965802,
            0.52938062,
            0.50484653,
            0.49333991,
            0.54722557,
            0.48973995,
            0.48463701,
            0.51808103,
            0.498132,
            0.4877335,
            0.50183169,
            0.47920261,
            0.54513737,
            0.50597045,
            0.52712034,
            0.48861526,
            0.48290516,
            0.48484848,
            0.49689441,
        ],
        [
            0.45743562,
            0.47103156,
            0.45154881,
            0.49534377,
            0.45884188,
            0.49024414,
            1.0,
            0.45991813,
            0.50960607,
            0.47625851,
            0.47196526,
            0.4995504,
            0.45825314,
            0.45562238,
            0.47049967,
            0.52058931,
            0.55328096,
            0.47492401,
            0.4355211,
            0.48612124,
            0.45454545,
            0.46242775,
            0.47449585,
            0.45055193,
            0.47427081,
            0.44253662,
        ],
        [
            0.47693995,
            0.50456633,
            0.47952431,
            0.49229557,
            0.47463097,
            0.50965802,
            0.45991813,
            1.0,
            0.49041244,
            0.49783442,
            0.5236973,
            0.52042675,
            0.50263885,
            0.53576212,
            0.47125353,
            0.46818671,
            0.49473111,
            0.49031625,
            0.47056609,
            0.49887753,
            0.50574015,
            0.4685816,
            0.49842995,
            0.49514755,
            0.47218812,
            0.48123195,
        ],
        [
            0.49363215,
            0.48574343,
            0.49195651,
            0.49487801,
            0.49324258,
            0.52938062,
            0.50960607,
            0.49041244,
            1.0,
            0.54472165,
            0.51056877,
            0.56599502,
            0.50030018,
            0.50456633,
            0.51623561,
            0.49002793,
            0.50684237,
            0.51730381,
            0.5009769,
            0.52413649,
            0.48360576,
            0.49603175,
            0.49402233,
            0.50510153,
            0.49960032,
            0.49771053,
        ],
        [
            0.48548403,
            0.4893565,
            0.48330192,
            0.50522912,
            0.50769153,
            0.50484653,
            0.47625851,
            0.49783442,
            0.54472165,
            1.0,
            0.50766575,
            0.51387461,
            0.4850131,
            0.49531923,
            0.51200655,
            0.49635181,
            0.51392743,
            0.49907671,
            0.49947555,
            0.51064699,
            0.4929508,
            0.49507401,
            0.49972515,
            0.51666236,
            0.50489751,
            0.4879715,
        ],
        [
            0.4743608,
            0.48527199,
            0.45446282,
            0.49072529,
            0.47114252,
            0.49333991,
            0.47196526,
            0.5236973,
            0.51056877,
            0.50766575,
            1.0,
            0.5262327,
            0.49630255,
            0.47744092,
            0.466157,
            0.49662296,
            0.49091802,
            0.49608096,
            0.48654698,
            0.52312199,
            0.49932591,
            0.50316997,
            0.51599587,
            0.51347882,
            0.48816207,
            0.48482498,
        ],
        [
            0.50140393,
            0.50459179,
            0.50115265,
            0.50932057,
            0.51469453,
            0.54722557,
            0.4995504,
            0.52042675,
            0.56599502,
            0.51387461,
            0.5262327,
            1.0,
            0.49142464,
            0.50122801,
            0.53183003,
            0.5107774,
            0.48962005,
            0.51234758,
            0.49785921,
            0.56186088,
            0.48234613,
            0.53587696,
            0.51017805,
            0.52648205,
            0.53561864,
            0.51859151,
        ],
        [
            0.49436425,
            0.49701789,
            0.48742445,
            0.49987503,
            0.50632911,
            0.48973995,
            0.45825314,
            0.50263885,
            0.50030018,
            0.4850131,
            0.49630255,
            0.49142464,
            1.0,
            0.54224054,
            0.47594117,
            0.49583499,
            0.48823357,
            0.505025,
            0.50505051,
            0.49331557,
            0.5251825,
            0.52227503,
            0.53296381,
            0.4950005,
            0.51411238,
            0.48971596,
        ],
        [
            0.49169043,
            0.49166626,
            0.47968533,
            0.49468217,
            0.47094283,
            0.48463701,
            0.45562238,
            0.53576212,
            0.50456633,
            0.49531923,
            0.47744092,
            0.50122801,
            0.54224054,
            1.0,
            0.46600494,
            0.49392473,
            0.49258657,
            0.48801913,
            0.48288184,
            0.48954815,
            0.51419169,
            0.49234405,
            0.50934651,
            0.49593335,
            0.4695938,
            0.4865233,
        ],
        [
            0.49370526,
            0.50289163,
            0.50017506,
            0.52037259,
            0.49485352,
            0.51808103,
            0.47049967,
            0.47125353,
            0.51623561,
            0.51200655,
            0.466157,
            0.53183003,
            0.47594117,
            0.46600494,
            1.0,
            0.52479664,
            0.53166038,
            0.46886722,
            0.48914107,
            0.50602166,
            0.4746535,
            0.45643343,
            0.47975437,
            0.477122,
            0.47993857,
            0.48100048,
        ],
        [
            0.48344211,
            0.52592826,
            0.47899602,
            0.49763623,
            0.51200655,
            0.498132,
            0.52058931,
            0.46818671,
            0.49002793,
            0.49635181,
            0.49662296,
            0.5107774,
            0.49583499,
            0.49392473,
            0.52479664,
            1.0,
            0.57019044,
            0.48579062,
            0.47612246,
            0.49588416,
            0.47840023,
            0.46718057,
            0.48614487,
            0.47571476,
            0.4850131,
            0.48388658,
        ],
        [
            0.47901897,
            0.49024414,
            0.48574343,
            0.51132587,
            0.46589638,
            0.4877335,
            0.55328096,
            0.49473111,
            0.50684237,
            0.51392743,
            0.49091802,
            0.48962005,
            0.48823357,
            0.49258657,
            0.53166038,
            0.57019044,
            1.0,
            0.49207755,
            0.46891119,
            0.48638132,
            0.48785247,
            0.46650494,
            0.49495149,
            0.47746371,
            0.48383975,
            0.49431537,
        ],
        [
            0.48971596,
            0.49509852,
            0.50673964,
            0.4810699,
            0.48383975,
            0.50183169,
            0.47492401,
            0.49031625,
            0.51730381,
            0.49907671,
            0.49608096,
            0.51234758,
            0.505025,
            0.48801913,
            0.46886722,
            0.48579062,
            0.49207755,
            1.0,
            0.51266277,
            0.53101105,
            0.50779465,
            0.49833059,
            0.50849181,
            0.51310996,
            0.4886869,
            0.50454087,
        ],
        [
            0.50431187,
            0.47140904,
            0.4865233,
            0.46598322,
            0.49682035,
            0.47920261,
            0.4355211,
            0.47056609,
            0.5009769,
            0.49947555,
            0.48654698,
            0.49785921,
            0.50505051,
            0.48288184,
            0.48914107,
            0.47612246,
            0.46891119,
            0.51266277,
            1.0,
            0.49783442,
            0.46685341,
            0.48046894,
            0.50558673,
            0.50617534,
            0.46554935,
            0.49419323,
        ],
        [
            0.49835543,
            0.48969198,
            0.49421765,
            0.49019608,
            0.49610557,
            0.54513737,
            0.48612124,
            0.49887753,
            0.52413649,
            0.51064699,
            0.52312199,
            0.56186088,
            0.49331557,
            0.48954815,
            0.50602166,
            0.49588416,
            0.48638132,
            0.53101105,
            0.49783442,
            1.0,
            0.49686972,
            0.49828093,
            0.50286634,
            0.49790878,
            0.50454087,
            0.49788399,
        ],
        [
            0.48058439,
            0.49156958,
            0.50248731,
            0.49190811,
            0.48909322,
            0.50597045,
            0.45454545,
            0.50574015,
            0.48360576,
            0.4929508,
            0.49932591,
            0.48234613,
            0.5251825,
            0.51419169,
            0.4746535,
            0.47840023,
            0.48785247,
            0.50779465,
            0.46685341,
            0.49686972,
            1.0,
            0.50441362,
            0.49360778,
            0.50535678,
            0.4846605,
            0.50730519,
        ],
        [
            0.48780488,
            0.48732943,
            0.47413589,
            0.4725228,
            0.48028433,
            0.52712034,
            0.46242775,
            0.4685816,
            0.49603175,
            0.49507401,
            0.50316997,
            0.53587696,
            0.52227503,
            0.49234405,
            0.45643343,
            0.46718057,
            0.46650494,
            0.49833059,
            0.48046894,
            0.49828093,
            0.50441362,
            1.0,
            0.5276488,
            0.50535678,
            0.49169043,
            0.49689441,
        ],
        [
            0.50175615,
            0.51395385,
            0.50052555,
            0.48875855,
            0.50223495,
            0.48861526,
            0.47449585,
            0.49842995,
            0.49402233,
            0.49972515,
            0.51599587,
            0.51017805,
            0.53296381,
            0.50934651,
            0.47975437,
            0.48614487,
            0.49495149,
            0.50849181,
            0.50558673,
            0.50286634,
            0.49360778,
            0.5276488,
            1.0,
            0.49662296,
            0.51583617,
            0.50403226,
        ],
        [
            0.47917964,
            0.47585058,
            0.47856049,
            0.48576703,
            0.47612246,
            0.48290516,
            0.45055193,
            0.49514755,
            0.50510153,
            0.51666236,
            0.51347882,
            0.52648205,
            0.4950005,
            0.49593335,
            0.477122,
            0.47571476,
            0.47746371,
            0.51310996,
            0.50617534,
            0.49790878,
            0.50535678,
            0.50535678,
            0.49662296,
            1.0,
            0.51279422,
            0.53625054,
        ],
        [
            0.45454545,
            0.47236656,
            0.46036277,
            0.47330557,
            0.48144047,
            0.48484848,
            0.47427081,
            0.47218812,
            0.49960032,
            0.50489751,
            0.48816207,
            0.53561864,
            0.51411238,
            0.4695938,
            0.47993857,
            0.4850131,
            0.48383975,
            0.4886869,
            0.46554935,
            0.50454087,
            0.4846605,
            0.49169043,
            0.51583617,
            0.51279422,
            1.0,
            0.49275648,
        ],
        [
            0.48716325,
            0.4799616,
            0.48311513,
            0.47671259,
            0.48074612,
            0.49689441,
            0.44253662,
            0.48123195,
            0.49771053,
            0.4879715,
            0.48482498,
            0.51859151,
            0.48971596,
            0.4865233,
            0.48100048,
            0.48388658,
            0.49431537,
            0.50454087,
            0.49419323,
            0.49788399,
            0.50730519,
            0.49689441,
            0.50403226,
            0.53625054,
            0.49275648,
            1.0,
        ],
    ]
)
