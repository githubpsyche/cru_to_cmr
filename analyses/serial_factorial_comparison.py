# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%

#!%load_ext autoreload
#!%autoreload 2

import json
import os
import warnings

import jax.numpy as jnp
from jax import random
import numpy as np

from cru_to_cmr.models.compterm_omnibus_cru_cmr import (
    BaseCMRFactory as compterm_model_factory,
)
from cru_to_cmr.models.omnibus_cru_cmr import BaseCMRFactory as base_model_factory
from cru_to_cmr.fitting import ScipyDE as fitting_method
from cru_to_cmr.experimental.confusable_likelihood import (
    MemorySearchLikelihoodFnGenerator as loss_fn_generator,
)
from cru_to_cmr.summarize import summarize_parameters
from cru_to_cmr.experimental.confusable_simulation import simulate_h5_from_h5
from cru_to_cmr.helpers import import_from_string, load_data, generate_trial_mask

# from cru_to_cmr.helpers import to_numba_typed_dict
from matplotlib import rcParams  # type: ignore
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

comparison_analysis_paths = [
    "cru_to_cmr.analyses.srac.plot_srac",
    "cru_to_cmr.analyses.intrusion_error_rate.plot_intrusion_error_rate",
    "cru_to_cmr.analyses.omission_error_rate.plot_omission_error_rate",
    "cru_to_cmr.analyses.order_error_rate.plot_order_error_rate",
    "cru_to_cmr.analyses.crp.plot_crp",
]

comparison_analyses = [import_from_string(path) for path in comparison_analysis_paths]


warnings.filterwarnings("ignore")

# %%

# data params
data_name = "Gordon2021"
data_query = "data['condition'] == 2"
data_path = "../data/Gordon2021.h5"

# fitting params
redo_fits = False
redo_sims = False
run_tag = "Fitting"
relative_tolerance = 0.001
popsize = 15
num_steps = 1000
cross_rate = 0.9
diff_w = 0.85
best_of = 1

# sim params
experiment_count = 50
seed = 0

base_params = {
    "start_drift_rate": 1.0,
    "shared_support": 0.0,
    "item_support": 0.0,
    "learning_rate": 0.0,
    "primacy_scale": 0.0,
    "primacy_decay": 0.0,
    "encoding_drift_decrease": 1.0,
    "item_sensitivity_max": 0.0,
    "item_sensitivity_decrease": 1.0,
    "allow_repeated_recalls": True,
}


# %%

# Below is an example "flat" dictionary, preserving your existing model names
# and extending the same naming convention to cover *all* combinations
# of toggles for CRU in free recall.
#
# Each entry is a single-level dict: no "fixed"/"free" subdict, just a direct
# mapping from parameter_name -> [lower_bound, upper_bound].
# This allows you to load each configuration and interpret it as the set
# of free parameters, while anything not listed is presumably held fixed
# at some default CRU baseline value.

base_model_configs = {
    # ------------------------------------------------------------
    # Existing entries you already defined (preserved as-is)
    # ------------------------------------------------------------
    "Omnibus": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "start_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "shared_support": [2.220446049250313e-16, 99.9999999999999998],
        "item_support": [2.220446049250313e-16, 99.9999999999999998],
        "learning_rate": [2.220446049250313e-16, 0.9999999999999998],
        "primacy_scale": [2.220446049250313e-16, 99.9999999999999998],
        "primacy_decay": [2.220446049250313e-16, 99.9999999999999998],
        "stop_probability_scale": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_growth": [2.220446049250313e-16, 9.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
    },
    "BaseCMR": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "start_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "shared_support": [2.220446049250313e-16, 99.9999999999999998],
        "item_support": [2.220446049250313e-16, 99.9999999999999998],
        "learning_rate": [2.220446049250313e-16, 0.9999999999999998],
        "primacy_scale": [2.220446049250313e-16, 99.9999999999999998],
        "primacy_decay": [2.220446049250313e-16, 99.9999999999999998],
        "stop_probability_scale": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_growth": [2.220446049250313e-16, 9.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        # "encoding_drift_decrease" is omitted for CMR
    },
    "BaseCRU": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_scale": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_growth": [2.220446049250313e-16, 9.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
        # No learning_rate, shared_support/item_support, primacy, or start_drift
        # because it's the pure CRU baseline
    },
    "CRU with Feature-to-Context Learning": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_scale": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_growth": [2.220446049250313e-16, 9.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "learning_rate": [2.220446049250313e-16, 0.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
    },
    "CRU with MCF Pre-Experimental Support": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_scale": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_growth": [2.220446049250313e-16, 9.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "shared_support": [2.220446049250313e-16, 99.9999999999999998],
        "item_support": [2.220446049250313e-16, 99.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
    },
    "CRU with Learning Rate Primacy": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_scale": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_growth": [2.220446049250313e-16, 9.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "primacy_scale": [2.220446049250313e-16, 99.9999999999999998],
        "primacy_decay": [2.220446049250313e-16, 99.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
    },
    "CRU with Free Start Drift Rate": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_scale": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_growth": [2.220446049250313e-16, 9.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "start_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
    },
    # ------------------------------------------------------------
    # NEW ENTRIES: all combinations of the four CRU toggles
    # (learning_rate, pre-experimental, primacy, recall-init).
    # We'll keep the same naming convention: "CRU with X and Y [and Z...]"
    # Each entry includes the base CRU free params, plus the toggled ones.
    # ------------------------------------------------------------
    "CRU with Feature-to-Context and Pre-Expt": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_scale": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_growth": [2.220446049250313e-16, 9.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "learning_rate": [2.220446049250313e-16, 0.9999999999999998],
        "shared_support": [2.220446049250313e-16, 99.9999999999999998],
        "item_support": [2.220446049250313e-16, 99.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
    },
    "CRU with Feature-to-Context and Primacy": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_scale": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_growth": [2.220446049250313e-16, 9.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "learning_rate": [2.220446049250313e-16, 0.9999999999999998],
        "primacy_scale": [2.220446049250313e-16, 99.9999999999999998],
        "primacy_decay": [2.220446049250313e-16, 99.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
    },
    "CRU with Feature-to-Context and StartDrift": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_scale": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_growth": [2.220446049250313e-16, 9.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "learning_rate": [2.220446049250313e-16, 0.9999999999999998],
        "start_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
    },
    "CRU with Pre-Expt and Primacy": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_scale": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_growth": [2.220446049250313e-16, 9.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "shared_support": [2.220446049250313e-16, 99.9999999999999998],
        "item_support": [2.220446049250313e-16, 99.9999999999999998],
        "primacy_scale": [2.220446049250313e-16, 99.9999999999999998],
        "primacy_decay": [2.220446049250313e-16, 99.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
    },
    "CRU with Pre-Expt and StartDrift": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_scale": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_growth": [2.220446049250313e-16, 9.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "shared_support": [2.220446049250313e-16, 99.9999999999999998],
        "item_support": [2.220446049250313e-16, 99.9999999999999998],
        "start_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
    },
    "CRU with Primacy and StartDrift": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_scale": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_growth": [2.220446049250313e-16, 9.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "primacy_scale": [2.220446049250313e-16, 99.9999999999999998],
        "primacy_decay": [2.220446049250313e-16, 99.9999999999999998],
        "start_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
    },
    # Triple combinations
    "CRU with Feature-to-Context, Pre-Expt, and Primacy": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_scale": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_growth": [2.220446049250313e-16, 9.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "learning_rate": [2.220446049250313e-16, 0.9999999999999998],
        "shared_support": [2.220446049250313e-16, 99.9999999999999998],
        "item_support": [2.220446049250313e-16, 99.9999999999999998],
        "primacy_scale": [2.220446049250313e-16, 99.9999999999999998],
        "primacy_decay": [2.220446049250313e-16, 99.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
    },
    "CRU with Feature-to-Context, Pre-Expt, and StartDrift": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_scale": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_growth": [2.220446049250313e-16, 9.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "learning_rate": [2.220446049250313e-16, 0.9999999999999998],
        "shared_support": [2.220446049250313e-16, 99.9999999999999998],
        "item_support": [2.220446049250313e-16, 99.9999999999999998],
        "start_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
    },
    "CRU with Feature-to-Context, Primacy, and StartDrift": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_scale": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_growth": [2.220446049250313e-16, 9.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "learning_rate": [2.220446049250313e-16, 0.9999999999999998],
        "primacy_scale": [2.220446049250313e-16, 99.9999999999999998],
        "primacy_decay": [2.220446049250313e-16, 99.9999999999999998],
        "start_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
    },
    "CRU with Pre-Expt, Primacy, and StartDrift": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_scale": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_growth": [2.220446049250313e-16, 9.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "shared_support": [2.220446049250313e-16, 99.9999999999999998],
        "item_support": [2.220446049250313e-16, 99.9999999999999998],
        "primacy_scale": [2.220446049250313e-16, 99.9999999999999998],
        "primacy_decay": [2.220446049250313e-16, 99.9999999999999998],
        "start_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
    },
    # All four toggles on: same as "Omnibus" but specifically named for clarity
    "CRU with Feature-to-Context, Pre-Expt, Primacy, and StartDrift": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "start_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "shared_support": [2.220446049250313e-16, 99.9999999999999998],
        "item_support": [2.220446049250313e-16, 99.9999999999999998],
        "learning_rate": [2.220446049250313e-16, 0.9999999999999998],
        "primacy_scale": [2.220446049250313e-16, 99.9999999999999998],
        "primacy_decay": [2.220446049250313e-16, 99.9999999999999998],
        "stop_probability_scale": [2.220446049250313e-16, 0.9999999999999998],
        "stop_probability_growth": [2.220446049250313e-16, 9.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
    },
}

compterm_model_configs = {
    # 1) Omnibus
    "Omnibus, and ContextTerm": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "start_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "shared_support": [2.220446049250313e-16, 99.9999999999999998],
        "item_support": [2.220446049250313e-16, 99.9999999999999998],
        "learning_rate": [2.220446049250313e-16, 0.9999999999999998],
        "primacy_scale": [2.220446049250313e-16, 99.9999999999999998],
        "primacy_decay": [2.220446049250313e-16, 99.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
        # Removed "stop_probability_scale" and "stop_probability_growth"
    },
    # 2) BaseCMR
    "BaseCMR with ContextTerm": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "start_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "shared_support": [2.220446049250313e-16, 99.9999999999999998],
        "item_support": [2.220446049250313e-16, 99.9999999999999998],
        "learning_rate": [2.220446049250313e-16, 0.9999999999999998],
        "primacy_scale": [2.220446049250313e-16, 99.9999999999999998],
        "primacy_decay": [2.220446049250313e-16, 99.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        # no "encoding_drift_decrease", no stop_probability params
    },
    # 3) BaseCRU
    "BaseCRU with ContextTerm": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
        # Removed "stop_probability_scale" and "stop_probability_growth"
    },
    # 4) CRU with Feature-to-Context Learning
    "CRU with Feature-to-Context Learning, and ContextTerm": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "learning_rate": [2.220446049250313e-16, 0.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
    },
    # 5) CRU with MCF Pre-Experimental Support
    "CRU with MCF Pre-Experimental Support, and ContextTerm": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "shared_support": [2.220446049250313e-16, 99.9999999999999998],
        "item_support": [2.220446049250313e-16, 99.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
    },
    # 6) CRU with Learning Rate Primacy
    "CRU with Learning Rate Primacy, and ContextTerm": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "primacy_scale": [2.220446049250313e-16, 99.9999999999999998],
        "primacy_decay": [2.220446049250313e-16, 99.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
    },
    # 7) CRU with Free Start Drift Rate
    "CRU with Free Start Drift Rate, and ContextTerm": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "start_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
    },
    # ------------------------------------------------------------
    # Multi-Factor combos (removing stop_probability params, adding suffix)
    # ------------------------------------------------------------
    "CRU with Feature-to-Context and Pre-Expt, and ContextTerm": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "learning_rate": [2.220446049250313e-16, 0.9999999999999998],
        "shared_support": [2.220446049250313e-16, 99.9999999999999998],
        "item_support": [2.220446049250313e-16, 99.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
    },
    "CRU with Feature-to-Context and Primacy, and ContextTerm": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "learning_rate": [2.220446049250313e-16, 0.9999999999999998],
        "primacy_scale": [2.220446049250313e-16, 99.9999999999999998],
        "primacy_decay": [2.220446049250313e-16, 99.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
    },
    "CRU with Feature-to-Context and StartDrift, and ContextTerm": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "learning_rate": [2.220446049250313e-16, 0.9999999999999998],
        "start_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
    },
    "CRU with Pre-Expt and Primacy, and ContextTerm": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "shared_support": [2.220446049250313e-16, 99.9999999999999998],
        "item_support": [2.220446049250313e-16, 99.9999999999999998],
        "primacy_scale": [2.220446049250313e-16, 99.9999999999999998],
        "primacy_decay": [2.220446049250313e-16, 99.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
    },
    "CRU with Pre-Expt and StartDrift, and ContextTerm": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "shared_support": [2.220446049250313e-16, 99.9999999999999998],
        "item_support": [2.220446049250313e-16, 99.9999999999999998],
        "start_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
    },
    "CRU with Primacy and StartDrift, and ContextTerm": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "primacy_scale": [2.220446049250313e-16, 99.9999999999999998],
        "primacy_decay": [2.220446049250313e-16, 99.9999999999999998],
        "start_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
    },
    # Triple combos
    "CRU with Feature-to-Context, Pre-Expt Primacy, and ContextTerm": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "learning_rate": [2.220446049250313e-16, 0.9999999999999998],
        "shared_support": [2.220446049250313e-16, 99.9999999999999998],
        "item_support": [2.220446049250313e-16, 99.9999999999999998],
        "primacy_scale": [2.220446049250313e-16, 99.9999999999999998],
        "primacy_decay": [2.220446049250313e-16, 99.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
    },
    "CRU with Feature-to-Context, Pre-Expt StartDrift, and ContextTerm": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "learning_rate": [2.220446049250313e-16, 0.9999999999999998],
        "shared_support": [2.220446049250313e-16, 99.9999999999999998],
        "item_support": [2.220446049250313e-16, 99.9999999999999998],
        "start_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
    },
    "CRU with Feature-to-Context, Primacy StartDrift, and ContextTerm": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "learning_rate": [2.220446049250313e-16, 0.9999999999999998],
        "primacy_scale": [2.220446049250313e-16, 99.9999999999999998],
        "primacy_decay": [2.220446049250313e-16, 99.9999999999999998],
        "start_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
    },
    "CRU with Pre-Expt, Primacy StartDrift, and ContextTerm": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "shared_support": [2.220446049250313e-16, 99.9999999999999998],
        "item_support": [2.220446049250313e-16, 99.9999999999999998],
        "primacy_scale": [2.220446049250313e-16, 99.9999999999999998],
        "primacy_decay": [2.220446049250313e-16, 99.9999999999999998],
        "start_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
    },
    # 4-factor
    "CRU with Feature-to-Context, Pre-Expt, Primacy StartDrift, and ContextTerm": {
        "encoding_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "start_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "recall_drift_rate": [2.220446049250313e-16, 0.9999999999999998],
        "shared_support": [2.220446049250313e-16, 99.9999999999999998],
        "item_support": [2.220446049250313e-16, 99.9999999999999998],
        "learning_rate": [2.220446049250313e-16, 0.9999999999999998],
        "primacy_scale": [2.220446049250313e-16, 99.9999999999999998],
        "primacy_decay": [2.220446049250313e-16, 99.9999999999999998],
        "choice_sensitivity": [2.220446049250313e-16, 99.9999999999999998],
        "encoding_drift_decrease": [2.220446049250313e-16, 0.9999999999999998],
        # "stop_probability_scale" & "growth" removed
    },
}


query_parameters = [
    "encoding_drift_rate",
    "start_drift_rate",
    "recall_drift_rate",
    "shared_support",
    "item_support",
    "learning_rate",
    "primacy_scale",
    "primacy_decay",
    "mcf_trace_sensitivity",
    "stop_probability_scale",
    "stop_probability_growth",
    "choice_sensitivity",
    "encoding_drift_decrease",
    "item_sensitivity_max",
    "item_sensitivity_decrease",
]


# %%

# add subdirectories for each product type: json, figures, h5
product_dirs = {}
for product in ["fits", "figures", "simulations"]:
    product_dir = os.path.join("..", product)
    product_dirs[product] = product_dir
    if not os.path.exists(product_dir):
        os.makedirs(product_dir)

data = load_data(data_path)
trial_mask = generate_trial_mask(data, data_query)

max_size = np.max(data["pres_itemnos"])
connections = jnp.zeros((max_size, max_size))

# %%

for model_factory, model_configs in zip(
    [base_model_factory],
    [base_model_configs],
):
    for model_name, bounds in model_configs.items():
        model_name += "+Confusable"
        bounds = {
            "item_sensitivity_max": [1e-12, 20.0],
            "item_sensitivity_decrease": [1e-12, 0.999999],
            **bounds,
        }
        file_model_name = model_name.replace(" ", "_")

        fit_path = os.path.join(
            product_dirs["fits"], f"{data_name}_{file_model_name}_{run_tag}.json"
        )
        print(fit_path)

        if os.path.exists(fit_path) and not redo_fits:
            print("Fit already exists. Skipping.")
            with open(fit_path) as f:
                results = json.load(f)
            if "subject" not in results["fits"]:
                results["fits"]["subject"] = results["subject"]

        else:
            fitter = fitting_method(
                data,
                connections,
                base_params,
                model_factory,
                loss_fn_generator,
                hyperparams={
                    "num_steps": num_steps,
                    "pop_size": popsize,
                    "relative_tolerance": relative_tolerance,
                    "cross_over_rate": cross_rate,
                    "diff_w": diff_w,
                    "progress_bar": True,
                    "display_iterations": False,
                    "bounds": bounds,
                    "best_of": best_of,
                },
            )

            results = fitter.fit(trial_mask)
            results = dict(results)

            results["data_query"] = data_query
            results["model"] = model_name
            results["name"] = f"{data_name}_{file_model_name}_{run_tag}"
            results["relative_tolerance"] = relative_tolerance
            results["popsize"] = popsize
            results["num_steps"] = num_steps
            results["cross_rate"] = cross_rate
            results["diff_w"] = diff_w

            with open(fit_path, "w") as f:
                json.dump(results, f, indent=4)

        print(
            summarize_parameters(
                [results], query_parameters, include_std=True, include_ci=True
            )
        )

        sim_path = os.path.join(
            "simulations/", f"{data_name}_{model_name}_{run_tag}_seed_{seed}.hdf5"
        )

        print(sim_path)

        if os.path.exists(sim_path) and not redo_sims:
            sim = load_data(sim_path)
        else:
            rng = random.PRNGKey(seed)
            rng, rng_iter = random.split(rng)
            sim = simulate_h5_from_h5(
                model_factory=model_factory,
                dataset=data,
                connections=connections,
                parameters={
                    key: jnp.array(val) for key, val in results["fits"].items()
                },  # type: ignore
                trial_mask=trial_mask,
                experiment_count=experiment_count,
                rng=rng_iter,
            )

        list_lengths = [5, 6, 7]
        model_trial_mask = generate_trial_mask(sim, data_query)

        for analysis in comparison_analyses:
            base_fig_stem = f"{results['name']}_{analysis.__name__[5:]}"

            for ll in list_lengths:
                color_cycle = [each["color"] for each in rcParams["axes.prop_cycle"]]
                list_query = f"data['listLength'] == {ll}"
                ll_model_mask = generate_trial_mask(sim, list_query)
                ll_data_mask = generate_trial_mask(data, list_query)
                joint_model_mask = np.logical_and(model_trial_mask, ll_model_mask)
                joint_data_mask = np.logical_and(trial_mask, ll_data_mask)

                figure_str = f"{base_fig_stem}_LL{ll}.tif"
                figure_path = os.path.join("../figures/", figure_str)
                print(figure_str)

                axis = analysis(
                    datasets=[sim, data],
                    trial_masks=[np.array(joint_model_mask), np.array(joint_data_mask)],
                    color_cycle=color_cycle,
                    labels=["Model", "Data"],
                    contrast_name="source",
                    axis=None,
                    # distances=1 - connections,
                )

                axis.tick_params(labelsize=14)
                axis.set_xlabel(axis.get_xlabel(), fontsize=16)
                axis.set_ylabel(axis.get_ylabel(), fontsize=16)
                # axis.set_title(f'{results["name"]}'.replace("_", " "))
                plt.savefig(figure_path, bbox_inches="tight", dpi=300)
                # plt.show()

                cmap = plt.get_cmap("binary_r")          # 0 → pure black, 1 → white
                color_cycle = [
                    mcolors.to_hex(cmap(x))              # convert RGBA → "#rrggbb"
                    for x in np.linspace(0.00, 0.70, 2)  # stay away from x≈1 (near‑white)
                ]

                axis = analysis(
                    datasets=[sim, data],
                    trial_masks=[np.array(joint_model_mask), np.array(joint_data_mask)],
                    color_cycle=color_cycle,
                    labels=["Model", "Data"],
                    contrast_name="source",
                    axis=None,
                    # distances=1 - connections,
                )

                axis.tick_params(labelsize=14)
                axis.set_xlabel(axis.get_xlabel(), fontsize=16)
                axis.set_ylabel(axis.get_ylabel(), fontsize=16)
                # axis.set_title(f'{results["name"]}'.replace("_", " "))

                bw_path = os.path.join(
                    "../figures/", f"bw_{figure_str}"
                )
                plt.savefig(bw_path, bbox_inches="tight", dpi=300)
                plt.show()                              # B/W display (optional)
                plt.close()                             # free memory

