"""Feature-based model configuration for CRU-to-CMR factorial comparison.

Each model variant is a combination of CMR features. Features define which
parameters are free (with bounds); parameters not listed are held fixed at
base_params values.

Two factory types produce two sets of configs:
- base: uses positional termination (stop_probability params are free)
- compterm: uses competitive termination (no stop_probability params)
"""

from itertools import combinations
from typing import Any

import jax.numpy as jnp

eps = jnp.finfo(jnp.float32).eps

# ---------------------------------------------------------------------------
# Parameter bound ranges
# ---------------------------------------------------------------------------
RATE = [float(eps), 1 - float(eps)]
GROWTH = [float(eps), 10 - float(eps)]
SUPPORT = [float(eps), 100 - float(eps)]

# ---------------------------------------------------------------------------
# Features and their free parameters
# ---------------------------------------------------------------------------
CORE_CRU = {
    "encoding_drift_rate": RATE,
    "recall_drift_rate": RATE,
    "choice_sensitivity": SUPPORT,
}

STOP_PROBABILITY = {
    "stop_probability_scale": RATE,
    "stop_probability_growth": GROWTH,
}

FEATURES = {
    "feature_to_context": {"learning_rate": RATE},
    "pre_expt": {"shared_support": SUPPORT, "item_support": SUPPORT},
    "primacy": {"primacy_scale": SUPPORT, "primacy_decay": SUPPORT},
    "start_drift": {"start_drift_rate": RATE},
    "encoding_decrease": {"encoding_drift_decrease": RATE},
}

# Human-readable labels — single-feature names use longer descriptive forms,
# multi-feature combos use abbreviated forms (matching existing fit file names)
FEATURE_LABELS_SINGLE = {
    "feature_to_context": "Feature-to-Context Learning",
    "pre_expt": "MCF Pre-Experimental Support",
    "primacy": "Learning Rate Primacy",
    "start_drift": "Free Start Drift Rate",
}
FEATURE_LABELS_COMBO = {
    "feature_to_context": "Feature-to-Context",
    "pre_expt": "Pre-Expt",
    "primacy": "Primacy",
    "start_drift": "StartDrift",
}

# The four CRU toggle features (encoding_decrease is CRU-specific, always on in Omnibus)
CRU_TOGGLES = ["feature_to_context", "pre_expt", "primacy", "start_drift"]


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def make_bounds(*feature_names: str, include_stop: bool = True) -> dict[str, list[float]]:
    """Merge CORE_CRU + selected features into a single bounds dict."""
    bounds = dict(CORE_CRU)
    if include_stop:
        bounds.update(STOP_PROBABILITY)
    for name in feature_names:
        bounds.update(FEATURES[name])
    return bounds


def _variant_name(feature_names: tuple[str, ...]) -> str:
    """Generate human-readable variant name from feature combination.

    Single-feature variants use descriptive names (e.g., "Feature-to-Context Learning").
    Multi-feature variants use abbreviated names (e.g., "Feature-to-Context, Pre-Expt").
    Names must match existing fit file naming conventions.
    """
    if not feature_names:
        return "BaseCRU"
    if len(feature_names) == 1:
        return f"CRU with {FEATURE_LABELS_SINGLE[feature_names[0]]}"
    labels = [FEATURE_LABELS_COMBO[f] for f in feature_names]
    if len(labels) == 2:
        return f"CRU with {labels[0]} and {labels[1]}"
    return f"CRU with {', '.join(labels[:-1])}, and {labels[-1]}"


def _compterm_name(base_name: str) -> str:
    """Convert a base variant name to its compterm (ContextTerm) equivalent.

    Single-feature and BaseCRU: append "with ContextTerm" or ", and ContextTerm"
    Multi-feature: replace ", and <last>" with " <last>, and ContextTerm"
    """
    if base_name in ("BaseCMR", "BaseCRU"):
        return f"{base_name} with ContextTerm"
    if ", and " in base_name:
        # Multi-feature: "CRU with X, and Y" -> "CRU with X Y, and ContextTerm"
        return base_name.replace(", and ", " ", 1).rstrip() + ", and ContextTerm"
    # Single-feature: "CRU with X" -> "CRU with X, and ContextTerm"
    return f"{base_name}, and ContextTerm"


def _generate_cru_factorial(include_stop: bool) -> dict[str, dict[str, list[float]]]:
    """Generate all CRU toggle combinations."""
    configs: dict[str, dict[str, list[float]]] = {}
    for r in range(len(CRU_TOGGLES) + 1):
        for combo in combinations(CRU_TOGGLES, r):
            name = _variant_name(combo)
            configs[name] = make_bounds(*combo, include_stop=include_stop)
    return configs


def generate_base_configs() -> dict[str, dict[str, list[float]]]:
    """Generate all free-recall model configs (positional termination)."""
    configs: dict[str, dict[str, list[float]]] = {}

    # Named models
    configs["Omnibus"] = make_bounds(*CRU_TOGGLES, "encoding_decrease", include_stop=True)
    configs["BaseCMR"] = make_bounds(*CRU_TOGGLES, include_stop=True)

    # CRU factorial
    configs.update(_generate_cru_factorial(include_stop=True))

    return configs


def generate_compterm_configs() -> dict[str, dict[str, list[float]]]:
    """Generate all free-recall model configs (competitive termination)."""
    configs: dict[str, dict[str, list[float]]] = {}

    # Named models with ", and ContextTerm" suffix
    configs["Omnibus, and ContextTerm"] = make_bounds(
        *CRU_TOGGLES, "encoding_decrease", include_stop=False
    )
    configs["BaseCMR with ContextTerm"] = make_bounds(*CRU_TOGGLES, include_stop=False)

    # CRU factorial with ContextTerm suffix
    for name, bounds in _generate_cru_factorial(include_stop=False).items():
        configs[_compterm_name(name)] = bounds

    return configs


# ---------------------------------------------------------------------------
# Fixed parameters (values used when a parameter is not free)
# ---------------------------------------------------------------------------
BASE_PARAMS: dict[str, Any] = {
    "start_drift_rate": 1.0,
    "shared_support": 0.0,
    "item_support": 0.0,
    "learning_rate": 0.0,
    "primacy_scale": 0.0,
    "primacy_decay": 0.0,
    "encoding_drift_decrease": 1.0,
    "allow_repeated_recalls": True,
}

# ---------------------------------------------------------------------------
# Serial recall (confusable) config generation
# ---------------------------------------------------------------------------
CONFUSABLE_PARAMS = {
    "item_sensitivity_max": [1e-12, 20.0],
    "item_sensitivity_decrease": [1e-12, 0.999999],
}


def _confusable_configs(configs: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, list[float]]]:
    """Add confusable parameters and +Confusable suffix to each config."""
    return {
        f"{name}+Confusable": {**CONFUSABLE_PARAMS, **bounds}
        for name, bounds in configs.items()
    }


def generate_serial_base_configs() -> dict[str, dict[str, list[float]]]:
    """Generate serial recall configs (base factory + confusable)."""
    return _confusable_configs(generate_base_configs())


def generate_serial_compterm_configs() -> dict[str, dict[str, list[float]]]:
    """Generate serial recall configs (compterm factory + confusable)."""
    return _confusable_configs(generate_compterm_configs())


# ---------------------------------------------------------------------------
# Pre-built config dicts
# ---------------------------------------------------------------------------
base_model_configs = generate_base_configs()
compterm_model_configs = generate_compterm_configs()
serial_base_configs = generate_serial_base_configs()
serial_compterm_configs = generate_serial_compterm_configs()
