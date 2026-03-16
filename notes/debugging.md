# Debugging Notes

## Import convention: module-as-alias — RESOLVED

The mainline jaxcmr models use a convention where component modules are imported as aliases:
```python
import cru_to_cmr.components.context as TemporalContext
import cru_to_cmr.components.linear_memory as LinearMemory
```

This makes `TemporalContext.init(...)` resolve to the module-level `init()` function, not a classmethod. The `models_cru_to_cmr/` files originally used `from ... import ClassName`, which broke this pattern. Fixed by adopting the mainline convention.

## Model refactoring — RESOLVED

All 4 model files refactored to match jaxcmr conventions:
- Create-fn injection pattern for components
- `PositionalTermination` replaces inline `exponential_stop_probability`
- `make_factory()` replaces static `CMRFactory`/`BaseCMRFactory` classes
- Custom `CompetitiveTermination` classes for compterm variants

## `distances` parameter — RESOLVED

`factorial_comparison.py` passed `distances=1-connections` to `plot_spc`/`plot_crp`/`plot_pnr`, but these functions don't accept that parameter. `connections` was a zero matrix, so `distances` was all ones — a no-op. Removed the kwarg.

## `connections` vs `features` — RESOLVED

`simulate_h5_from_h5` expects `features=` but the script passed `connections=`. Renamed in the call site.
