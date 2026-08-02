# TACE Code Audit Report

## Audit requirement

Read [`NO_NEED_TO_MODIFY.md`](NO_NEED_TO_MODIFY.md) before starting or updating
the audit. Behaviors documented there are intentional constraints or project
decisions and must not be reported as defects unless their documented behavior
is broken.

## Remaining issues

### TACE-001: Optimizer configuration mutates the stored configuration

Relevant code:

- `tace/lightning/lit_model.py:482-490`
- `tace/lightning/lit_model.py:567-572`

Optimizer and scheduler setup removes fields such as `_target_` with `pop()`
from the model's stored configuration. Repeated calls to
`configure_optimizers()` can behave differently or fail, and the retained
configuration no longer matches the configuration originally supplied.
