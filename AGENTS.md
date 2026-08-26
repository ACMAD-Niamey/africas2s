# Agent instructions for africas2s

Instructions for any AI coding agent (Claude Code, Codex, OpenCode, etc.) working in this repository.

## Before you start

Read [`skills/africas2s/SKILL.md`](skills/africas2s/SKILL.md) before writing or modifying code that uses africas2s. It is the authoritative quick reference for the public API, data conventions, method/metric/strategy registries, and the statistical discipline rules (tercile leakage, grid rules, CV requirements), with deeper detail in `skills/africas2s/references/`. If your harness supports Agent Skills natively, load the skill instead of re-deriving the API from source.

## Keep the docs and skill in sync — this is a hard requirement

The skill is a snapshot of the source. Any change that alters observable behavior MUST update the matching documentation **in the same PR**:

| If you change... | You must update... |
|---|---|
| Public verb signatures/behavior (`downscale`, `optimize`, `train`, `calibrate`, `ensemble`, `skill`, `skill_compare`, `seasonal_mme`, `flex_forecast`, ...) or result dataclasses | `skills/africas2s/SKILL.md` + `skills/africas2s/references/api.md` |
| `Index` (`indices.py`): named indices, `REGIONS`, `transform`/`weights`/`baseline`, `reduce` signature | `skills/africas2s/references/api.md` |
| Analog selection (`analog.py`), scenario completion (`completion.py`), climate positioning (`climate.py`), scalar-series calibration (`series.py`), calendar/season-step utilities (`time.py`) | `skills/africas2s/references/analog-completion.md` (+ `SKILL.md` "Analog completion & climate positioning") |
| Methods, calibrators, ensemble strategies, CV schemes, `pool_ensembles` (`ensemble.py`), tercile combination/masking (`combine.py`) — add/remove/rename, parameter changes | `skills/africas2s/references/methods.md` |
| Metrics, tercile conversion, boundaries, significance tools (`metrics/significance.py`, `metrics/cross_validation.py`) | `skills/africas2s/references/metrics-and-terciles.md` |
| Plotting/reporting functions or export formats — incl. `plotting/maps.py`, `plotting/scenarios.py`, `plotting/forecasts.py` | `skills/africas2s/references/plotting-reporting.md` |
| Error messages, extras, environment requirements | `skills/africas2s/references/troubleshooting.md` |
| Rosetta integration / input data shapes | `skills/africas2s/SKILL.md` ("Getting data in") |
| Anything user-facing | `README.md` if it covers the topic |

Also update `skills/africas2s/examples/` if a change breaks or obsoletes an example. If you are unsure whether a change is documented, grep `skills/` and `README.md` for the function, method name, or parameter you touched — stale docs are treated as bugs.

## Repo conventions

- Package layout: `src/africas2s/` (import name `africas2s`, distribution `africas2s`). Python ≥ 3.10, `uv` for dependency management.
- Methods/metrics/strategies/calibrators are looked up by name via `africas2s/registry.py` — new capabilities register there rather than being hard-wired.
- Tests: bare `pytest` must stay fast (< 30 s) and green; markers `integration`, `agreement`, `gpu` gate slow/real-data suites. Coverage target > 85% on `src/africas2s`. New behavior needs tests.
- Statistical honesty is a design invariant: cross-validated outputs must never leak held-out years (use `to_tercile_cv`, nested CV safeguards). Do not weaken these paths for convenience.
- CCA numerics intentionally match CPT Fortran 17.8.3 — changes there must preserve parity (`scripts/reproduce.py`, `agreement` tests).
