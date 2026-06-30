# Moewe

Moewe is a code repository for small fixed-wing glider simulation and control in local updrafts.

The current codebase contains:

- `moewe/sim`: state, frame, actuator, aerodynamic, rigid-body, updraft, and integration utilities.
- `moewe/tasks`: gate-traversal scenario geometry and metrics.
- `moewe/control`: pseudo-trim, finite-difference linearisation, local controllers, and closed-loop rollout utilities.
- `moewe/primitives`: structured primitive generation, rollout validation, compression, and runtime retrieval.
- `moewe/returnability`: primitive-transition graphs and recoverability class-set reports.
- `config`: small smoke configurations for the implemented modules.
- `tests`: unit and smoke tests for the implemented behavior.

This repository does not contain experiment datasets, paper figures, videos, large generated results, hardware logs, or private planning notes.

## Test

```bash
python -m pytest
```

## Scope

The repository is an early research implementation. It does not yet provide a final online viability governor, benchmark campaign, real-flight dataset, firmware package, or paper artifact.

## License

MIT License.
