# Moewe

Moewe is a code repository for small fixed-wing glider simulation and control in local updrafts.

The current codebase contains:

- `moewe/sim`: state, frame, actuator, aerodynamic, rigid-body, updraft, and integration utilities.
- `moewe/tasks`: gate-traversal scenario geometry and metrics.
- `moewe/control`: pseudo-trim, finite-difference linearisation, local controllers, and closed-loop rollout utilities.
- `moewe/primitives`: structured primitive generation, rollout validation, compression, and runtime retrieval.
- `moewe/returnability`: primitive-transition graphs, recoverability class-set reports, and empirical certificate scaffolding.
- `moewe/governor`: deterministic online filtering plus manuscript-facing request and decision records for a manoeuvre primitive governor.
- `moewe/objectives`: objective proposer scaffolds that produce closed-loop primitive requests without safety filtering.
- `moewe/baselines`: smoke-scale baseline and ablation utilities covering ungoverned primitive selection, filter-only selection, no-returnability selection, tracking, guidance, wind-aware guidance, inner-loop substitution, and lift-evidence removal.
- `moewe/campaigns`: smoke-scale decision, rollout, diagnostic, and random updraft challenge campaign utilities for comparing selector and baseline behavior.
- `moewe/benchmarks`: final benchmark preflight, method registry, schema, and guard utilities without generated benchmark results.
- `config`: small smoke configurations for the implemented modules.
- `tests`: unit and smoke tests for the implemented behavior.

This repository does not contain experiment datasets, generated benchmark outputs, paper figures, videos, large generated results, or hardware logs.

## Test

```bash
python -m pytest
```

## Scope

The repository is an early research implementation. It includes manuscript-facing interfaces for closed-loop primitive requests, primitive evidence records, returnability certificates, and accept/degrade/reject/rank decision records at smoke scale. These interfaces are preflight scaffolding for a future frozen benchmark campaign; they are not a formal viability proof and are not final benchmark evidence. The repository does not yet provide final benchmark outputs, a real-flight dataset, a firmware package, or a paper artifact.

## License

MIT License.
