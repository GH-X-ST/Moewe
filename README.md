<p align="center">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/Project Status-Archived-ffffff?style=for-the-badge&labelColor=0d1117">
      <source media="(prefers-color-scheme: light)" srcset="https://img.shields.io/badge/Project Status-Archived-000000?style=for-the-badge&labelColor=ffffff">
      <img src="https://img.shields.io/badge/Read-Thesis-6e7781?style=for-the-badge&labelColor=ffffff" alt="Read thesis">
    </picture>
  </a>

<p align="center">
  <sub>Research implementation of the Joint-Flow Capture Governor</sub><br>
  <sub>for robust terminal control of unpowered fixed-wing robots</sub>
</p>

![Cover light](Cover.png#gh-light-mode-only)
![Cover dark](Cover_dark.png#gh-dark-mode-only)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/Python-3.12-FFE873?style=for-the-badge&labelColor=0d1117">
    <source media="(prefers-color-scheme: light)" srcset="https://img.shields.io/badge/Python-3.12-306998?style=for-the-badge&labelColor=ffffff">
    <img src="https://img.shields.io/badge/Python-3.12-3776ab?style=for-the-badge&labelColor=ffffff" alt="Python 3.12">
  </picture>
</p>

## About

Moewe accompanies the draft manuscript *Joint-Flow Capture Governor for Robust Terminal Control of Unpowered Fixed-Wing Robots*. It investigates the final approach of an unpowered aircraft when recovery distance is limited, commands are delayed, airflow varies across the airframe, and the complete aircraft must safely pass through a gate or make an admissible first landing contact.

The Joint-Flow Capture Governor (JFCG) supervises a nominal controller's aileron, elevator, and rudder references. Expensive reachable-set and terminal-event verification is performed offline. Online, a fixed three-variable quadratic program modifies the proposed references only when required by the stored certificate.

## Method Summary

- A motion-based observer supplies a containing aircraft-state set and centre-flow enclosure from timestamped pose and issued-command history.
- Centre flow and spatial-gradient uncertainty are shared across all aerodynamic strips; independent strip remainders cover unresolved local variation.
- An offline nonlinear oracle verifies gain cells, delayed command queues, state-domain validity, swept-airframe geometry, approach progress, and terminal events.
- A distance-indexed capture cover permits larger terminal error when more correction distance remains and stores a complete-cell backup reference.
- The runtime affine predictor converts robust constraints into linear inequalities in the three control-surface references.
- Gate missions check full-airframe aperture passage. Landing missions check first permitted contact, forbidden-body clearance, footprint containment, contact velocity, and touchdown attitude.
- Invalid estimates, out-of-envelope flow, failed lookup, invalid queues, or failed containment produce no JFCG reference and require a separate flight-stack contingency.

### Timing

| Operation | Period or bound |
|---|---:|
| Motion observer | 5 ms |
| Fast feedback | 20 ms |
| Reference governor | 100 ms |
| Maximum command-onset delay | 140 ms |
| Prediction horizon | 240 ms, 12 fast stages |

The horizon covers one governor period plus the declared command-onset delay. Post-onset actuator motion is represented separately by the actuator states and uncertain actuator time constants.

## Repository Guide

| Path | Purpose |
|---|---|
| `models/` | Aircraft dynamics, state, rigid-body geometry, and updraft utilities |
| `control/flow.py` | Dependency-preserving local-flow representation |
| `control/observer.py` | State and centre-flow integrity interface |
| `control/predictor.py` | Generated aircraft core and runtime affine predictor |
| `control/oracle.py` | Offline validated nonlinear propagation |
| `control/capture.py` | Distance-indexed capture compilation |
| `control/governor.py` | Deterministic three-reference active-set governor |
| `control/missions.py` | Gate, landing, free-space, and terminal constraints |
| `simulation/` | Arena and terminal-event adapters |
| `tests/` | Unit, soundness, integration, and runtime-isolation tests |

## Setup

The archive was last verified with:

| Package | Version |
|---|---:|
| Python | 3.12.11 |
| NumPy | 2.2.6 |
| SciPy | 1.16.3 |
| pytest | 9.0.3 |

Clone and create the environment:

```powershell
git clone https://github.com/GH-X-ST/Moewe.git
cd Moewe
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install numpy==2.2.6 scipy==1.16.3 pytest==9.0.3
```

Run the tests from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Last local result:

```text
126 passed, 1056 subtests passed
```

There is no stable command-line interface or bundled production configuration. The tests are the executable construction examples.

## Research Status

Moewe is a substantial software prototype, not a validated flight-control result. The architecture is implemented and software-tested, but work stopped before realistic uncertainty calibration, a nonempty production certificate, comparative ablations, an end-to-end simulation campaign, or hardware validation. The 5.4 m operational corridor also permits only about 0.9 s of flight near 6 m/s—roughly nine governor updates—while command onset may take 140 ms. These limits prevented a convincing demonstration of JFCG's benefit, so the associated draft was not considered ready for T-RO or RA-L.

Any guarantee remains conditional on calibrated uncertainty containment, conservative geometry, verified initialization, sound generated prediction, and the runtime timing contract. Test values are fixtures unless supported by measured evidence; passing the test suite does not establish flight safety. The archive does not claim global safety, route selection, structural-load certification, or post-contact stability.

Related experimental context is available in the [Nausicaa repository](https://github.com/GH-X-ST/Nausicaa) and [public thesis site](https://gh-x-st.github.io/Nausicaa-Thesis/).

## Citation and License

The associated manuscript remains a draft, and this software snapshot has no DOI. Until a formal record is available, cite the repository URL, exact Git commit, access date, and manuscript title.

No license file is included. Do not assume that public availability grants permission to copy, modify, redistribute, or use the software operationally.
