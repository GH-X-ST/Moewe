<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/Project Status-Archived-ffffff?style=for-the-badge&labelColor=0d1117">
    <source media="(prefers-color-scheme: light)" srcset="https://img.shields.io/badge/Project Status-Archived-000000?style=for-the-badge&labelColor=ffffff">
    <img src="https://img.shields.io/badge/Project Status-Archived-6e7781?style=for-the-badge&labelColor=ffffff" alt="Project status: archived">
  </picture>
</p>

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

## Research Status

Moewe was archived because its intended advantage could not be demonstrated convincingly within the available facility and project window—not because the implementation was empty.

**The arena conflicts with the claimed advantage.** The usable flight length is only 5.4 m: near 6 m/s, a traversal lasts about 0.9 s and permits roughly nine governor updates. A 140 ms command-onset delay consumes about 0.84 m, while the 240 ms prediction horizon spans about 1.44 m. JFCG is intended to make repeated, recursively safe corrections as distance decreases; here, only a few commands can become effective, making a successful run difficult to distinguish from launch stabilisation followed by one terminal correction.

**The novelty is real but limited.** The distinctive contribution is the integration of shared centre-flow and spatial-gradient factors with delayed-command propagation, full-body terminal geometry, a distance-indexed capture cover, and a stored backup reference. Its underlying ingredients—reference governors, zonotopes, reachability, local feedback, and a small online QP—are established methods. Without ablations showing that the joint-flow structure materially enlarges the certified region or improves flight outcomes, the work demonstrates a thoughtful integration rather than a strong state-of-the-art advance.

Development therefore stopped before realistic uncertainty calibration, a nonempty production certificate, comparative baselines, an end-to-end simulation campaign, or hardware validation. The associated draft was not considered ready for T-RO or RA-L. Any guarantee remains conditional on calibrated uncertainty containment, conservative geometry, verified initialization, sound generated prediction, and the runtime timing contract; passing the software tests does not establish flight safety.

## About

Moewe accompanies the draft manuscript *Joint-Flow Capture Governor for Robust Terminal Control of Unpowered Fixed-Wing Robots*. It investigates the final approach of an unpowered aircraft when recovery distance is limited, commands are delayed, airflow varies across the airframe, and the complete aircraft must safely pass through a gate or make an admissible first landing contact.

## Method and Timing

JFCG supervises the nominal controller's aileron, elevator, and rudder references:

- **Before flight:** a nonlinear verifier propagates state, shared centre-flow and gradient uncertainty, local strip remainders, delayed commands, and swept-airframe geometry. It stores a distance-indexed capture cover and a feasible backup reference. The 240 ms horizon uses twelve 20 ms stages, covering one 100 ms governor period plus the maximum 140 ms command-onset delay.
- **During flight:** the motion-based observer updates its containing state and centre-flow set every 5 ms; nominal feedback runs every 20 ms; and every 100 ms a three-variable QP keeps or minimally adjusts the proposed references using the stored certificate.
- **At the terminal event:** the complete aircraft must pass through the gate, or the permitted landing points must make first contact while the remaining body, footprint, velocity, and attitude constraints hold. Invalid estimates, queues, flow, lookup, or containment return no JFCG reference and require a separate flight-stack contingency.

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

Related experimental context is available in the [Nausicaa repository](https://github.com/GH-X-ST/Nausicaa) and [public thesis site](https://gh-x-st.github.io/Nausicaa-Thesis/).

## Citation and License

The associated manuscript remains a draft, and this software snapshot has no DOI. Until a formal record is available, cite the repository URL, exact Git commit, access date, and manuscript title.

No license file is included. Do not assume that public availability grants permission to copy, modify, redistribute, or use the software operationally.
