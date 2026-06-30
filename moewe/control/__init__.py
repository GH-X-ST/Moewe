"""Local-control utilities for trim, linearisation, and short rollouts."""

from .interface import CommandLimits, ControllerMetadata, LocalController
from .linearise import (
    Linearisation,
    LinearisationCheck,
    finite_difference_linearisation,
    linearisation_step_check,
    to_euler_discrete,
)
from .lqr import LQRController, build_lqr_controller, solve_discrete_lqr
from .pd import PDController, PDGains
from .rollout import ClosedLoopResult, run_closed_loop
from .trim import TrimResidual, TrimResult, TrimSpec, pseudo_trim

__all__ = [
    "ClosedLoopResult",
    "CommandLimits",
    "ControllerMetadata",
    "LQRController",
    "Linearisation",
    "LinearisationCheck",
    "LocalController",
    "PDController",
    "PDGains",
    "TrimResidual",
    "TrimResult",
    "TrimSpec",
    "build_lqr_controller",
    "finite_difference_linearisation",
    "linearisation_step_check",
    "pseudo_trim",
    "run_closed_loop",
    "solve_discrete_lqr",
    "to_euler_discrete",
]
