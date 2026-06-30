from __future__ import annotations

import numpy as np

from moewe.primitives import PrimitiveGrammarSpec, generate_primitives


def test_phase_references_return_finite_state_and_command_vectors() -> None:
    primitive = generate_primitives(PrimitiveGrammarSpec.smoke())[0]

    for phase in primitive.phases:
        for local_time_s in (0.0, 0.5 * phase.duration_s, phase.duration_s):
            sample = phase.sample(local_time_s)
            assert sample.state.finite()
            assert sample.command_rad.shape == (3,)
            assert np.isfinite(sample.command_rad).all()


def test_reference_sampling_includes_final_time_and_uses_command_order() -> None:
    primitive = generate_primitives(PrimitiveGrammarSpec.smoke())[0]
    reference = primitive.reference
    times = reference.sample_times(0.05)

    assert times[0] == 0.0
    assert np.isclose(times[-1], reference.total_duration_s)
    for time_s in times:
        state = reference.state_at(float(time_s))
        command = reference.command_at(float(time_s))
        assert state.finite()
        assert command.shape == (3,)
        assert np.isfinite(command).all()


def test_pitch_pulse_returns_to_entry_pitch_at_phase_end() -> None:
    primitive = generate_primitives(PrimitiveGrammarSpec.smoke())[0]
    pitch_phase = primitive.phases[2]

    start_pitch = pitch_phase.sample(0.0).state.euler_rad[1]
    end_pitch = pitch_phase.sample(pitch_phase.duration_s).state.euler_rad[1]

    assert np.isclose(start_pitch, end_pitch)
