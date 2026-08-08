from __future__ import annotations

import numpy as np
import pytest

from multisport_sim.simulation import Simulation


def test_headless_smoke_and_reset_are_deterministic() -> None:
    simulation = Simulation("campus")
    initial = simulation.data.qpos.copy()
    stats = simulation.run_headless(0.02)
    assert stats.steps == 20
    assert stats.simulated_seconds == pytest.approx(0.02)
    assert not np.array_equal(simulation.data.qpos, initial)
    simulation.reset()
    np.testing.assert_allclose(simulation.data.qpos, initial)
