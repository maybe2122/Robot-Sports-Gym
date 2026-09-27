"""The flight predictor is only worth having if it matches the backend."""

from __future__ import annotations

import numpy as np
import pytest

from multisport_sim.benchmark.ball_flight import (
    MAGNUS_LIFT_CEILING,
    MAGNUS_SCALE,
    BallFlightModel,
    TableBounce,
)
from multisport_sim.benchmark.rules.table_tennis import TABLE_TENNIS
from multisport_sim.physics import Atmosphere

PLANE_X = -1.55


class TestTheForceLaw:
    def test_it_uses_the_same_magnus_constants_as_the_simulation(self) -> None:
        """A predictor with its own lift law would be modelling another ball."""
        assert MAGNUS_SCALE == Atmosphere().magnus_scale
        # The ceiling is written into physics.Aerodynamics.apply as a literal;
        # this is the assertion that notices if either side moves.
        assert MAGNUS_LIFT_CEILING == 0.35

    def test_a_ball_at_rest_only_falls(self) -> None:
        model = BallFlightModel()
        acceleration = model.acceleration(np.zeros(3), None)
        assert acceleration[0] == acceleration[1] == 0.0
        assert acceleration[2] == pytest.approx(-9.81)

    def test_drag_opposes_motion(self) -> None:
        model = BallFlightModel()
        acceleration = model.acceleration(np.array([5.0, 0.0, 0.0]), None)
        assert acceleration[0] < 0.0

    def test_spin_below_the_speed_threshold_adds_no_lift(self) -> None:
        """The backend switches Magnus off below 0.1 m/s and so does this."""
        model = BallFlightModel()
        slow = np.array([0.05, 0.0, 0.0])
        spin = np.array([0.0, 200.0, 0.0])
        assert model.acceleration(slow, spin) == pytest.approx(
            model.acceleration(slow, None)
        )


class TestThePrediction:
    def test_it_declines_when_the_ball_is_already_past_the_plane(self) -> None:
        model = BallFlightModel()
        assert (
            model.predict_plane_crossing(
                (-2.0, 0.0, 1.0), (-3.0, 0.0, 0.0), plane_x=PLANE_X
            )
            is None
        )

    def test_it_declines_when_the_ball_is_moving_away(self) -> None:
        model = BallFlightModel()
        assert (
            model.predict_plane_crossing(
                (0.0, 0.0, 1.0), (3.0, 0.0, 0.0), plane_x=PLANE_X
            )
            is None
        )

    def test_it_declines_rather_than_reporting_a_crossing_at_the_horizon(self) -> None:
        model = BallFlightModel()
        assert (
            model.predict_plane_crossing(
                (0.0, 0.0, 1.0), (-0.05, 0.0, 0.0), plane_x=PLANE_X, horizon_s=0.5
            )
            is None
        )

    def test_it_rejects_a_malformed_state(self) -> None:
        model = BallFlightModel()
        with pytest.raises(ValueError, match="three-vectors"):
            model.predict_plane_crossing((0.0, 0.0), (-3.0, 0.0, 0.0), plane_x=PLANE_X)

    def test_free_flight_matches_a_hand_integrated_parabola_without_drag(self) -> None:
        """With drag scaled away the integrator must reproduce the closed form."""
        model = BallFlightModel()
        model._drag_factor = 0.0
        crossing = model.predict_plane_crossing(
            (0.0, 0.0, 1.0), (-4.0, 0.0, 0.0), plane_x=PLANE_X
        )
        assert crossing is not None
        expected_time = 1.55 / 4.0
        assert crossing.time_s == pytest.approx(expected_time, abs=2e-3)
        assert crossing.z == pytest.approx(
            1.0 - 0.5 * 9.81 * expected_time**2, abs=2e-3
        )
        assert crossing.bounces == 0

    def test_drag_makes_the_ball_arrive_later_than_a_parabola_would(self) -> None:
        model = BallFlightModel()
        crossing = model.predict_plane_crossing(
            (0.0, 0.0, 1.0), (-4.0, 0.0, 0.0), plane_x=PLANE_X
        )
        assert crossing is not None
        assert crossing.time_s > 1.55 / 4.0


class TestTheBounce:
    def test_it_reverses_the_normal_component_and_keeps_the_tangent(self) -> None:
        bounce = TableBounce.from_surface(TABLE_TENNIS)
        velocity, spin = bounce.apply(
            np.array([-4.0, 0.5, -3.0]), None, radius=0.02
        )
        assert velocity[2] == pytest.approx(3.0 * bounce.restitution)
        assert velocity[0] == pytest.approx(-4.0 * bounce.tangential_retention)
        assert spin is None

    def test_the_table_footprint_comes_from_the_rule_surface(self) -> None:
        """Predictor and judge must not disagree about where the table is."""
        bounce = TableBounce.from_surface(TABLE_TENNIS)
        assert bounce.surface_z == TABLE_TENNIS.top_height_m
        assert bounce.half_length_x == pytest.approx(TABLE_TENNIS.length_m / 2.0)
        assert bounce.covers(0.5, 0.0)
        assert not bounce.covers(-3.0, 0.0)
        assert not bounce.covers(0.0, 2.0)

    def test_a_ball_dropped_onto_the_table_crosses_the_plane_after_bouncing(self) -> None:
        model = BallFlightModel()
        bounce = TableBounce.from_surface(TABLE_TENNIS)
        state = ((0.5, 0.0, 1.0), (-4.0, 0.0, -1.0))
        without = model.predict_plane_crossing(*state, plane_x=PLANE_X)
        with_bounce = model.predict_plane_crossing(*state, plane_x=PLANE_X, bounce=bounce)
        # Without the table the ball falls through it and never reaches the
        # plane at a sane height; with it, it bounces up and arrives.
        assert with_bounce is not None
        assert with_bounce.bounces >= 1
        assert with_bounce.z > TABLE_TENNIS.top_height_m
        assert without is None or without.z < with_bounce.z

    def test_it_stops_bouncing_once_the_budget_is_spent(self) -> None:
        model = BallFlightModel()
        bounce = TableBounce.from_surface(TABLE_TENNIS)
        crossing = model.predict_plane_crossing(
            (1.3, 0.0, 0.9),
            (-2.0, 0.0, -1.0),
            plane_x=PLANE_X,
            bounce=bounce,
            max_bounces=1,
        )
        assert crossing is None or crossing.bounces <= 1


@pytest.mark.parametrize("level", ["L1", "L2", "L4"])
def test_it_predicts_the_real_crossing_to_within_a_blade_radius(level: str) -> None:
    """The claim this module exists for, checked against the actual backend.

    The swing commits about 160 ms before the ball reaches the strike plane, so
    that is where the prediction is sampled.  A free-flight predictor is wrong
    by a median 33-53 cm above L1 because most shots bounce inside that window;
    the tolerance here is 5 cm, comfortably below the 7.5 cm blade radius and
    comfortably above the ~1 cm the model actually achieves.
    """
    mujoco = pytest.importorskip("mujoco")
    del mujoco
    from multisport_sim.benchmark.backends.mujoco import MujocoShotBackend
    from multisport_sim.benchmark.shot_bank import ShotBank

    lead_s = 0.16
    model = BallFlightModel()
    bounce = TableBounce.from_surface(TABLE_TENNIS)
    backend = MujocoShotBackend()
    bank = ShotBank.from_resource(split="dev", task="table_tennis/return-v1")
    shots = [shot for shot in bank if shot.level == level][:6]
    assert shots

    errors = []
    for shot in shots:
        backend.reset()
        backend.launch_ball(shot)
        trajectory = []
        previous_x = None
        crossing = None
        for _ in range(4000):
            backend.step()
            ball = backend.get_ball_state()
            trajectory.append(
                (
                    backend.time,
                    np.array(ball.position),
                    np.array(ball.linear_velocity),
                    np.array(ball.angular_velocity),
                )
            )
            x = ball.position[0]
            if previous_x is not None and previous_x >= PLANE_X > x and ball.position[2] > 0.55:
                crossing = (backend.time, float(ball.position[1]), float(ball.position[2]))
                break
            previous_x = x
            if ball.position[2] < 0.2:
                break
        assert crossing is not None, f"{shot.shot_id} never reached the strike plane"
        time_s, y, z = crossing
        _, position, velocity, spin = min(
            trajectory, key=lambda row: abs((time_s - row[0]) - lead_s)
        )
        predicted = model.predict_plane_crossing(
            position, velocity, plane_x=PLANE_X, angular_velocity=spin, bounce=bounce
        )
        assert predicted is not None, f"{shot.shot_id} was not predicted to arrive"
        errors.append(float(np.hypot(predicted.y - y, predicted.z - z)))

    assert max(errors) < 0.05, f"worst prediction error {max(errors):.3f} m"


class TestTheScalarInnerLoop:
    """The integrator runs on floats for speed; it must still be the same law.

    With three-element numpy arrays one prediction cost 23 ms against a 5 ms
    control period -- the controller could not have run in real time, and
    ``inference_latency_ms`` would have reported it.  The scalar form costs
    0.6 ms.  These tests are what keeps the fast path honest.
    """

    @pytest.mark.parametrize(
        ("velocity", "spin"),
        [
            ((4.0, 0.0, 0.0), None),
            ((-4.0, 0.5, -3.0), (0.0, 60.0, 0.0)),
            ((-2.0, 0.0, 1.0), (10.0, -30.0, 5.0)),
            ((0.05, 0.0, 0.0), (0.0, 200.0, 0.0)),
        ],
    )
    def test_the_scalar_acceleration_matches_the_array_form(
        self, velocity: tuple[float, float, float], spin: tuple[float, float, float] | None
    ) -> None:
        model = BallFlightModel()
        scalar = model._scalar_acceleration(*velocity, spin)
        array = model.acceleration(
            np.asarray(velocity, dtype=float),
            None if spin is None else np.asarray(spin, dtype=float),
        )
        assert scalar == pytest.approx(array)

    @pytest.mark.parametrize("spin", [None, (0.0, 60.0, -3.0)])
    def test_the_scalar_bounce_matches_the_array_form(
        self, spin: tuple[float, float, float] | None
    ) -> None:
        bounce = TableBounce.from_surface(TABLE_TENNIS)
        velocity = (-4.0, 0.5, -3.0)
        scalar_v, scalar_w = bounce.apply_scalar(velocity, spin, radius=0.02)
        array_v, array_w = bounce.apply(
            np.asarray(velocity, dtype=float),
            None if spin is None else np.asarray(spin, dtype=float),
            radius=0.02,
        )
        assert scalar_v == pytest.approx(array_v)
        if spin is None:
            assert scalar_w is None and array_w is None
        else:
            assert scalar_w == pytest.approx(array_w)

    def test_one_prediction_fits_inside_the_control_period(self) -> None:
        """A controller that cannot answer within its own period is not one."""
        import time

        model = BallFlightModel()
        bounce = TableBounce.from_surface(TABLE_TENNIS)
        # The fastest of several batches: a shared CI runner can be descheduled
        # mid-batch, and that measures the runner, not the predictor.
        repeats = 20
        batches = []
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(repeats):
                model.predict_plane_crossing(
                    (1.30, 0.0, 0.95),
                    (-4.0, 0.1, 1.0),
                    plane_x=PLANE_X,
                    angular_velocity=(0.0, 60.0, 0.0),
                    bounce=bounce,
                )
            batches.append((time.perf_counter() - start) / repeats * 1000.0)
        per_call_ms = min(batches)
        # The task runs at 200 Hz, so the budget is 5 ms.  A tighter 2.5 ms
        # bound failed intermittently on GitHub's Python 3.10 runners (1.15 ms
        # locally on 3.10, ~2.5x slower there); the regression this guards
        # against -- the numpy-array integrator -- took 23 ms.
        assert per_call_ms < 5.0, f"{per_call_ms:.2f} ms per prediction"
