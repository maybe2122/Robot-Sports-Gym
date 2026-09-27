"""Where a ball will be, according to a model a policy is allowed to have.

A scripted baseline has to predict the interception point before it can swing at
it.  The obvious predictor -- constant horizontal speed and a parabola in ``z``
-- is wrong by more than a blade radius on this bank, because the simulation
flies the ball through quadratic drag and a Magnus force and the predictor does
not.  Measured on ``table_tennis/return-v1``, the ballistic predictor leaves the
blade a median 8-11 cm from the ball, against a blade radius of 7.5 cm: the arm
reaches the right neighbourhood at the right time and still misses.

So this module integrates the *same* force law the backend applies, as the
controller's own model.  That is a legitimate thing for a baseline to know: a
real policy with a camera and a physics model could compute it too, and nothing
here reads the running simulation.  What it is not allowed to be is a lookup of
the answer, which is why it takes an estimated state and integrates from it --
a noisy or delayed estimate produces a correspondingly wrong prediction, which
is exactly what should happen on the perturbed levels and the vision track.

Spin is optional and that is the point.  The state track observes the ball's
angular velocity, so it gets the Magnus term; the vision track's tracker
recovers position and velocity only, so it predicts without one and eats the
error.  See :class:`BallFlightModel.predict_plane_crossing`.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import TYPE_CHECKING

import numpy as np

from ..specs import AIR_DENSITY, BALLS, Sport

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Sequence

GRAVITY = 9.81

# The lift coefficient bound and the spin scale are the backend's, not new
# numbers: a predictor that used a different Magnus law would be modelling a
# different ball.  They live in ``multisport_sim.physics`` for the simulation
# side; duplicating the two constants here keeps this module free of a MuJoCo
# import, and the shared-value test in tests/test_ball_flight.py fails if the
# two ever drift apart.
MAGNUS_SCALE = 0.20
MAGNUS_LIFT_CEILING = 0.35


@dataclass(frozen=True)
class PlaneCrossing:
    """Where and when the ball is predicted to reach a vertical plane."""

    time_s: float
    y: float
    z: float
    speed_mps: float
    bounces: int = 0


@dataclass(frozen=True)
class TableBounce:
    """The table contact, as an impulse law a predictor can integrate through.

    Free flight is not enough on this task.  On ``table_tennis/return-v1`` the
    ball's last bounce before the strike plane happens a median 142 ms (L2) to
    84 ms (L5) before the crossing, and the swing has to commit around 160 ms
    out -- so on 55-85% of the shots above L1 the swing is planned *across* a
    bounce.  A predictor that flies straight through the table is wrong by a
    median half metre there, which is why the arm arrives 8-11 cm from a ball it
    had ample time to reach.

    The coefficients are measured, not assumed: 595 bounces sampled from the
    training split, velocities read 5 ms clear of the contact interval at either
    end so the soft contact is never sampled mid-impact.  The tangential and
    spin maps fit at R^2 = 0.999; the normal coefficient has a real spread
    (median 0.915, 10th-90th 0.83-1.01) because contact stiffness varies with
    impact speed, and the median is used.

    Nothing here reads the simulation: these are constants a policy could fit
    from its own observations of the same table, which is the standard the
    scripted baseline is held to.
    """

    surface_z: float
    half_length_x: float
    half_width_y: float
    net_plane_x: float = 0.0
    center_y: float = 0.0
    restitution: float = 0.915
    tangential_retention: float = 0.990
    tangential_from_spin: float = 0.0093
    spin_retention: float = 0.971
    spin_from_tangential: float = 0.317

    @classmethod
    def from_surface(cls, surface: object, **overrides: float) -> TableBounce:
        """Build from a :class:`~...rules.base.RectangularSurface`-shaped spec.

        Taking the geometry from the rule engine's own surface means the
        predictor and the judge can never disagree about where the table is.
        """
        return cls(
            surface_z=float(surface.top_height_m),
            half_length_x=float(surface.length_m) / 2.0,
            half_width_y=float(surface.width_m) / 2.0,
            net_plane_x=float(getattr(surface, "net_plane_x_m", 0.0)),
            center_y=float(getattr(surface, "center_y_m", 0.0)),
            **overrides,
        )

    def covers(self, x: float, y: float) -> bool:
        """Is ``(x, y)`` over the table top?"""
        return (
            abs(x - self.net_plane_x) <= self.half_length_x
            and abs(y - self.center_y) <= self.half_width_y
        )

    def apply(
        self,
        velocity: np.ndarray,
        angular_velocity: np.ndarray | None,
        *,
        radius: float,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Post-impact velocity and spin, in the same frame.

        ``tangential_from_spin`` multiplies ``radius * omega`` -- the surface
        speed of the ball -- because that, not the spin rate, is what the
        contact sees.
        """
        out = velocity.copy()
        out[2] = -self.restitution * velocity[2]
        if angular_velocity is None:
            out[0] = self.tangential_retention * velocity[0]
            out[1] = self.tangential_retention * velocity[1]
            return out, None
        coupling = self.tangential_from_spin * radius
        out[0] = self.tangential_retention * velocity[0] + coupling * angular_velocity[1]
        out[1] = self.tangential_retention * velocity[1] - coupling * angular_velocity[0]
        spin = angular_velocity.copy()
        spin[1] = self.spin_retention * angular_velocity[1] + self.spin_from_tangential * velocity[0]
        spin[0] = self.spin_retention * angular_velocity[0] - self.spin_from_tangential * velocity[1]
        return out, spin

    def apply_scalar(
        self,
        velocity: tuple[float, float, float],
        angular_velocity: tuple[float, float, float] | None,
        *,
        radius: float,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float] | None]:
        """:meth:`apply` on plain floats, for the integrator's inner loop."""
        vx, vy, vz = velocity
        out_z = -self.restitution * vz
        if angular_velocity is None:
            return (
                self.tangential_retention * vx,
                self.tangential_retention * vy,
                out_z,
            ), None
        wx, wy, wz = angular_velocity
        coupling = self.tangential_from_spin * radius
        return (
            self.tangential_retention * vx + coupling * wy,
            self.tangential_retention * vy - coupling * wx,
            out_z,
        ), (
            self.spin_retention * wx - self.spin_from_tangential * vy,
            self.spin_retention * wy + self.spin_from_tangential * vx,
            wz,
        )


class BallFlightModel:
    """Integrate a ball's free flight under gravity, drag and Magnus lift.

    The integration is fixed-step RK4.  At the 2 ms default it costs about
    30 microseconds to predict a full second of flight, which is affordable at a
    200 Hz control rate and leaves the prediction error dominated by the state
    estimate rather than by the integrator.
    """

    def __init__(
        self,
        sport: Sport = Sport.TABLE_TENNIS,
        *,
        air_density: float = AIR_DENSITY,
        timestep_s: float = 0.002,
    ) -> None:
        spec = BALLS[sport]
        if timestep_s <= 0.0 or not np.isfinite(timestep_s):
            raise ValueError("timestep_s must be positive and finite")
        self.sport = sport
        self.mass = float(spec.mass)
        self.radius = float(spec.radius)
        self.timestep_s = float(timestep_s)
        # Everything the force law needs, pre-multiplied: the integrator runs
        # inside the control loop and should not be recomputing constants.
        self._drag_factor = (
            0.5 * float(air_density) * float(spec.drag_coefficient) * float(spec.cross_section)
        )
        self._lift_factor = 0.5 * float(air_density) * float(spec.cross_section)

    def acceleration(
        self, velocity: np.ndarray, angular_velocity: np.ndarray | None
    ) -> np.ndarray:
        """Gravity plus drag plus Magnus lift, in m/s^2."""
        acceleration = np.array([0.0, 0.0, -GRAVITY])
        speed = float(np.linalg.norm(velocity))
        if speed < 1e-9:
            return acceleration
        acceleration -= (self._drag_factor * speed / self.mass) * velocity
        if angular_velocity is None or speed <= 0.1:
            # The backend applies no Magnus force below 0.1 m/s either; matching
            # the threshold matters more than the force it would have produced.
            return acceleration
        spin_axis = np.cross(angular_velocity, velocity)
        spin_norm = float(np.linalg.norm(spin_axis))
        if spin_norm < 1e-9:
            return acceleration
        lift_coefficient = min(
            MAGNUS_LIFT_CEILING,
            MAGNUS_SCALE * self.radius * float(np.linalg.norm(angular_velocity)) / speed,
        )
        acceleration += (
            self._lift_factor * speed**2 * lift_coefficient / self.mass
        ) * (spin_axis / spin_norm)
        return acceleration

    def _scalar_acceleration(
        self,
        vx: float,
        vy: float,
        vz: float,
        spin: tuple[float, float, float] | None,
    ) -> tuple[float, float, float]:
        """:meth:`acceleration` on three floats instead of an array.

        This is the inner loop, and it is written on scalars for one measured
        reason: with three-element numpy arrays the dispatch overhead dominates
        and a single prediction costs 23 ms against a 5 ms control period, so
        the controller could not run in real time and ``inference_latency_ms``
        would say so.  The scalar form is the same arithmetic and is checked
        against the array form in tests/test_ball_flight.py.
        """
        speed = sqrt(vx * vx + vy * vy + vz * vz)
        if speed < 1e-9:
            return 0.0, 0.0, -GRAVITY
        drag = self._drag_factor * speed / self.mass
        ax, ay, az = -drag * vx, -drag * vy, -GRAVITY - drag * vz
        if spin is None or speed <= 0.1:
            # The backend applies no Magnus force below 0.1 m/s either; matching
            # the threshold matters more than the force it would have produced.
            return ax, ay, az
        wx, wy, wz = spin
        # spin_axis = omega x v
        sx = wy * vz - wz * vy
        sy = wz * vx - wx * vz
        sz = wx * vy - wy * vx
        spin_norm = sqrt(sx * sx + sy * sy + sz * sz)
        if spin_norm < 1e-9:
            return ax, ay, az
        rate = sqrt(wx * wx + wy * wy + wz * wz)
        lift = min(MAGNUS_LIFT_CEILING, MAGNUS_SCALE * self.radius * rate / speed)
        scale = self._lift_factor * speed * speed * lift / (self.mass * spin_norm)
        return ax + scale * sx, ay + scale * sy, az + scale * sz

    def fly_until(
        self,
        position: Sequence[float],
        velocity: Sequence[float],
        event: Callable[[tuple[float, float, float], tuple[float, float, float]], float],
        *,
        horizon_s: float = 3.0,
        floor_z: float | None = None,
    ) -> tuple[float, tuple[float, float, float], tuple[float, float, float]] | None:
        """Integrate free flight until ``event(position, velocity)`` changes sign.

        ``event`` is evaluated after every step; the first step at which it
        goes from negative to non-negative is the event, located by linear
        interpolation inside that step.  Returns ``(time, position, velocity)``
        at the event, or ``None`` if it does not happen within ``horizon_s`` or
        the ball's centre first drops below ``floor_z``.

        The launch tasks' planners ask "where does this ball cross the goal
        line" or "where does it come down through the rim's plane"; this is the
        one integrator both use, with the backend's force law and no spin.
        """
        px, py, pz = (float(value) for value in position)
        vx, vy, vz = (float(value) for value in velocity)
        dt = self.timestep_s
        half, sixth = 0.5 * dt, dt / 6.0
        elapsed = 0.0
        previous_value = event((px, py, pz), (vx, vy, vz))
        while elapsed < horizon_s:
            ppx, ppy, ppz, pvx, pvy, pvz = px, py, pz, vx, vy, vz
            a1x, a1y, a1z = self._scalar_acceleration(vx, vy, vz, None)
            b1x, b1y, b1z = vx + half * a1x, vy + half * a1y, vz + half * a1z
            a2x, a2y, a2z = self._scalar_acceleration(b1x, b1y, b1z, None)
            b2x, b2y, b2z = vx + half * a2x, vy + half * a2y, vz + half * a2z
            a3x, a3y, a3z = self._scalar_acceleration(b2x, b2y, b2z, None)
            b3x, b3y, b3z = vx + dt * a3x, vy + dt * a3y, vz + dt * a3z
            a4x, a4y, a4z = self._scalar_acceleration(b3x, b3y, b3z, None)
            px += sixth * (vx + 2.0 * b1x + 2.0 * b2x + b3x)
            py += sixth * (vy + 2.0 * b1y + 2.0 * b2y + b3y)
            pz += sixth * (vz + 2.0 * b1z + 2.0 * b2z + b3z)
            vx += sixth * (a1x + 2.0 * a2x + 2.0 * a3x + a4x)
            vy += sixth * (a1y + 2.0 * a2y + 2.0 * a3y + a4y)
            vz += sixth * (a1z + 2.0 * a2z + 2.0 * a3z + a4z)
            elapsed += dt
            if floor_z is not None and pz < floor_z:
                return None
            value = event((px, py, pz), (vx, vy, vz))
            if previous_value < 0.0 <= value:
                span = value - previous_value
                alpha = 1.0 if span <= 0.0 else -previous_value / span
                return (
                    elapsed - dt + alpha * dt,
                    (ppx + alpha * (px - ppx), ppy + alpha * (py - ppy), ppz + alpha * (pz - ppz)),
                    (pvx + alpha * (vx - pvx), pvy + alpha * (vy - pvy), pvz + alpha * (vz - pvz)),
                )
            previous_value = value
        return None

    def predict_plane_crossing(
        self,
        position: Sequence[float],
        velocity: Sequence[float],
        *,
        plane_x: float,
        angular_velocity: Sequence[float] | None = None,
        horizon_s: float = 2.0,
        floor_z: float = 0.0,
        bounce: TableBounce | None = None,
        max_bounces: int = 2,
    ) -> PlaneCrossing | None:
        """First crossing of ``x = plane_x`` while travelling in ``-x``.

        Returns ``None`` when the ball is already past the plane, is not moving
        toward it, hits ``floor_z`` first, or does not arrive within
        ``horizon_s`` -- every one of which is a case where a swing should not
        be planned at all, and none of which should be reported as a crossing at
        the horizon's edge.

        ``angular_velocity`` may be omitted, and then the prediction carries no
        Magnus term.  That is the honest thing for a caller who cannot observe
        spin: the resulting error is the cost of not seeing it, and hiding it
        behind an assumed spin would be a worse answer than a wrong one.
        """
        try:
            px, py, pz = (float(value) for value in position)
            vx, vy, vz = (float(value) for value in velocity)
        except (TypeError, ValueError) as exc:
            raise ValueError("position and velocity must be three-vectors") from exc
        if not all(isfinite(value) for value in (px, py, pz, vx, vy, vz)):
            return None
        spin: tuple[float, float, float] | None = None
        if angular_velocity is not None:
            try:
                wx, wy, wz = (float(value) for value in angular_velocity)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "angular_velocity must be a finite three-vector"
                ) from exc
            if not all(isfinite(value) for value in (wx, wy, wz)):
                raise ValueError("angular_velocity must be a finite three-vector")
            spin = (wx, wy, wz)
        if px <= plane_x or vx >= -1e-6:
            return None

        dt = self.timestep_s
        half, sixth = 0.5 * dt, dt / 6.0
        elapsed = 0.0
        bounces = 0
        contact_z = floor_z if bounce is None else bounce.surface_z + self.radius
        while elapsed < horizon_s:
            ppx, ppy, ppz = px, py, pz
            pvx, pvy, pvz = vx, vy, vz
            previous_t = elapsed

            a1x, a1y, a1z = self._scalar_acceleration(vx, vy, vz, spin)
            b1x, b1y, b1z = vx + half * a1x, vy + half * a1y, vz + half * a1z
            a2x, a2y, a2z = self._scalar_acceleration(b1x, b1y, b1z, spin)
            b2x, b2y, b2z = vx + half * a2x, vy + half * a2y, vz + half * a2z
            a3x, a3y, a3z = self._scalar_acceleration(b2x, b2y, b2z, spin)
            b3x, b3y, b3z = vx + dt * a3x, vy + dt * a3y, vz + dt * a3z
            a4x, a4y, a4z = self._scalar_acceleration(b3x, b3y, b3z, spin)

            px += sixth * (vx + 2.0 * b1x + 2.0 * b2x + b3x)
            py += sixth * (vy + 2.0 * b1y + 2.0 * b2y + b3y)
            pz += sixth * (vz + 2.0 * b1z + 2.0 * b2z + b3z)
            vx += sixth * (a1x + 2.0 * a2x + 2.0 * a3x + a4x)
            vy += sixth * (a1y + 2.0 * a2y + 2.0 * a3y + a4y)
            vz += sixth * (a1z + 2.0 * a2z + 2.0 * a3z + a4z)
            elapsed += dt

            if (
                bounce is not None
                and bounces < max_bounces
                and vz < 0.0
                and pz <= contact_z
                and bounce.covers(px, py)
            ):
                # Rewind to the surface before reflecting, so the bounce is not
                # displaced by up to one integration step of free fall.
                drop = ppz - pz
                alpha = 0.0 if drop <= 0.0 else (ppz - contact_z) / drop
                alpha = min(max(alpha, 0.0), 1.0)
                px, py, pz = (
                    ppx + alpha * (px - ppx),
                    ppy + alpha * (py - ppy),
                    ppz + alpha * (pz - ppz),
                )
                vx, vy, vz = (
                    pvx + alpha * (vx - pvx),
                    pvy + alpha * (vy - pvy),
                    pvz + alpha * (vz - pvz),
                )
                elapsed = previous_t + alpha * dt
                reflected, spin = bounce.apply_scalar((vx, vy, vz), spin, radius=self.radius)
                vx, vy, vz = reflected
                bounces += 1
                continue
            if pz < floor_z:
                return None
            if px <= plane_x:
                # Linear interpolation inside one integration step: at 5 m/s
                # that is a centimetre of travel, and the residual is far below
                # the estimate error the prediction is built on.
                span = ppx - px
                alpha = 0.0 if span <= 0.0 else (ppx - plane_x) / span
                alpha = min(max(alpha, 0.0), 1.0)
                sy_ = ppy + alpha * (py - ppy)
                sz_ = ppz + alpha * (pz - ppz)
                ux = pvx + alpha * (vx - pvx)
                uy = pvy + alpha * (vy - pvy)
                uz = pvz + alpha * (vz - pvz)
                return PlaneCrossing(
                    time_s=previous_t + alpha * dt,
                    y=sy_,
                    z=sz_,
                    speed_mps=sqrt(ux * ux + uy * uy + uz * uz),
                    bounces=bounces,
                )
        return None


__all__ = ["GRAVITY", "BallFlightModel", "PlaneCrossing", "TableBounce"]
