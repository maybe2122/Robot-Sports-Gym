"""Table-tennis single-shot rules: regulation geometry plus the shared engine.

Task-local coordinates follow the existing simulator scene: the table is centred
at the origin, its length is the x axis, the robot occupies x < 0, the opponent
occupies x > 0, and z points upward.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...specs import TABLE_TENNIS as PHYSICS_TABLE_TENNIS
from ...specs import Sport
from .base import RectangularSurface
from .net_return import NetReturnJudge


@dataclass(frozen=True)
class TableTennisTableSpec(RectangularSurface):
    """Geometry needed by the rule engine, independent of render geometry."""

    length_m: float = PHYSICS_TABLE_TENNIS.table_length
    width_m: float = PHYSICS_TABLE_TENNIS.table_width
    top_height_m: float = PHYSICS_TABLE_TENNIS.table_height
    net_plane_x_m: float = PHYSICS_TABLE_TENNIS.net_plane_x

    @property
    def tabletop_height_tolerance_m(self) -> float:
        """Name retained from the first released task."""
        return self.height_tolerance_m

    @property
    def minimum_tabletop_normal_z(self) -> float:
        """Name retained from the first released task."""
        return self.minimum_normal_z


TABLE_TENNIS = TableTennisTableSpec()


class TableTennisReturnJudge(NetReturnJudge):
    """Judge one table-tennis return against the regulation table."""

    def __init__(
        self,
        *,
        timeout_s: float = 2.0,
        table_spec: TableTennisTableSpec = TABLE_TENNIS,
    ) -> None:
        super().__init__(
            sport=Sport.TABLE_TENNIS.value,
            surface=table_spec,
            timeout_s=timeout_s,
        )

    @property
    def table_spec(self) -> TableTennisTableSpec:
        """Name retained from the first released task; this is the surface."""
        return self.surface  # type: ignore[return-value]


# Short name retained for the terminology used in docs/Benchmark.md.
TableTennisJudge = TableTennisReturnJudge
