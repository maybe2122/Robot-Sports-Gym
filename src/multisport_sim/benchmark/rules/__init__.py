"""Sport-specific benchmark rule engines built on one shared contract."""

from .base import PlayingSurface, RectangularSurface, ShotJudge
from .net_return import NetReturnJudge
from .table_tennis import (
    TABLE_TENNIS,
    TableTennisJudge,
    TableTennisReturnJudge,
    TableTennisTableSpec,
)

__all__ = [
    "TABLE_TENNIS",
    "NetReturnJudge",
    "PlayingSurface",
    "RectangularSurface",
    "ShotJudge",
    "TableTennisJudge",
    "TableTennisReturnJudge",
    "TableTennisTableSpec",
]
