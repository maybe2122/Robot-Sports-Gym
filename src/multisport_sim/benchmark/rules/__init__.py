"""Sport-specific benchmark rule engines."""

from .table_tennis import (
    TABLE_TENNIS,
    TableTennisJudge,
    TableTennisReturnJudge,
    TableTennisTableSpec,
)

__all__ = [
    "TABLE_TENNIS",
    "TableTennisJudge",
    "TableTennisReturnJudge",
    "TableTennisTableSpec",
]
