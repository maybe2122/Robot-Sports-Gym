"""Physics foundation and Shot Skill harness for Robot Sports Gym."""

from .isaac_scene import build_isaac_scene_spec
from .specs import BALLS, COURTS, SCENES, TABLE_TENNIS, Sport, TableTennisSpec


def build_model(*args: object, **kwargs: object):
    """Compile a MuJoCo model without making the Isaac path import MuJoCo."""
    from .scene import build_model as _build_model

    return _build_model(*args, **kwargs)


def build_xml(*args: object, **kwargs: object) -> str:
    """Build MJCF lazily so an Isaac-only Python environment remains usable."""
    from .scene import build_xml as _build_xml

    return _build_xml(*args, **kwargs)

__all__ = [
    "BALLS",
    "COURTS",
    "SCENES",
    "TABLE_TENNIS",
    "Sport",
    "TableTennisSpec",
    "build_isaac_scene_spec",
    "build_model",
    "build_xml",
]
__version__ = "0.2.0"
