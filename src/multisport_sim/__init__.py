"""Physics foundation and Shot Skill harness for Robot Sports Gym."""

from .isaac_scene import build_isaac_scene_spec
from .scene import build_model, build_xml
from .specs import BALLS, COURTS, SCENES, TABLE_TENNIS, Sport, TableTennisSpec

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
