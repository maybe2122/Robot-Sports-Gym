"""Resolution and provenance of third-party robot assets.

The benchmark does not vendor robot models.  Copying an upstream asset into
this repository would fork it silently: the copy stops receiving fixes, its
licence text drifts from the original, and a result can no longer say which
version of the model produced it.  Instead every asset is declared here with
its upstream URL, pinned version, licence and any local modification, and is
loaded from a checkout the user provides.

Resolution order for a checkout root, first hit wins:

1. the ``MULTISPORT_MENAGERIE_PATH`` environment variable;
2. the ``MUJOCO_MENAGERIE_PATH`` environment variable, which several other
   MuJoCo projects already set;
3. ``~/mujoco_menagerie``.

When no checkout is found, :func:`require_asset` raises
:class:`AssetUnavailableError` with the exact command that fixes it, and tests
that need the asset skip rather than fail.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MENAGERIE_ENV_VARS = ("MULTISPORT_MENAGERIE_PATH", "MUJOCO_MENAGERIE_PATH")
MENAGERIE_DEFAULT = Path.home() / "mujoco_menagerie"
MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie"


class AssetUnavailableError(RuntimeError):
    """Raised when a declared asset is not present on this machine."""


@dataclass(frozen=True)
class AssetSource:
    """One third-party asset: where it comes from and what we changed.

    ``modifications`` is the honest part of the record.  The benchmark attaches
    a paddle and mounts the arm on a pedestal; neither edits the upstream files,
    but both change the model a result was produced with, so both are listed.
    """

    asset_id: str
    relative_path: str
    upstream_url: str
    version: str
    license_id: str
    license_path: str
    modifications: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "asset_id",
            "relative_path",
            "upstream_url",
            "version",
            "license_id",
            "license_path",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        modifications = tuple(self.modifications)
        if any(not isinstance(item, str) or not item.strip() for item in modifications):
            raise ValueError("modifications must be non-empty strings")
        object.__setattr__(self, "modifications", modifications)

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "relative_path": self.relative_path,
            "upstream_url": self.upstream_url,
            "version": self.version,
            "license": self.license_id,
            "license_path": self.license_path,
            "modifications": list(self.modifications),
        }


FRANKA_PANDA = AssetSource(
    asset_id="franka_emika_panda",
    relative_path="franka_emika_panda/panda_nohand.xml",
    upstream_url=f"{MENAGERIE_URL}/tree/main/franka_emika_panda",
    # Menagerie has no release tags; the model's own changelog is the version.
    version="menagerie franka_emika_panda CHANGELOG 2024-06-03",
    license_id="Apache-2.0",
    license_path="franka_emika_panda/LICENSE",
    modifications=(
        (
            "loaded through mujoco.MjSpec and attached to the table-tennis scene "
            "with the joint/actuator prefix 'rb_'; upstream files are not edited"
        ),
        "a benchmark paddle body is added as a child of the 'attachment' body",
        "a fixed pedestal geom is added under the arm base",
        (
            "the parent scene's impratio=1 is kept instead of the model's "
            "impratio=10, because one scene cannot hold two values"
        ),
    ),
)
"""7-DoF arm used by the table-tennis, tennis, badminton and basketball tasks.

Chosen because MuJoCo Menagerie and Isaac Lab both ship it under a clear
licence, which is what lets the same robot run the same task on both backends.
"""

UNITREE_G1 = AssetSource(
    asset_id="unitree_g1",
    relative_path="unitree_g1/g1.xml",
    upstream_url=f"{MENAGERIE_URL}/tree/main/unitree_g1",
    version="menagerie unitree_g1 CHANGELOG 2025-08-07",
    license_id="BSD-3-Clause",
    license_path="unitree_g1/LICENSE",
    modifications=(
        (
            "loaded through mujoco.MjSpec and attached to the table-tennis scene "
            "with the joint/actuator prefix 'g1_'; upstream files are not edited"
        ),
        (
            "the pelvis free joint is deleted, fixing the base: this task scores "
            "a return, not balance, and a free-standing humanoid would be scored "
            "on staying upright"
        ),
        (
            "the 'stand' keyframe is dropped because its qpos still counts the "
            "deleted free joint; its pose is carried in robots/g1.py STAND_QPOS"
        ),
        "a benchmark paddle body is added as a child of 'right_wrist_yaw_link'",
        (
            "the parent scene's impratio=1 is kept instead of the model's own "
            "value, because one compiled model holds one impratio"
        ),
        (
            "joint speed limits are NOT from this asset: it declares "
            "actuatorfrcrange but no velocity bound, so the benchmark supplies a "
            "uniform placeholder and marks it unverified in every report"
        ),
    ),
)
"""29-DoF humanoid, used fixed-base as the table-tennis task's second embodiment.

It exists to test that the task is robot-agnostic.  Its ten commanded joints and
39-number observation share every field *name* with the Panda's seven and 33, so
a policy written against the published field names transfers and a policy
written against offsets does not.
"""

ASSETS: dict[str, AssetSource] = {
    FRANKA_PANDA.asset_id: FRANKA_PANDA,
    UNITREE_G1.asset_id: UNITREE_G1,
}


def menagerie_root() -> Path | None:
    """Return the first usable Menagerie checkout, or ``None``.

    A directory only counts when it actually contains models, so an empty or
    stale environment variable falls through to the next candidate instead of
    producing a confusing missing-file error later.
    """
    candidates: list[Path] = []
    for variable in MENAGERIE_ENV_VARS:
        value = os.environ.get(variable)
        if value and value.strip():
            candidates.append(Path(value).expanduser())
    candidates.append(MENAGERIE_DEFAULT)
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*/*.xml")):
            return candidate
    return None


def resolve_asset(source: AssetSource = FRANKA_PANDA) -> Path | None:
    """Return the asset's model file if it is present, otherwise ``None``."""
    if not isinstance(source, AssetSource):
        raise TypeError("source must be an AssetSource")
    root = menagerie_root()
    if root is None:
        return None
    path = root / source.relative_path
    return path if path.is_file() else None


def asset_available(source: AssetSource = FRANKA_PANDA) -> bool:
    """Whether this asset can be loaded; the predicate tests skip on."""
    return resolve_asset(source) is not None


def missing_asset_message(source: AssetSource) -> str:
    """Explain how to obtain one asset, naming the exact commands."""
    return (
        f"robot asset {source.asset_id!r} is not available.\n"
        f"It is not vendored: {source.upstream_url} ({source.license_id}).\n"
        "Obtain it with a sparse checkout:\n"
        f"    git clone --filter=blob:none --sparse {MENAGERIE_URL} ~/mujoco_menagerie\n"
        f"    git -C ~/mujoco_menagerie sparse-checkout add {source.asset_id}\n"
        "Then point the benchmark at it if it lives elsewhere:\n"
        "    export MULTISPORT_MENAGERIE_PATH=/path/to/mujoco_menagerie"
    )


def require_asset(source: AssetSource = FRANKA_PANDA) -> Path:
    """Return the asset's model file, raising with instructions if absent."""
    path = resolve_asset(source)
    if path is None:
        raise AssetUnavailableError(missing_asset_message(source))
    return path


def license_text(source: AssetSource = FRANKA_PANDA) -> str:
    """Read the asset's own licence file from the checkout."""
    root = menagerie_root()
    if root is None:
        raise AssetUnavailableError(missing_asset_message(source))
    path = root / source.license_path
    if not path.is_file():
        raise AssetUnavailableError(
            f"{source.asset_id} is present but its licence file {source.license_path} is not"
        )
    return path.read_text(encoding="utf-8")


def license_manifest() -> list[dict[str, Any]]:
    """Every declared asset with its provenance, for reports and documentation."""
    return [
        {**source.to_dict(), "available": asset_available(source)}
        for _, source in sorted(ASSETS.items())
    ]


__all__ = [
    "ASSETS",
    "FRANKA_PANDA",
    "MENAGERIE_ENV_VARS",
    "MENAGERIE_URL",
    "AssetSource",
    "AssetUnavailableError",
    "asset_available",
    "license_manifest",
    "license_text",
    "menagerie_root",
    "missing_asset_message",
    "require_asset",
    "resolve_asset",
]
