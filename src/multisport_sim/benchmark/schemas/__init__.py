"""Versioned JSON Schemas for every file the benchmark publishes.

A schema file is named ``<artifact>.v<N>.json``; its ``$id`` never changes once
released, and a breaking change to a file format gets a new version rather
than an edit.  :func:`validate` checks a document against one (it needs the
optional ``jsonschema`` package, installed with the ``test`` extra).
"""

from __future__ import annotations

import json
from functools import cache
from importlib import resources
from typing import Any

SCHEMAS = {
    "episode-result": "episode-result.v0.json",
    "shot-record": "shot-record.v1.json",
    "shot-bank-manifest": "shot-bank-manifest.v1.json",
    "shot-skill-report": "shot-skill-report.v0.json",
    "baseline-table": "baseline-table.v0.json",
    "submission": "submission.v0.json",
    "cross-task-summary": "cross-task-summary.v0.json",
    "backend-parity": "backend-parity.v0.json",
    "learned-baseline": "learned-baseline.v0.json",
}


@cache
def load(name: str) -> dict[str, Any]:
    """Return one schema document by artifact name (see :data:`SCHEMAS`)."""
    filename = SCHEMAS[name]
    return json.loads(resources.files(__package__).joinpath(filename).read_text("utf-8"))


def validate(name: str, document: Any) -> None:
    """Raise ``jsonschema.ValidationError`` unless ``document`` matches the schema."""
    import jsonschema
    from referencing import Registry, Resource

    registry = Registry().with_resources(
        (load(other)["$id"], Resource.from_contents(load(other))) for other in SCHEMAS
    )
    jsonschema.Draft202012Validator(load(name), registry=registry).validate(document)


__all__ = ["SCHEMAS", "load", "validate"]
