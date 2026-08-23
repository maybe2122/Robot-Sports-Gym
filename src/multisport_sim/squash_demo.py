"""Isaac-independent validation for the committed squash score demonstration."""

from __future__ import annotations

import argparse
import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path

EXPECTED_EVENT_KINDS = (
    "serve",
    "front_wall",
    "floor_bounce",
    "racket_hit",
    "front_wall",
    "floor_bounce",
    "floor_bounce",
    "point",
)
EXPECTED_EVENT_PLAYERS = ("A", "A", "A", "B", "B", "B", "B", "B")


class SquashDemoValidationError(ValueError):
    """Raised when a report or GIF does not prove the expected scoring rally."""


@dataclass(frozen=True)
class GifInfo:
    width: int
    height: int
    frames: int
    duration_ms: int
    bytes: int


def validate_score_report(report: object) -> dict[str, object]:
    """Validate and summarize the canonical A-serve/B-second-bounce rally."""
    root = _mapping(report, "report")
    if root.get("schema") != "squash-score-demo-v1":
        raise SquashDemoValidationError("report schema must be squash-score-demo-v1")
    if root.get("complete") is not True:
        raise SquashDemoValidationError("report must mark the rally complete")
    if root.get("score") != {"A": 0, "B": 1}:
        raise SquashDemoValidationError("final score must be A 0-1 B")
    if root.get("server") != "B":
        raise SquashDemoValidationError("point winner B must become the next server")

    rally = _mapping(root.get("rally"), "rally")
    expected_rally = {
        "number": 1,
        "server": "A",
        "winner": "B",
        "shots": 2,
        "reason": "second_bounce",
    }
    if rally != expected_rally:
        raise SquashDemoValidationError(f"unexpected rally result: {dict(rally)!r}")

    events = root.get("events")
    if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
        raise SquashDemoValidationError("events must be an ordered sequence")
    if len(events) != len(EXPECTED_EVENT_KINDS):
        raise SquashDemoValidationError(
            f"expected {len(EXPECTED_EVENT_KINDS)} events, received {len(events)}"
        )

    times: list[float] = []
    floor_details: list[str] = []
    for index, (raw_event, kind, player) in enumerate(
        zip(events, EXPECTED_EVENT_KINDS, EXPECTED_EVENT_PLAYERS, strict=True),
        start=1,
    ):
        event = _mapping(raw_event, f"event {index}")
        if event.get("number") != index:
            raise SquashDemoValidationError(f"event {index} has an invalid sequence number")
        if event.get("kind") != kind or event.get("player") != player:
            raise SquashDemoValidationError(
                f"event {index} must be {kind} by player {player}"
            )
        raw_time = event.get("time_s")
        if isinstance(raw_time, bool) or not isinstance(raw_time, (int, float)):
            raise SquashDemoValidationError(f"event {index} time_s must be finite")
        time_s = float(raw_time)
        if not isfinite(time_s) or time_s < 0.0:
            raise SquashDemoValidationError(f"event {index} time_s must be finite and non-negative")
        if times and time_s < times[-1]:
            raise SquashDemoValidationError("event times must be monotonic")
        times.append(time_s)
        if kind == "floor_bounce":
            floor_details.append(str(event.get("detail")))

    if times[0] != 0.0:
        raise SquashDemoValidationError("serve must occur at time zero")
    if floor_details != ["1", "1", "2"]:
        raise SquashDemoValidationError("floor-bounce counters must be 1, 1, 2")
    if times[-1] != times[-2]:
        raise SquashDemoValidationError("point must be awarded at the second-bounce time")

    duration = root.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise SquashDemoValidationError("duration_s must be finite")
    if not isfinite(float(duration)) or float(duration) < times[-1]:
        raise SquashDemoValidationError("duration_s must include the point event")
    return {
        "schema": root["schema"],
        "score": root["score"],
        "winner": rally["winner"],
        "reason": rally["reason"],
        "events": len(events),
        "point_time_s": times[-1],
    }


def inspect_gif(path: str | Path) -> GifInfo:
    """Parse GIF container metadata without depending on Pillow."""
    source = Path(path)
    data = source.read_bytes()

    def need(offset: int, size: int, context: str) -> None:
        if offset < 0 or size < 0 or offset + size > len(data):
            raise SquashDemoValidationError(f"truncated GIF while reading {context}")

    need(0, 13, "header")
    if data[:6] not in (b"GIF87a", b"GIF89a"):
        raise SquashDemoValidationError("asset is not a GIF87a/GIF89a file")
    width, height, packed = struct.unpack_from("<HHB", data, 6)
    cursor = 13
    if packed & 0x80:
        cursor += 3 * 2 ** ((packed & 0x07) + 1)
        need(0, cursor, "global color table")

    def skip_sub_blocks(offset: int) -> int:
        while True:
            need(offset, 1, "sub-block size")
            size = data[offset]
            offset += 1
            if size == 0:
                return offset
            need(offset, size, "sub-block data")
            offset += size

    frames = 0
    duration_ms = 0
    while True:
        need(cursor, 1, "block marker")
        marker = data[cursor]
        cursor += 1
        if marker == 0x3B:
            break
        if marker == 0x21:
            need(cursor, 1, "extension label")
            label = data[cursor]
            cursor += 1
            if label == 0xF9:
                need(cursor, 6, "graphic control extension")
                block_size = data[cursor]
                if block_size != 4 or data[cursor + 5] != 0:
                    raise SquashDemoValidationError("invalid graphic control extension")
                duration_ms += struct.unpack_from("<H", data, cursor + 2)[0] * 10
                cursor += 6
            else:
                cursor = skip_sub_blocks(cursor)
            continue
        if marker != 0x2C:
            raise SquashDemoValidationError(f"unexpected GIF block marker 0x{marker:02x}")
        need(cursor, 9, "image descriptor")
        frames += 1
        image_packed = data[cursor + 8]
        cursor += 9
        if image_packed & 0x80:
            cursor += 3 * 2 ** ((image_packed & 0x07) + 1)
            need(0, cursor, "local color table")
        need(cursor, 1, "LZW code size")
        cursor += 1
        cursor = skip_sub_blocks(cursor)
    return GifInfo(width, height, frames, duration_ms, len(data))


def validate_gif(path: str | Path) -> GifInfo:
    info = inspect_gif(path)
    if (info.width, info.height) != (640, 360):
        raise SquashDemoValidationError("GIF dimensions must be 640x360")
    if info.frames < 40:
        raise SquashDemoValidationError("GIF must contain at least 40 frames")
    if info.duration_ms < 4_000:
        raise SquashDemoValidationError("GIF must run for at least four seconds")
    if info.bytes < 1_000_000:
        raise SquashDemoValidationError("GIF is unexpectedly small")
    return info


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SquashDemoValidationError(f"{field} must be an object")
    return value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--report", required=True, help="squash-score-demo-v1 JSON report")
    result.add_argument("--gif", required=True, help="annotated squash demonstration GIF")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    with Path(args.report).open(encoding="utf-8") as stream:
        report_summary = validate_score_report(json.load(stream))
    gif_info = validate_gif(args.gif)
    print(
        json.dumps(
            {"valid": True, "report": report_summary, "gif": asdict(gif_info)},
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
