"""Quantitative simulation-fidelity benchmarks shared by both backends."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import exp, log, sqrt
from pathlib import Path
from typing import Any, Iterable

from .specs import BALLS, Sport


@dataclass(frozen=True)
class BounceTarget:
    """Reference interval for a first-rebound apex measured under the ball."""

    sport: Sport
    drop_height_m: float
    surface_height_m: float
    ideal_m: float
    lower_m: float
    upper_m: float
    source_kind: str
    source: str
    source_url: str | None

    @property
    def tolerance_m(self) -> float:
        return max(self.ideal_m - self.lower_m, self.upper_m - self.ideal_m)


# Official intervals are used where the governing body publishes the relevant
# drop test. Football and badminton rules do not define an equivalent playing-
# surface test, so their limits are explicitly labelled engineering references.
BOUNCE_TARGETS: dict[Sport, BounceTarget] = {
    Sport.TENNIS: BounceTarget(
        Sport.TENNIS,
        2.54,
        0.0,
        1.41,
        1.35,
        1.47,
        "official",
        "ITF 2025 Technical Booklet, Type 2 ball rebound",
        "https://m.itftennis.com/media/13753/2025-technical-booklet.pdf",
    ),
    Sport.TABLE_TENNIS: BounceTarget(
        Sport.TABLE_TENNIS,
        0.30,
        0.76,
        0.245,
        0.230,
        0.260,
        "official",
        "ITTF Technical Leaflet T1, table bounce/restitution",
        "https://documents.ittf.sport/sites/default/files/public/2024-03/2024_ITTF_Council_documents_EN.pdf",
    ),
    Sport.BASKETBALL: BounceTarget(
        Sport.BASKETBALL,
        1.80,
        0.0,
        1.060,
        1.035,
        1.085,
        "official",
        "FIBA Basketball Equipment, ball rebound",
        "https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2024-official-basketball-rules-and-basketball-equipment.pdf",
    ),
    Sport.FOOTBALL: BounceTarget(
        Sport.FOOTBALL,
        2.00,
        0.0,
        1.00,
        0.85,
        1.15,
        "engineering",
        "Project natural-grass interaction calibration range",
        None,
    ),
    Sport.BADMINTON: BounceTarget(
        Sport.BADMINTON,
        1.80,
        0.0,
        0.0,
        0.0,
        0.06,
        "engineering",
        "Project cork-first low-rebound acceptance limit",
        None,
    ),
}


@dataclass(frozen=True)
class FidelityMetric:
    metric_id: str
    category: str
    sport: str
    measured: float
    ideal: float
    lower: float
    upper: float
    unit: str
    absolute_error: float
    relative_error_percent: float | None
    tolerance_ratio: float
    score: float
    passed: bool
    source_kind: str
    source: str
    source_url: str | None
    derived: dict[str, float]


def score_bounce(sport: Sport, measured_m: float) -> FidelityMetric:
    """Score a rebound measurement; the acceptance boundary maps to 50/100."""
    target = BOUNCE_TARGETS[sport]
    error = abs(measured_m - target.ideal_m)
    tolerance = target.tolerance_m
    ratio = error / tolerance if tolerance > 0.0 else float("inf")
    score = 100.0 * exp(-log(2.0) * ratio**2)
    relative = None if target.ideal_m == 0.0 else 100.0 * error / target.ideal_m
    return FidelityMetric(
        metric_id=f"{sport.value}.first_rebound_height",
        category="contact_dynamics",
        sport=sport.value,
        measured=measured_m,
        ideal=target.ideal_m,
        lower=target.lower_m,
        upper=target.upper_m,
        unit="m",
        absolute_error=error,
        relative_error_percent=relative,
        tolerance_ratio=ratio,
        score=score,
        passed=target.lower_m <= measured_m <= target.upper_m,
        source_kind=target.source_kind,
        source=target.source,
        source_url=target.source_url,
        derived={"effective_restitution": sqrt(max(0.0, measured_m) / target.drop_height_m)},
    )


def build_report(
    backend: str,
    metrics: Iterable[FidelityMetric],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metric_list = list(metrics)
    official = [metric for metric in metric_list if metric.source_kind == "official"]

    def average(items: list[FidelityMetric]) -> float | None:
        return sum(item.score for item in items) / len(items) if items else None

    return {
        "schema_version": 1,
        "suite": "multisport-fidelity-v1",
        "backend": backend,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coverage": {
            "categories": sorted({metric.category for metric in metric_list}),
            "claim": "Scores apply only to the listed measured metrics, not total simulator realism.",
        },
        "metadata": metadata or {},
        "summary": {
            "metric_count": len(metric_list),
            "passed": sum(metric.passed for metric in metric_list),
            "failed": sum(not metric.passed for metric in metric_list),
            "overall_score": average(metric_list),
            "official_metrics_score": average(official),
        },
        "metrics": [asdict(metric) for metric in metric_list],
    }


def report_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    overall = summary["overall_score"]
    official = summary["official_metrics_score"]
    lines = [
        f"# MultiSport fidelity report — {report['backend']}",
        "",
        f"- Suite: `{report['suite']}`",
        "- Coverage: **first-rebound contact dynamics only**",
        f"- Overall score: **{overall:.2f}/100**" if overall is not None else "- Overall score: n/a",
        (
            f"- Official-metric score: **{official:.2f}/100**"
            if official is not None
            else "- Official-metric score: n/a"
        ),
        f"- Passed: **{summary['passed']}/{summary['metric_count']}**",
        "",
        "| Metric | Measured | Reference interval | Error | Tolerance | Score | Result | Basis |",
        "|---|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for metric in report["metrics"]:
        basis = metric["source_kind"]
        if metric["source_url"]:
            basis = f"[{basis}]({metric['source_url']})"
        lines.append(
            "| {metric_id} | {measured:.4f} {unit} | {lower:.4f}–{upper:.4f} {unit} | "
            "{absolute_error:.4f} {unit} | {tolerance_ratio:.3f}× | {score:.2f} | "
            "{result} | {basis} |".format(
                **metric,
                result="PASS" if metric["passed"] else "FAIL",
                basis=basis,
            )
        )
    lines += [
        "",
        "Scoring: `100 × exp(-ln(2) × tolerance_ratio²)`; an acceptance boundary scores 50.",
        "Official and engineering reference metrics are reported separately.",
        "",
    ]
    return "\n".join(lines)


def write_report(report: dict[str, Any], path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in {".md", ".markdown"}:
        output.write_text(report_markdown(report), encoding="utf-8")
    else:
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def aggregate_reports(reports: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Combine single-sport reports from one backend into a suite report."""
    report_list = list(reports)
    if not report_list:
        raise ValueError("at least one fidelity report is required")
    backends = {report["backend"] for report in report_list}
    if len(backends) != 1:
        raise ValueError("cannot aggregate reports from different backends")
    metrics: dict[str, FidelityMetric] = {}
    for report in report_list:
        if report.get("suite") != "multisport-fidelity-v1":
            raise ValueError("incompatible fidelity report suite")
        for raw_metric in report["metrics"]:
            metric = FidelityMetric(**raw_metric)
            metrics[metric.metric_id] = metric
    return build_report(
        backends.pop(),
        metrics.values(),
        metadata={"component_reports": len(report_list)},
    )


def load_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def measure_mujoco_bounce(sport: Sport) -> float:
    """Run a real MuJoCo drop test and return first-apex clearance."""
    import mujoco

    from .simulation import Simulation

    target = BOUNCE_TARGETS[sport]
    simulation = Simulation(sport.value)
    joint_id = mujoco.mj_name2id(
        simulation.model, mujoco.mjtObj.mjOBJ_JOINT, f"{sport.value}_ball_free"
    )
    qpos = simulation.model.jnt_qposadr[joint_id]
    dof = simulation.model.jnt_dofadr[joint_id]
    simulation.data.qpos[qpos + 2] = (
        target.surface_height_m + target.drop_height_m + BALLS[sport].radius
    )
    simulation.data.qvel[dof : dof + 6] = 0.0
    mujoco.mj_forward(simulation.model, simulation.data)

    rising = False
    previous_vz = -1.0
    apex = 0.0
    for _ in range(round(15.0 / simulation.model.opt.timestep)):
        simulation.step()
        vertical_velocity = float(simulation.data.qvel[dof + 2])
        if not rising and previous_vz < 0.0 < vertical_velocity:
            rising = True
        if rising:
            clearance = (
                float(simulation.data.qpos[qpos + 2])
                - target.surface_height_m
                - BALLS[sport].radius
            )
            apex = max(apex, clearance)
        if rising and previous_vz > 0.0 >= vertical_velocity:
            return apex
        previous_vz = vertical_velocity
    raise RuntimeError(f"{sport.value} did not complete a first rebound within 15 seconds")


def run_mujoco_fidelity(sports: Iterable[Sport] = tuple(Sport)) -> dict[str, Any]:
    metrics = [score_bounce(sport, measure_mujoco_bounce(sport)) for sport in sports]
    return build_report(
        "mujoco",
        metrics,
        metadata={"timestep_seconds": 0.001, "integrator": "implicitfast"},
    )
