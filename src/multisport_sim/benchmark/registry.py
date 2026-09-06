"""Registry of benchmark tasks.

Every task publishes one entry: its shared configuration, the judge that scores
it, the MuJoCo backend that runs it, and the Gymnasium entry point that wraps
it.  The CLI, the environment registration and the reporting layer all read
this registry, so adding a sport means adding an entry -- not editing each of
them in turn.

Factories are callables rather than imported objects so that registering a task
never drags a simulator import into a process that only wants to list tasks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass

from .rules.base import ShotJudge
from .task_config import ShotTaskConfig

JudgeFactory = Callable[[ShotTaskConfig], ShotJudge]
BackendFactory = Callable[[ShotTaskConfig], object]


@dataclass(frozen=True)
class TaskEntry:
    """One registered benchmark task and everything needed to run it."""

    config: ShotTaskConfig
    judge_factory: JudgeFactory
    backend_factory: BackendFactory
    env_entry_point: str
    # A second environment for the same task on the vision track.  It is not a
    # second *task*: same judge, same bank, same thresholds, and its results
    # belong in the same table -- only what the policy may see differs.
    vision_env_entry_point: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.config, ShotTaskConfig):
            raise TypeError("config must be a ShotTaskConfig")
        for name in ("judge_factory", "backend_factory"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")
        if ":" not in self.env_entry_point:
            raise ValueError("env_entry_point must look like 'module:Class'")
        if self.vision_env_entry_point is not None:
            if ":" not in self.vision_env_entry_point:
                raise ValueError("vision_env_entry_point must look like 'module:Class'")
            if getattr(self.config, "vision_env_id", None) is None:
                raise ValueError(
                    "a task with a vision environment must declare vision_env_id"
                )

    @property
    def task_id(self) -> str:
        return self.config.task_id

    def make_judge(self, config: ShotTaskConfig | None = None) -> ShotJudge:
        return self.judge_factory(config or self.config)

    def make_backend(self, config: ShotTaskConfig | None = None) -> object:
        return self.backend_factory(config or self.config)


_TASKS: dict[str, TaskEntry] = {}


def register_task(entry: TaskEntry, *, replace: bool = False) -> TaskEntry:
    """Register a task, refusing to shadow a different task with the same id."""
    if not isinstance(entry, TaskEntry):
        raise TypeError("entry must be a TaskEntry")
    existing = _TASKS.get(entry.task_id)
    if existing is not None and not replace and existing != entry:
        raise ValueError(f"task {entry.task_id!r} is already registered")
    _TASKS[entry.task_id] = entry
    return entry


def get_task(task_id: str) -> TaskEntry:
    try:
        return _TASKS[task_id]
    except KeyError:
        raise KeyError(
            f"unknown task {task_id!r}; registered tasks: {', '.join(task_ids())}"
        ) from None


def tasks_for_sport(sport: str) -> tuple[TaskEntry, ...]:
    """Return every task registered for one sport, ordered by task id.

    A sport has more than one task as soon as it is played by more than one
    embodiment -- table tennis ships both the mocap fixture and the Panda task.
    """
    matches = tuple(
        sorted(
            (entry for entry in _TASKS.values() if entry.config.sport == sport),
            key=lambda entry: entry.task_id,
        )
    )
    if not matches:
        raise KeyError(f"no task is registered for sport {sport!r}")
    return matches


def task_for_sport(sport: str) -> TaskEntry:
    """Return the one task registered for a sport, refusing to guess.

    Naming a sport stops identifying a task once that sport has several, so
    this raises rather than picking one; use :func:`tasks_for_sport` to list
    them and :func:`get_task` to address one.
    """
    matches = tasks_for_sport(sport)
    if len(matches) > 1:
        found = ", ".join(entry.task_id for entry in matches)
        raise KeyError(f"sport {sport!r} has several tasks; name one of: {found}")
    return matches[0]


def task_ids() -> tuple[str, ...]:
    return tuple(sorted(_TASKS))


def tasks() -> Mapping[str, TaskEntry]:
    """A read-only view; mutate the registry only through :func:`register_task`."""
    return dict(_TASKS)


def iter_tasks() -> Iterator[TaskEntry]:
    for task_id in task_ids():
        yield _TASKS[task_id]
