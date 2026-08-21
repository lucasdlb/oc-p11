from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    name: str
    seed: int = 42


@dataclass(frozen=True)
class AgentConfig:
    type: str


@dataclass(frozen=True)
class EnvConfig:
    id: str = "LunarLander-v3"
    enable_wind: bool = False
    wind_power: float = 15.0
    turbulence_power: float = 1.5


@dataclass(frozen=True)
class TrainConfig:
    total_timesteps: int = 100_000


@dataclass(frozen=True)
class EvaluationConfig:
    episodes: int = 50
    max_steps: int | None = 1000


@dataclass(frozen=True)
class VideoConfig:
    fps: int = 30
    max_steps: int | None = 1000


@dataclass(frozen=True)
class PathsConfig:
    root: Path = Path(".")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")
    runs_dir: Path = Path("runs")


@dataclass(frozen=True)
class ResolvedPaths:
    model: Path
    model_config: Path
    training_metrics: Path
    evaluation: Path
    video: Path
    tensorboard_root: Path
    tensorboard_run: Path
    monitor: Path


@dataclass(frozen=True)
class ExperimentConfig:
    run: RunConfig
    agent: AgentConfig
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    algorithm: dict[str, Any] = field(default_factory=dict)
    resolved_paths: ResolvedPaths | None = None


def load_experiment_config(path: Path) -> ExperimentConfig:
    raw = _load_with_includes(path)
    run = RunConfig(**raw["run"])
    agent = AgentConfig(**raw["agent"])
    paths = _paths_config(raw.get("paths", {}))
    resolved_paths = resolve_paths(run.name, agent.type, paths)

    return ExperimentConfig(
        run=run,
        agent=agent,
        env=EnvConfig(**raw.get("env", {})),
        train=TrainConfig(**raw.get("train", {})),
        evaluation=EvaluationConfig(**raw.get("evaluation", {})),
        video=VideoConfig(**raw.get("video", {})),
        paths=paths,
        algorithm=raw.get("algorithm", {}),
        resolved_paths=resolved_paths,
    )


def resolve_paths(run_name: str, agent_type: str, paths: PathsConfig) -> ResolvedPaths:
    suffix = _model_suffix(agent_type)
    root = paths.root
    models_dir = root / paths.models_dir
    reports_dir = root / paths.reports_dir
    runs_dir = root / paths.runs_dir

    return ResolvedPaths(
        model=models_dir / f"{run_name}{suffix}",
        model_config=models_dir / f"{run_name}.json",
        training_metrics=reports_dir / "training" / f"{run_name}.csv",
        evaluation=reports_dir / "evaluations" / f"{run_name}.csv",
        video=reports_dir / "videos" / f"{run_name}.mp4",
        tensorboard_root=runs_dir / "tensorboard",
        tensorboard_run=runs_dir / "tensorboard" / run_name,
        monitor=runs_dir / "monitor" / f"{run_name}_monitor.csv",
    )


def _load_with_includes(path: Path) -> dict[str, Any]:
    raw = _read_toml(path)
    includes = raw.pop("includes", {})
    merged: dict[str, Any] = {}

    for include_type, alias in includes.items():
        include_dir = "envs" if include_type == "env" else include_type
        include_path = path.parent.parent / include_dir / f"{alias}.toml"
        merged = _deep_merge(merged, _read_toml(include_path))

    return _deep_merge(merged, raw)


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return tomllib.load(file)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _paths_config(raw: dict[str, Any]) -> PathsConfig:
    return PathsConfig(
        root=Path(raw.get("root", ".")),
        models_dir=Path(raw.get("models_dir", "models")),
        reports_dir=Path(raw.get("reports_dir", "reports")),
        runs_dir=Path(raw.get("runs_dir", "runs")),
    )


def _model_suffix(agent_type: str) -> str:
    if agent_type == "sb3_dqn":
        return ".zip"
    if agent_type == "manual_dqn":
        return ".pt"
    return ".unused"
