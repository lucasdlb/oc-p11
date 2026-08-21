from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EpisodeResult:
    episode: int
    steps: int
    total_reward: float
    terminated: bool
    truncated: bool


def summarize_results(results: list[EpisodeResult]) -> dict[str, float]:
    rewards = np.asarray([result.total_reward for result in results], dtype=float)
    steps = np.asarray([result.steps for result in results], dtype=float)
    return {
        "episodes": float(len(results)),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "reward_min": float(rewards.min()),
        "reward_max": float(rewards.max()),
        "steps_mean": float(steps.mean()),
    }


def write_results_csv(results: list[EpisodeResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "episode",
                "steps",
                "total_reward",
                "terminated",
                "truncated",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "episode": result.episode,
                    "steps": result.steps,
                    "total_reward": f"{result.total_reward:.6f}",
                    "terminated": result.terminated,
                    "truncated": result.truncated,
                }
            )
