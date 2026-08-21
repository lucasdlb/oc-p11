from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from src.agent_factory import load_agent
from src.config import ExperimentConfig, load_experiment_config
from src.envs import ACTION_LABELS, OBSERVATION_LABELS, make_env


DEFAULT_CONFIG = Path("configs/experiments/sb3_dqn_exp_001.toml")

app = FastAPI(
    title="Eagle-1 LunarLander API",
    description="API minimale pour interagir avec le pilote RL LunarLander-v3.",
    version="0.1.0",
)


class PredictionRequest(BaseModel):
    observation: list[float] = Field(
        min_length=8,
        max_length=8,
        description="Observation LunarLander-v3 dans l'ordre documente par OBSERVATION_LABELS.",
    )
    deterministic: bool = True


class PredictionResponse(BaseModel):
    action: int
    action_label: str


class EpisodeStep(BaseModel):
    step: int
    observation: list[float]
    action: int
    action_label: str
    reward: float
    terminated: bool
    truncated: bool


class EpisodeResponse(BaseModel):
    total_reward: float
    steps: int
    terminated: bool
    truncated: bool
    trajectory: list[EpisodeStep]


@lru_cache(maxsize=1)
def get_config() -> ExperimentConfig:
    return load_experiment_config(DEFAULT_CONFIG)


@lru_cache(maxsize=1)
def get_agent():
    return load_agent(get_config())


@app.get("/health")
def health() -> dict[str, object]:
    config = get_config()
    paths = config.resolved_paths
    return {
        "status": "ok",
        "run": config.run.name,
        "agent": config.agent.type,
        "model": str(paths.model) if paths else None,
        "observation_labels": OBSERVATION_LABELS,
        "action_labels": ACTION_LABELS,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    action = get_agent().predict(
        np.asarray(payload.observation, dtype=np.float32),
        deterministic=payload.deterministic,
    )
    return PredictionResponse(action=action, action_label=ACTION_LABELS[action])


@app.get("/play", response_model=EpisodeResponse)
def play(
    seed: int | None = Query(default=42),
    max_steps: Annotated[int, Query(ge=1, le=1000)] = 1000,
) -> EpisodeResponse:
    config = get_config()
    agent = get_agent()
    env = make_env(config.env)
    trajectory: list[EpisodeStep] = []

    try:
        observation, _ = env.reset(seed=seed)
        total_reward = 0.0
        terminated = False
        truncated = False

        for step in range(1, max_steps + 1):
            current_observation = np.asarray(observation, dtype=np.float32)
            action = agent.predict(current_observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            trajectory.append(
                EpisodeStep(
                    step=step,
                    observation=current_observation.tolist(),
                    action=action,
                    action_label=ACTION_LABELS[action],
                    reward=float(reward),
                    terminated=terminated,
                    truncated=truncated,
                )
            )
            if terminated or truncated:
                break
    finally:
        env.close()

    return EpisodeResponse(
        total_reward=total_reward,
        steps=len(trajectory),
        terminated=terminated,
        truncated=truncated,
        trajectory=trajectory,
    )
