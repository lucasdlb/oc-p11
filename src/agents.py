from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from src.config import EnvConfig
from src.envs import make_env


@dataclass(frozen=True)
class DQNConfig:
    total_timesteps: int = 100_000
    learning_rate: float = 1e-4
    buffer_size: int = 100_000
    learning_starts: int = 1_000
    batch_size: int = 64
    gamma: float = 0.99
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 1_000
    exploration_fraction: float = 0.2
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    net_arch: tuple[int, ...] = (256, 256)
    seed: int = 42
    verbose: int = 1


class SB3DQNAgent:
    name = "sb3_dqn"

    def __init__(self, model: DQN) -> None:
        self.model = model

    @classmethod
    def load(cls, model_path: Path) -> SB3DQNAgent:
        return cls(DQN.load(model_path))

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> int:
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(np.asarray(action).item())


def dqn_config_from_mapping(
    raw: dict[str, object],
    *,
    total_timesteps: int,
    seed: int,
) -> DQNConfig:
    payload = dict(raw)
    if "net_arch" in payload:
        payload["net_arch"] = tuple(int(item) for item in payload["net_arch"])
    return DQNConfig(
        total_timesteps=total_timesteps,
        seed=seed,
        **payload,
    )


def train_dqn(
    *,
    config: DQNConfig,
    env_config: EnvConfig,
    model_path: Path,
    model_config_path: Path,
    tensorboard_root: Path,
    monitor_path: Path,
    run_name: str,
    logger: logging.Logger,
) -> SB3DQNAgent:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_config_path.parent.mkdir(parents=True, exist_ok=True)
    tensorboard_root.mkdir(parents=True, exist_ok=True)
    monitor_path.parent.mkdir(parents=True, exist_ok=True)

    env = Monitor(make_env(env_config), filename=str(monitor_path))

    try:
        logger.info(
            "training agent=sb3_dqn timesteps=%s model=%s tensorboard=%s",
            config.total_timesteps,
            model_path,
            tensorboard_root,
        )
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            learning_starts=config.learning_starts,
            batch_size=config.batch_size,
            gamma=config.gamma,
            train_freq=config.train_freq,
            gradient_steps=config.gradient_steps,
            target_update_interval=config.target_update_interval,
            exploration_fraction=config.exploration_fraction,
            exploration_initial_eps=config.exploration_initial_eps,
            exploration_final_eps=config.exploration_final_eps,
            policy_kwargs={"net_arch": list(config.net_arch)},
            tensorboard_log=str(tensorboard_root),
            seed=config.seed,
            verbose=config.verbose,
        )
        model.learn(
            total_timesteps=config.total_timesteps,
            tb_log_name=run_name,
            progress_bar=False,
        )
        model.save(model_path)
        write_dqn_config(config, model_config_path)
        logger.info("saved_model=%s", model_path)
        return SB3DQNAgent(model)
    finally:
        env.close()


def write_dqn_config(config: DQNConfig, path: Path) -> None:
    payload = asdict(config)
    payload["net_arch"] = list(config.net_arch)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
