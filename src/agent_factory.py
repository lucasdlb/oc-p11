from __future__ import annotations

import logging

from src.agent_interface import Agent
from src.agents import SB3DQNAgent, dqn_config_from_mapping, train_dqn
from src.config import ExperimentConfig
from src.envs import make_env
from src.manual_dqn import (
    ManualDQNAgent,
    manual_dqn_config_from_mapping,
    train_manual_dqn,
)
from src.random_agent import RandomAgent


def load_agent(config: ExperimentConfig) -> Agent:
    paths = _paths(config)

    if config.agent.type == "random":
        env = make_env(config.env)
        try:
            return RandomAgent(env.action_space)
        finally:
            env.close()

    if config.agent.type == "sb3_dqn":
        return SB3DQNAgent.load(paths.model)

    if config.agent.type == "manual_dqn":
        return ManualDQNAgent.load(paths.model)

    raise ValueError(f"Unsupported agent type: {config.agent.type}")


def train_agent(config: ExperimentConfig, logger: logging.Logger) -> Agent:
    paths = _paths(config)

    if config.agent.type == "sb3_dqn":
        dqn_config = dqn_config_from_mapping(
            config.algorithm,
            total_timesteps=config.train.total_timesteps,
            seed=config.run.seed,
        )
        return train_dqn(
            config=dqn_config,
            env_config=config.env,
            model_path=paths.model,
            model_config_path=paths.model_config,
            tensorboard_root=paths.tensorboard_root,
            monitor_path=paths.monitor,
            run_name=config.run.name,
            logger=logger,
        )

    if config.agent.type == "manual_dqn":
        manual_config = manual_dqn_config_from_mapping(
            config.algorithm,
            total_timesteps=config.train.total_timesteps,
            seed=config.run.seed,
        )
        return train_manual_dqn(
            config=manual_config,
            env_config=config.env,
            model_path=paths.model,
            model_config_path=paths.model_config,
            tensorboard_run=paths.tensorboard_run,
            metrics_path=paths.training_metrics,
            logger=logger,
        )

    if config.agent.type == "random":
        raise ValueError("RandomAgent does not support training.")

    raise ValueError(f"Unsupported agent type: {config.agent.type}")


def _paths(config: ExperimentConfig):
    if config.resolved_paths is None:
        raise ValueError("Experiment config has no resolved paths.")
    return config.resolved_paths
