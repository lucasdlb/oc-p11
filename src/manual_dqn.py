from __future__ import annotations

import csv
import json
import logging
import math
import random
from collections import deque, namedtuple
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.config import EnvConfig
from src.envs import make_env
from src.metrics import EpisodeResult


Transition = namedtuple(
    "Transition",
    ("state", "action", "next_state", "reward"),
)


@dataclass(frozen=True)
class ManualDQNConfig:
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    replay_size: int = 100_000
    learning_starts: int = 1_000
    batch_size: int = 128
    gamma: float = 0.99
    eps_start: float = 0.9
    eps_end: float = 0.01
    eps_decay: int = 25_000
    tau: float = 0.005
    train_freq: int = 1
    gradient_clip: float = 100.0
    net_arch: tuple[int, ...] = (128, 128)
    seed: int = 42
    log_interval: int = 10


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.memory: deque[Transition] = deque([], maxlen=capacity)

    def push(self, *args: Any) -> None:
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class QNetwork(nn.Module):
    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        net_arch: tuple[int, ...],
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_size = n_observations

        for layer_size in net_arch:
            layers.append(nn.Linear(previous_size, layer_size))
            layers.append(nn.ReLU())
            previous_size = layer_size

        layers.append(nn.Linear(previous_size, n_actions))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ManualDQNAgent:
    name = "manual_dqn"

    def __init__(self, model: QNetwork, device: torch.device) -> None:
        self.model = model
        self.device = device

    @classmethod
    def load(cls, model_path: Path) -> ManualDQNAgent:
        model, _, device = load_manual_dqn(model_path)
        return cls(model, device)

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> int:
        return predict_manual_action(self.model, observation, self.device)


def manual_dqn_config_from_mapping(
    raw: dict[str, object],
    *,
    total_timesteps: int,
    seed: int,
) -> ManualDQNConfig:
    payload = dict(raw)
    if "net_arch" in payload:
        payload["net_arch"] = tuple(int(item) for item in payload["net_arch"])
    return ManualDQNConfig(
        total_timesteps=total_timesteps,
        seed=seed,
        **payload,
    )


def train_manual_dqn(
    *,
    config: ManualDQNConfig,
    env_config: EnvConfig,
    model_path: Path,
    model_config_path: Path,
    tensorboard_run: Path,
    metrics_path: Path,
    logger: logging.Logger,
) -> ManualDQNAgent:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_config_path.parent.mkdir(parents=True, exist_ok=True)
    tensorboard_run.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    _seed_everything(config.seed)
    device = _get_device()
    env = make_env(env_config)
    writer = SummaryWriter(log_dir=str(tensorboard_run))

    try:
        state, _ = env.reset(seed=config.seed)
        n_observations = len(state)
        n_actions = int(env.action_space.n)
        policy_net = QNetwork(n_observations, n_actions, config.net_arch).to(device)
        target_net = QNetwork(n_observations, n_actions, config.net_arch).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        optimizer = optim.AdamW(
            policy_net.parameters(),
            lr=config.learning_rate,
            amsgrad=True,
        )
        memory = ReplayMemory(config.replay_size)
        episode_metrics: list[EpisodeResult] = []
        total_steps = 0
        episode = 0

        logger.info(
            "training algorithm=manual_dqn timesteps=%s model=%s device=%s",
            config.total_timesteps,
            model_path,
            device,
        )

        while total_steps < config.total_timesteps:
            episode += 1
            episode_seed = config.seed + episode - 1
            observation, _ = env.reset(seed=episode_seed)
            state_tensor = _state_tensor(observation, device)
            total_reward = 0.0
            episode_steps = 0
            terminated = False
            truncated = False

            while not terminated and not truncated:
                action_tensor, epsilon = _select_action(
                    state_tensor,
                    policy_net=policy_net,
                    env=env,
                    steps_done=total_steps,
                    config=config,
                    device=device,
                )
                observation, reward, terminated, truncated, _ = env.step(
                    int(action_tensor.item())
                )
                reward_tensor = torch.tensor([reward], device=device)
                done = terminated or truncated
                next_state_tensor = None if done else _state_tensor(observation, device)

                memory.push(
                    state_tensor,
                    action_tensor,
                    next_state_tensor,
                    reward_tensor,
                )
                state_tensor = next_state_tensor
                total_reward += float(reward)
                episode_steps += 1
                total_steps += 1

                if (
                    total_steps >= config.learning_starts
                    and total_steps % config.train_freq == 0
                ):
                    loss = _optimize_model(
                        memory=memory,
                        policy_net=policy_net,
                        target_net=target_net,
                        optimizer=optimizer,
                        config=config,
                        device=device,
                    )
                    if loss is not None:
                        writer.add_scalar("train/loss", loss, total_steps)

                _soft_update_target(policy_net, target_net, tau=config.tau)

                if total_steps >= config.total_timesteps:
                    break

            result = EpisodeResult(
                episode=episode,
                steps=episode_steps,
                total_reward=total_reward,
                terminated=terminated,
                truncated=truncated,
            )
            episode_metrics.append(result)
            writer.add_scalar("rollout/episode_reward", total_reward, total_steps)
            writer.add_scalar("rollout/episode_steps", episode_steps, total_steps)
            writer.add_scalar("rollout/epsilon", epsilon, total_steps)

            if episode == 1 or episode % config.log_interval == 0:
                recent = episode_metrics[-config.log_interval :]
                recent_mean = float(np.mean([item.total_reward for item in recent]))
                logger.info(
                    "episode=%s total_steps=%s reward=%.3f recent_mean=%.3f "
                    "epsilon=%.4f memory=%s",
                    episode,
                    total_steps,
                    total_reward,
                    recent_mean,
                    epsilon,
                    len(memory),
                )

        _save_manual_model(
            model_path=model_path,
            model_config_path=model_config_path,
            policy_net=policy_net,
            config=config,
            n_observations=n_observations,
            n_actions=n_actions,
        )
        _write_training_metrics(episode_metrics, metrics_path)
        logger.info("saved_manual_model=%s", model_path)
        logger.info("manual_training_metrics=%s", metrics_path)
        return ManualDQNAgent(policy_net, device)
    finally:
        writer.close()
        env.close()


def load_manual_dqn(model_path: Path) -> tuple[QNetwork, ManualDQNConfig, torch.device]:
    device = _get_device()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config_payload = checkpoint["config"]
    config_payload["net_arch"] = tuple(config_payload["net_arch"])
    config = ManualDQNConfig(**config_payload)
    model = QNetwork(
        int(checkpoint["n_observations"]),
        int(checkpoint["n_actions"]),
        config.net_arch,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config, device


def predict_manual_action(
    model: QNetwork,
    observation: np.ndarray,
    device: torch.device,
) -> int:
    with torch.no_grad():
        state = _state_tensor(observation, device)
        return int(model(state).max(1).indices.item())


def _select_action(
    state: torch.Tensor,
    *,
    policy_net: QNetwork,
    env: Any,
    steps_done: int,
    config: ManualDQNConfig,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    epsilon = config.eps_end + (config.eps_start - config.eps_end) * math.exp(
        -1.0 * steps_done / config.eps_decay
    )
    if random.random() > epsilon:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1), epsilon

    return (
        torch.tensor(
            [[env.action_space.sample()]],
            device=device,
            dtype=torch.long,
        ),
        epsilon,
    )


def _optimize_model(
    *,
    memory: ReplayMemory,
    policy_net: QNetwork,
    target_net: QNetwork,
    optimizer: optim.Optimizer,
    config: ManualDQNConfig,
    device: torch.device,
) -> float | None:
    if len(memory) < config.batch_size:
        return None

    transitions = memory.sample(config.batch_size)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(
        tuple(state is not None for state in batch.next_state),
        device=device,
        dtype=torch.bool,
    )
    non_final_states = [state for state in batch.next_state if state is not None]
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(config.batch_size, device=device)

    with torch.no_grad():
        if non_final_states:
            non_final_next_states = torch.cat(non_final_states)
            next_state_values[non_final_mask] = target_net(
                non_final_next_states
            ).max(1).values

    expected_state_action_values = reward_batch + config.gamma * next_state_values
    loss = F.smooth_l1_loss(
        state_action_values,
        expected_state_action_values.unsqueeze(1),
    )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(
        policy_net.parameters(),
        config.gradient_clip,
    )
    optimizer.step()
    return float(loss.item())


def _soft_update_target(
    policy_net: QNetwork,
    target_net: QNetwork,
    *,
    tau: float,
) -> None:
    target_state_dict = target_net.state_dict()
    policy_state_dict = policy_net.state_dict()

    for key in policy_state_dict:
        target_state_dict[key] = (
            policy_state_dict[key] * tau + target_state_dict[key] * (1.0 - tau)
        )

    target_net.load_state_dict(target_state_dict)


def _state_tensor(observation: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        observation,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)


def _save_manual_model(
    *,
    model_path: Path,
    model_config_path: Path,
    policy_net: QNetwork,
    config: ManualDQNConfig,
    n_observations: int,
    n_actions: int,
) -> None:
    payload = {
        "algorithm": "manual_dqn",
        "config": _config_payload(config),
        "n_observations": n_observations,
        "n_actions": n_actions,
        "state_dict": policy_net.state_dict(),
    }
    torch.save(payload, model_path)
    model_config_path.write_text(
        json.dumps(
            {
                "algorithm": "manual_dqn",
                "config": _config_payload(config),
                "n_observations": n_observations,
                "n_actions": n_actions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_training_metrics(results: list[EpisodeResult], path: Path) -> None:
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


def _config_payload(config: ManualDQNConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["net_arch"] = list(config.net_arch)
    return payload


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
