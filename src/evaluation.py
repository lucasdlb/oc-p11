from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.agent_interface import Agent
from src.config import EnvConfig, EvaluationConfig, VideoConfig
from src.envs import make_env
from src.metrics import EpisodeResult, summarize_results, write_results_csv
from src.video import write_video


def evaluate_agent(
    *,
    agent: Agent,
    env_config: EnvConfig,
    evaluation_config: EvaluationConfig,
    video_config: VideoConfig | None = None,
    seed: int | None,
    output_path: Path,
    video_output_path: Path | None = None,
    logger: logging.Logger,
) -> list[EpisodeResult]:
    env = make_env(env_config)
    results: list[EpisodeResult] = []

    try:
        for episode in range(1, evaluation_config.episodes + 1):
            episode_seed = None if seed is None else seed + episode - 1
            observation, _ = env.reset(seed=episode_seed)
            total_reward = 0.0
            step_number = 0
            terminated = False
            truncated = False

            while not terminated and not truncated:
                if (
                    evaluation_config.max_steps is not None
                    and step_number >= evaluation_config.max_steps
                ):
                    truncated = True
                    break

                action = agent.predict(np.asarray(observation), deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                step_number += 1

            result = EpisodeResult(
                episode=episode,
                steps=step_number,
                total_reward=total_reward,
                terminated=terminated,
                truncated=truncated,
            )
            results.append(result)
            logger.info(
                "agent=%s episode=%s steps=%s total_reward=%.3f "
                "terminated=%s truncated=%s",
                agent.name,
                result.episode,
                result.steps,
                result.total_reward,
                result.terminated,
                result.truncated,
            )
    finally:
        env.close()

    write_results_csv(results, output_path)
    summary = summarize_results(results)
    logger.info(
        "evaluation agent=%s episodes=%d reward_mean=%.3f reward_std=%.3f "
        "reward_min=%.3f reward_max=%.3f steps_mean=%.1f csv=%s",
        agent.name,
        int(summary["episodes"]),
        summary["reward_mean"],
        summary["reward_std"],
        summary["reward_min"],
        summary["reward_max"],
        summary["steps_mean"],
        output_path,
    )

    if video_config is not None and video_output_path is not None:
        record_agent_episode(
            agent=agent,
            env_config=env_config,
            video_config=video_config,
            seed=seed,
            output_path=video_output_path,
            logger=logger,
        )

    return results


def record_agent_episode(
    *,
    agent: Agent,
    env_config: EnvConfig,
    video_config: VideoConfig,
    seed: int | None,
    output_path: Path,
    logger: logging.Logger,
) -> EpisodeResult:
    env = make_env(env_config, render_mode="rgb_array")
    frames: list[np.ndarray] = []

    try:
        observation, _ = env.reset(seed=seed)
        first_frame = env.render()
        if first_frame is not None:
            frames.append(np.asarray(first_frame))

        total_reward = 0.0
        step_number = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            if video_config.max_steps is not None and step_number >= video_config.max_steps:
                truncated = True
                break

            action = agent.predict(np.asarray(observation), deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            step_number += 1
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame))

        write_video(frames, output_path, fps=video_config.fps)
        result = EpisodeResult(
            episode=1,
            steps=step_number,
            total_reward=total_reward,
            terminated=terminated,
            truncated=truncated,
        )
        logger.info(
            "video agent=%s path=%s steps=%s total_reward=%.3f "
            "terminated=%s truncated=%s",
            agent.name,
            output_path,
            result.steps,
            result.total_reward,
            result.terminated,
            result.truncated,
        )
        return result
    finally:
        env.close()
