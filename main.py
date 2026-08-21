from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.agent_factory import load_agent, train_agent
from src.config import load_experiment_config
from src.envs import inspect_environment
from src.evaluation import evaluate_agent, record_agent_episode


def main() -> None:
    args = parse_args()
    logger = configure_logging(args.log_level)
    config = load_experiment_config(args.config)
    paths = config.resolved_paths

    if paths is None:
        raise ValueError("Config paths were not resolved.")

    logger.info(
        "command=%s run=%s agent=%s config=%s",
        args.command,
        config.run.name,
        config.agent.type,
        args.config,
    )

    if args.command == "inspect":
        inspect_environment(config.env, logger)
        return

    if args.command == "train":
        train_agent(config, logger)
        return

    if args.command == "evaluate":
        agent = load_agent(config)
        evaluate_agent(
            agent=agent,
            env_config=config.env,
            evaluation_config=config.evaluation,
            seed=config.run.seed,
            output_path=paths.evaluation,
            logger=logger,
        )
        return

    if args.command == "record":
        agent = load_agent(config)
        record_agent_episode(
            agent=agent,
            env_config=config.env,
            video_config=config.video,
            seed=config.run.seed,
            output_path=paths.video,
            logger=logger,
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Config-driven LunarLander RL tools.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Python logging level.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "train", "evaluate", "record"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument(
            "--config",
            type=Path,
            required=True,
            help="Experiment TOML config path.",
        )

    return parser.parse_args()


def configure_logging(log_level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("lunarlander")


if __name__ == "__main__":
    main()
