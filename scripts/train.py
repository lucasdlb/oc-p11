from __future__ import annotations

import argparse
from pathlib import Path

from common import add_log_level, configure_logging
from src.agent_factory import train_agent
from src.config import load_experiment_config


def main() -> None:
    args = parse_args()
    logger = configure_logging(args.log_level)
    config = load_experiment_config(args.config)

    logger.info(
        "command=train run=%s agent=%s config=%s",
        config.run.name,
        config.agent.type,
        args.config,
    )
    train_agent(config, logger)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an agent from an experiment config.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Experiment TOML config path.",
    )
    add_log_level(parser)
    return parser.parse_args()


if __name__ == "__main__":
    main()
