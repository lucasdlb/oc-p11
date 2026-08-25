from __future__ import annotations

import argparse
from pathlib import Path

from common import add_log_level, configure_logging
from src.config import load_env_config
from src.envs import inspect_environment


def main() -> None:
    args = parse_args()
    logger = configure_logging(args.log_level)
    env_config = load_env_config(args.env)

    logger.info("command=inspect env_config=%s", args.env)
    inspect_environment(env_config, logger)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a Gymnasium environment config.")
    parser.add_argument(
        "--env",
        type=Path,
        required=True,
        help="Environment TOML config path.",
    )
    add_log_level(parser)
    return parser.parse_args()


if __name__ == "__main__":
    main()
