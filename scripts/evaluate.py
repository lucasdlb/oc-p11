from __future__ import annotations

import argparse
from pathlib import Path

from common import add_log_level, configure_logging
from src.agent_factory import load_agent
from src.config import load_experiment_config
from src.evaluation import evaluate_agent


def main() -> None:
    args = parse_args()
    logger = configure_logging(args.log_level)
    config = load_experiment_config(args.config)
    paths = config.resolved_paths

    if paths is None:
        raise ValueError("Config paths were not resolved.")

    logger.info(
        "command=evaluate run=%s agent=%s config=%s",
        config.run.name,
        config.agent.type,
        args.config,
    )
    agent = load_agent(config)
    evaluate_agent(
        agent=agent,
        env_config=config.env,
        evaluation_config=config.evaluation,
        video_config=config.video,
        seed=config.run.seed,
        output_path=paths.evaluation,
        video_output_path=paths.video,
        logger=logger,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an agent and write metrics plus a sampled video."
    )
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
