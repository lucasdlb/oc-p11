from __future__ import annotations

import logging

import gymnasium as gym

from pretty_printer import PrettyPrinter
from src.config import EnvConfig


OBSERVATION_LABELS = [
    "x_position",
    "y_position",
    "x_velocity",
    "y_velocity",
    "angle",
    "angular_velocity",
    "left_leg_contact",
    "right_leg_contact",
]

OBSERVATION_UNITS = [
    "norm",
    "norm",
    "norm",
    "norm",
    "rad",
    "rad/s",
    "bool",
    "bool",
]

OBSERVATION_DESCRIPTIONS = [
    "Horizontal position of the lander relative to the landing pad.",
    "Vertical position of the lander relative to the landing pad.",
    "Horizontal speed of the lander.",
    "Vertical speed of the lander.",
    "Lander angle relative to upright orientation.",
    "Angular speed of the lander.",
    "Whether the left leg touches the ground.",
    "Whether the right leg touches the ground.",
]

ACTION_LABELS = {
    0: "do_nothing",
    1: "fire_left_engine",
    2: "fire_main_engine",
    3: "fire_right_engine",
}


def make_env(config: EnvConfig, *, render_mode: str | None = None) -> gym.Env:
    return gym.make(
        config.id,
        render_mode=render_mode,
        enable_wind=config.enable_wind,
        wind_power=config.wind_power,
        turbulence_power=config.turbulence_power,
    )


def make_pretty_printer() -> PrettyPrinter:
    return PrettyPrinter(
        observation_labels=OBSERVATION_LABELS,
        observation_units=OBSERVATION_UNITS,
        observation_descriptions=OBSERVATION_DESCRIPTIONS,
        action_labels=ACTION_LABELS,
    )


def inspect_environment(config: EnvConfig, logger: logging.Logger) -> None:
    env = make_env(config)
    printer = make_pretty_printer()

    try:
        logger.info(
            "Inspecting env=%s enable_wind=%s wind_power=%.2f turbulence_power=%.2f",
            config.id,
            config.enable_wind,
            config.wind_power,
            config.turbulence_power,
        )
        printer.env(env)
    finally:
        env.close()
