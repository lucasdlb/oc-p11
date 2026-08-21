from __future__ import annotations

import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from src.agent_factory import load_agent
from src.config import load_experiment_config
from src.envs import ACTION_LABELS, make_env


DEFAULT_CONFIG = Path("configs/experiments/sb3_dqn_exp_001.toml")


@lru_cache(maxsize=1)
def load_runtime():
    config = load_experiment_config(DEFAULT_CONFIG)
    return config, load_agent(config)


def read_evaluation(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_model_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def browser_video_path(path: Path) -> Path:
    output_path = path.with_name(f"{path.stem}_streamlit.mp4")
    if output_path.exists() and output_path.stat().st_mtime >= path.stat().st_mtime:
        return output_path

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(path),
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return output_path


def play_episode(seed: int, max_steps: int) -> dict[str, object]:
    config, agent = load_runtime()
    env = make_env(config.env)
    actions: list[int] = []
    rewards: list[float] = []

    try:
        observation, _ = env.reset(seed=seed)
        terminated = False
        truncated = False
        total_reward = 0.0

        for _ in range(max_steps):
            action = agent.predict(np.asarray(observation), deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            actions.append(action)
            rewards.append(float(reward))
            total_reward += float(reward)
            if terminated or truncated:
                break
    finally:
        env.close()

    return {
        "total_reward": total_reward,
        "steps": len(actions),
        "terminated": terminated,
        "truncated": truncated,
        "actions": actions,
        "rewards": rewards,
    }


def main() -> None:
    st.set_page_config(page_title="Eagle-1 Dashboard", layout="wide")
    st.title("Eagle-1 LunarLander")
    st.caption("GUI et dashboard pour le pilote automatique RL.")

    config, _ = load_runtime()
    paths = config.resolved_paths
    if paths is None:
        st.error("Les chemins de configuration ne sont pas resolus.")
        return

    evaluation = read_evaluation(paths.evaluation)
    model_config = read_model_config(paths.model_config)

    st.sidebar.header("Configuration")
    st.sidebar.write(f"Run: `{config.run.name}`")
    st.sidebar.write(f"Agent: `{config.agent.type}`")
    st.sidebar.write(f"Modele: `{paths.model}`")

    metric_cols = st.columns(4)
    if evaluation.empty:
        for col, label in zip(metric_cols, ["Episodes", "Reward moyen", "Ecart-type", "Max"]):
            col.metric(label, "n/a")
    else:
        rewards = evaluation["total_reward"]
        metric_cols[0].metric("Episodes", len(evaluation))
        metric_cols[1].metric("Reward moyen", f"{rewards.mean():.2f}")
        metric_cols[2].metric("Ecart-type", f"{rewards.std(ddof=0):.2f}")
        metric_cols[3].metric("Max", f"{rewards.max():.2f}")

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Performance")
        if evaluation.empty:
            st.info(f"Aucun CSV d'evaluation trouve: `{paths.evaluation}`")
        else:
            chart_data = evaluation.set_index("episode")[["total_reward"]]
            st.line_chart(chart_data)
            st.dataframe(evaluation, width="stretch")

    with right:
        st.subheader("Video")
        if paths.video.exists():
            try:
                video_path = browser_video_path(paths.video)
            except (FileNotFoundError, subprocess.CalledProcessError):
                video_path = paths.video
                st.warning("Conversion H.264 indisponible, affichage de la video originale.")
            st.video(video_path.read_bytes(), format="video/mp4")
        else:
            st.info(f"Aucune video trouvee: `{paths.video}`")

        st.subheader("Hyperparametres")
        if model_config:
            st.json(model_config)
        else:
            st.info(f"Aucune config modele trouvee: `{paths.model_config}`")

    st.subheader("Jouer Un Episode")
    controls = st.columns([1, 1, 2])
    seed = controls[0].number_input("Seed", min_value=0, value=config.run.seed, step=1)
    max_steps = controls[1].number_input("Max steps", min_value=1, max_value=1000, value=1000, step=10)

    if controls[2].button("Lancer l'agent", type="primary"):
        result = play_episode(seed=int(seed), max_steps=int(max_steps))
        st.write(
            {
                "total_reward": round(float(result["total_reward"]), 3),
                "steps": result["steps"],
                "terminated": result["terminated"],
                "truncated": result["truncated"],
            }
        )
        actions = pd.DataFrame(
            {
                "step": range(1, len(result["actions"]) + 1),
                "action": result["actions"],
                "action_label": [ACTION_LABELS[action] for action in result["actions"]],
                "reward": result["rewards"],
            }
        )
        st.dataframe(actions, width="stretch")


if __name__ == "__main__":
    main()
