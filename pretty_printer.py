from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PrettyPrinter:
    """Affichage lisible des données d'un environnement Gymnasium."""

    def __init__(
        self,
        *,
        observation_labels: Sequence[str] | None = None,
        observation_units: Sequence[str] | None = None,
        observation_descriptions: Sequence[str] | None = None,
        action_labels: Mapping[int, str] | None = None,
        width: int = 88,
        precision: int = 4,
    ) -> None:
        self.observation_labels = list(observation_labels or [])
        self.observation_units = list(observation_units or [])
        self.observation_descriptions = list(
            observation_descriptions or []
        )
        self.action_labels = dict(action_labels or {})

        self.width = width
        self.precision = precision

        self._validate_configuration()

    # ================================================================
    # Validation et formatage interne
    # ================================================================

    def _validate_configuration(self) -> None:
        """Vérifie la cohérence des labels, unités et descriptions."""
        lengths = [
            len(values)
            for values in (
                self.observation_labels,
                self.observation_units,
                self.observation_descriptions,
            )
            if values
        ]

        if lengths and len(set(lengths)) > 1:
            raise ValueError(
                "observation_labels, observation_units et "
                "observation_descriptions doivent avoir la même taille."
            )

    def _line(self, character: str = "─") -> None:
        print(character * self.width)

    def _title(self, text: str, character: str = "═") -> None:
        print()
        self._line(character)
        print(text)
        self._line(character)

    def _format_number(self, value: Any) -> str:
        """Formate une valeur numérique, y compris les infinis."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)

        if np.isposinf(number):
            return "+∞"

        if np.isneginf(number):
            return "-∞"

        if np.isnan(number):
            return "NaN"

        return f"{number:.{self.precision}f}"

    def _observation_metadata(
        self,
        size: int,
    ) -> tuple[list[str], list[str], list[str]]:
        """Retourne les métadonnées d'observation avec valeurs par défaut."""
        labels = self.observation_labels or [
            f"Variable {index}" for index in range(size)
        ]

        units = self.observation_units or ["—"] * size
        descriptions = self.observation_descriptions or ["—"] * size

        if len(labels) != size:
            raise ValueError(
                f"L'observation contient {size} valeurs, "
                f"mais {len(labels)} labels ont été configurés."
            )

        if len(units) != size:
            raise ValueError(
                f"L'observation contient {size} valeurs, "
                f"mais {len(units)} unités ont été configurées."
            )

        if len(descriptions) != size:
            raise ValueError(
                f"L'observation contient {size} valeurs, "
                f"mais {len(descriptions)} descriptions ont été configurées."
            )

        return labels, units, descriptions

    # ================================================================
    # Affichage de l'environnement et de ses espaces
    # ================================================================

    def env(self, env: gym.Env) -> None:
        """Affiche les informations générales de l'environnement."""
        env_id = getattr(env.spec, "id", type(env).__name__)

        self._title(f"ENVIRONNEMENT : {env_id}", "█")

        print(f"Type Python          : {type(env).__name__}")
        print(f"Espace d'observation : {env.observation_space}")
        print(f"Espace d'action      : {env.action_space}")

        if env.spec is not None:
            print(f"Identifiant          : {env.spec.id}")
            print(f"Étapes maximales     : {env.spec.max_episode_steps}")

        self.observation_space(env.observation_space)
        self.action_space(env.action_space)

    def observation_space(self, space: spaces.Space) -> None:
        """Affiche la structure de l'espace d'observation."""
        self._title("ESPACE D'OBSERVATION")

        print(f"Classe : {type(space).__name__}")
        print(f"Forme  : {getattr(space, 'shape', None)}")
        print(f"Type   : {getattr(space, 'dtype', None)}")
        print(f"Objet  : {space}")

        if not isinstance(space, spaces.Box):
            print(
                "\nL'affichage détaillé des bornes est prévu "
                "principalement pour les espaces Box."
            )
            return

        low = np.asarray(space.low).reshape(-1)
        high = np.asarray(space.high).reshape(-1)

        labels, units, descriptions = self._observation_metadata(
            len(low)
        )

        print()
        print(
            f"{'Index':<7}"
            f"{'Variable':<31}"
            f"{'Unité':<10}"
            f"{'Minimum':>12}"
            f"{'Maximum':>12}"
        )
        self._line()

        for index, (minimum, maximum) in enumerate(zip(low, high)):
            print(
                f"{index:<7}"
                f"{labels[index]:<31}"
                f"{units[index]:<10}"
                f"{self._format_number(minimum):>12}"
                f"{self._format_number(maximum):>12}"
            )

            description = descriptions[index]

            if description != "—":
                print(f"{'':<7}↳ {description}")

    def action_space(self, space: spaces.Space) -> None:
        """Affiche la structure de l'espace d'action."""
        self._title("ESPACE D'ACTION")

        print(f"Classe : {type(space).__name__}")
        print(f"Forme  : {getattr(space, 'shape', None)}")
        print(f"Type   : {getattr(space, 'dtype', None)}")
        print(f"Objet  : {space}")

        if isinstance(space, spaces.Discrete):
            print()
            print(f"{'Valeur':<10}Signification")
            self._line()

            first_value = int(space.start)
            last_value = first_value + int(space.n)

            for value in range(first_value, last_value):
                label = self.action_labels.get(
                    value,
                    "Action non documentée",
                )

                print(f"{value:<10}{label}")

        elif isinstance(space, spaces.Box):
            print("\nBornes basses :", space.low)
            print("Bornes hautes  :", space.high)

        else:
            print(
                "\nAucun affichage spécialisé n'est défini "
                f"pour {type(space).__name__}."
            )

    # ================================================================
    # Affichage des données pendant l'exécution
    # ================================================================

    def observation(
        self,
        observation: Any,
        *,
        title: str = "Observation",
        show_descriptions: bool = False,
    ) -> None:
        """Affiche une observation concrète."""
        values = np.asarray(observation).reshape(-1)

        labels, units, descriptions = self._observation_metadata(
            len(values)
        )

        print(f"{title} :")

        for index, value in enumerate(values):
            print(
                f"  [{index}] "
                f"{labels[index]:<31} "
                f"= {self._format_number(value):>10} "
                f"{units[index]}"
            )

            if show_descriptions and descriptions[index] != "—":
                print(f"      ↳ {descriptions[index]}")

    def action(
        self,
        action: Any,
        *,
        title: str = "Action",
    ) -> None:
        """Affiche une action concrète."""
        action_value = int(np.asarray(action).item())

        label = self.action_labels.get(
            action_value,
            "Action non documentée",
        )

        print(f"{title:<19}: {action_value} — {label}")

    def action_probs(self, env: gym.Env, observation: Any, *, show_descriptions: bool = False) -> None:

        for action, transitions in env.unwrapped.P[observation].items():
            print(f"\nAction {action} — {self.action_labels[action]}")

            for probability, next_state, reward, terminated in transitions:
                print(
                    f"  p={probability:.3f}"
                    f": {observation} -> state={next_state:2d} "
                    f"reward={reward} "
                    f"terminated={terminated}"
                )

    def reward(
        self,
        reward: float,
        *,
        total: float | None = None,
    ) -> None:
        """Affiche la récompense immédiate et éventuellement le cumul."""
        print(
            f"{'Récompense':<19}: "
            f"{self._format_number(reward)}"
        )

        if total is not None:
            print(
                f"{'Récompense cumulée':<19}: "
                f"{self._format_number(total)}"
            )

    def status(
        self,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Affiche l'état de fin de l'épisode."""
        if terminated:
            label = "épisode terminé"
        elif truncated:
            label = "épisode tronqué"
        else:
            label = "épisode en cours"

        print(f"{'Statut':<19}: {label}")
        print(f"{'terminated':<19}: {terminated}")
        print(f"{'truncated':<19}: {truncated}")

    def info(self, info: Mapping[str, Any]) -> None:
        """Affiche le dictionnaire info."""
        print(f"{'Info':<19}: {dict(info)}")

    # ================================================================
    # Affichages composés
    # ================================================================

    def episode_start(
        self,
        episode: int,
        observation: Any,
    ) -> None:
        """Affiche le début d'un épisode."""
        self._title(f"ÉPISODE {episode}")

        self.observation(
            observation,
            title="Observation initiale",
        )

    def step(
        self,
        *,
        step_number: int,
        observation_before: Any,
        action: Any,
        observation_after: Any,
        reward: float,
        total_reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any] | None = None,
        show_before: bool = False,
        show_info: bool = False,
    ) -> None:
        """Affiche une transition complète."""
        self._title(f"ÉTAPE {step_number}", "─")

        if show_before:
            self.observation(
                observation_before,
                title="Observation avant",
            )
            print()

        self.action(action)
        self.reward(reward, total=total_reward)
        self.status(terminated, truncated)

        print()
        self.observation(
            observation_after,
            title="Nouvelle observation",
        )

        if show_info and info is not None:
            print()
            self.info(info)

    def step_compact(
        self,
        *,
        step_number: int,
        action: Any,
        observation_before: Any,
        observation_after: Any,
        reward: float,
        total_reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Affiche une transition sur une seule ligne."""
        action_value = int(np.asarray(action).item())
        action_label = self.action_labels.get(
            action_value,
            "Action inconnue",
        )

        if terminated:
            status = "TERMINATED"
        elif truncated:
            status = "TRUNCATED"
        else:
            status = "RUNNING"

        print(
            f"Étape {step_number:>3} │ "
            f"State {observation_before:>3} │ "
            f"action={action_value} │ "
            f"({action_label:<26}) │ "
            f"Next State {observation_after:>3} │ "
            f"reward={float(reward):>5.1f} │ "
            f"total={float(total_reward):>6.1f} │ "
            f"{status}"
        )

    def episode_summary(
        self,
        *,
        episode: int,
        steps: int,
        total_reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Affiche le résumé final d'un épisode."""
        self._title(f"RÉSUMÉ DE L'ÉPISODE {episode}")

        print(f"{'Nombre d’étapes':<24}: {steps}")
        print(
            f"{'Récompense totale':<24}: "
            f"{self._format_number(total_reward)}"
        )

        if terminated:
            cause = "condition terminale de l'environnement"
        elif truncated:
            cause = "limite externe ou temporelle"
        else:
            cause = "épisode encore actif"

        print(f"{'Cause de fin':<24}: {cause}")