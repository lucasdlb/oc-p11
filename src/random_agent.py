from __future__ import annotations

from typing import Any

import numpy as np


class RandomAgent:
    name = "random"

    def __init__(self, action_space: Any) -> None:
        self.action_space = action_space

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> int:
        return int(self.action_space.sample())
