from __future__ import annotations

from typing import Protocol

import numpy as np


class Agent(Protocol):
    name: str

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> int:
        ...
