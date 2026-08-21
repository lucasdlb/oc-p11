from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def write_video(frames: list[np.ndarray], output_path: Path, *, fps: int) -> None:
    if not frames:
        raise RuntimeError("No frame was produced by the environment.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )

    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
