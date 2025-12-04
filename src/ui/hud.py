"""Heads-up display helpers for the webcam preview."""
from __future__ import annotations

import cv2
import numpy as np

from gestures.commands import GestureCommand
from physics.drone_state import DroneState


def draw_status_panel(frame: np.ndarray, command: GestureCommand, state: DroneState, fps: float) -> np.ndarray:
    overlay = frame.copy()
    h, w, _ = frame.shape
    panel_height = 110
    cv2.rectangle(overlay, (0, h - panel_height), (w, h), (12, 12, 12), -1)
    alpha = 0.65
    blended = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    lines = [
        f"Gesture: {command.action.name}",
        f"Drone: {state.status}",
        f"Pos: x={state.position[0]:.2f} y={state.position[1]:.2f} z={state.position[2]:.2f}",
        f"FPS: {fps:05.1f}",
    ]
    for idx, text in enumerate(lines):
        cv2.putText(
            blended,
            text,
            (20, h - panel_height + 30 + idx * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 180),
            2,
            cv2.LINE_AA,
        )
    return blended
