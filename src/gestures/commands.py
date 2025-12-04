"""Gesture command definitions for drone control."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class DroneAction(Enum):
    IDLE = auto()
    TAKEOFF = auto()
    LAND = auto()
    MOVE_FORWARD = auto()
    MOVE_BACKWARD = auto()
    ROTATE_LEFT = auto()
    ROTATE_RIGHT = auto()
    PITCH_UP = auto()
    PITCH_DOWN = auto()
    EMERGENCY_STOP = auto()
    HOVER = auto()


@dataclass(slots=True)
class GestureCommand:
    action: DroneAction = DroneAction.IDLE
    intensity: float = 0.0
    timestamp: float = 0.0
