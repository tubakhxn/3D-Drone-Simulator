"""Simplified 6-DOF drone physics integrator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from gestures.commands import DroneAction, GestureCommand
import config


@dataclass(slots=True)
class DroneState:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))  # roll, pitch, yaw (rad)
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    propeller_speed: float = 0.0
    status: str = "LANDED"


class DronePhysics:
    """Applies gesture commands to drone state with smooth interpolation."""

    def __init__(self) -> None:
        self.state = DroneState()
        self._velocity_target = np.zeros(3, dtype=np.float32)
        self._angular_target = np.zeros(3, dtype=np.float32)
        self._altitude_target = 0.0
        self._max_speed = 3.0
        self._max_angular_speed = np.radians(90.0)

    def step(self, command: GestureCommand, dt: float) -> DroneState:
        self._apply_command(command)
        self._integrate_linear(dt)
        self._integrate_angular(dt)
        self._update_propellers()
        return self.state

    def _apply_command(self, command: GestureCommand) -> None:
        action = command.action
        intensity = np.clip(command.intensity, 0.0, 1.0)
        if action == DroneAction.TAKEOFF:
            self.state.status = "Takeoff"
            self._altitude_target = max(1.2, self.state.position[1])
        elif action == DroneAction.LAND:
            self.state.status = "Landing"
            self._altitude_target = 0.0
        elif action == DroneAction.EMERGENCY_STOP:
            self.state.status = "Emergency"
            self._altitude_target = self.state.position[1]
            self._velocity_target *= 0.0
            self._angular_target *= 0.0
            self.state.velocity *= 0.0
            self.state.angular_velocity *= 0.0
            return
        elif action == DroneAction.HOVER:
            self.state.status = "Hover"
            self._velocity_target *= 0.0
        elif action in (DroneAction.MOVE_FORWARD, DroneAction.MOVE_BACKWARD):
            direction = self._forward_vector()
            scalar = intensity * self._max_speed
            if action == DroneAction.MOVE_BACKWARD:
                scalar *= -1.0
            self._velocity_target = direction * scalar
            self.state.status = "Moving"
        elif action in (DroneAction.ROTATE_LEFT, DroneAction.ROTATE_RIGHT):
            scalar = intensity * self._max_angular_speed
            if action == DroneAction.ROTATE_LEFT:
                scalar *= -1.0
            self._angular_target[1] = scalar  # yaw axis
            self.state.status = "Yaw"
        elif action in (DroneAction.PITCH_UP, DroneAction.PITCH_DOWN):
            scalar = intensity * self._max_angular_speed * 0.6
            if action == DroneAction.PITCH_DOWN:
                scalar *= -1.0
            self._angular_target[0] = scalar  # roll for visual pitch
            self.state.status = "Pitch"

    def _integrate_linear(self, dt: float) -> None:
        altitude_error = self._altitude_target - self.state.position[1]
        altitude_thrust = np.clip(altitude_error * 4.0, -2.0, 2.0)
        gravity = np.array([0.0, config.GRAVITY, 0.0], dtype=np.float32)
        lift = np.array([0.0, altitude_thrust * config.THROTTLE_FORCE, 0.0], dtype=np.float32)

        desired = self._velocity_target.copy()
        desired[1] += altitude_thrust
        self.state.velocity = self._lerp(self.state.velocity, desired, config.POSITION_SMOOTHING)
        self.state.velocity += (gravity + lift) * dt / config.DRONE_MASS
        self.state.velocity *= config.LINEAR_DAMPING

        self.state.position += self.state.velocity * dt
        if self.state.position[1] <= 0.0 and self._altitude_target == 0.0:
            self.state.position[1] = 0.0
            self.state.velocity[1] = 0.0
            self.state.status = "Landed"

    def _integrate_angular(self, dt: float) -> None:
        self.state.angular_velocity = self._lerp(
            self.state.angular_velocity, self._angular_target, config.ROTATION_SMOOTHING
        )
        self.state.angular_velocity *= config.ANGULAR_DAMPING
        self.state.rotation += self.state.angular_velocity * dt
        # keep angles bounded for readability
        self.state.rotation = (self.state.rotation + np.pi) % (2 * np.pi) - np.pi
        if np.allclose(self._angular_target, 0.0, atol=1e-3):
            self.state.status = "Hover" if self.state.status == "Yaw" else self.state.status

    def _update_propellers(self) -> None:
        linear_speed = float(np.linalg.norm(self.state.velocity))
        angular_speed = float(np.linalg.norm(self.state.angular_velocity))
        throttle = np.clip(self._altitude_target, 0.0, 2.0)
        speed = config.PROPELLER_BASE_SPEED + 200.0 * linear_speed + 400.0 * angular_speed + 120.0 * throttle
        self.state.propeller_speed = float(np.clip(speed, config.PROPELLER_BASE_SPEED, config.PROPELLER_MAX_SPEED))

    def _forward_vector(self) -> np.ndarray:
        roll, pitch, yaw = self.state.rotation
        forward = np.array(
            [
                np.sin(yaw) * np.cos(pitch),
                -np.sin(pitch),
                np.cos(yaw) * np.cos(pitch),
            ],
            dtype=np.float32,
        )
        norm = np.linalg.norm(forward)
        return forward / norm if norm else forward

    @staticmethod
    def _lerp(current: np.ndarray, target: np.ndarray, factor: float) -> np.ndarray:
        return current + (target - current) * np.clip(factor, 0.0, 1.0)

    @property
    def orientation_degrees(self) -> Tuple[float, float, float]:
        return tuple(np.degrees(self.state.rotation).tolist())
