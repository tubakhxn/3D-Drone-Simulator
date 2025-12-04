"""MediaPipe-powered gesture recognition for drone control."""
from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np

from gestures.commands import DroneAction, GestureCommand
import config

HandLandmark = mp.solutions.hands.HandLandmark


@dataclass(slots=True)
class HandObservation:
    landmarks: List[mp.framework.formats.landmark_pb2.NormalizedLandmark]
    handedness: str

    @property
    def wrist(self) -> mp.framework.formats.landmark_pb2.NormalizedLandmark:
        return self.landmarks[HandLandmark.WRIST]


class HandGestureController:
    """Transforms raw hand landmarks into discrete drone commands."""

    def __init__(self) -> None:
        self._hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )
        self._last_command = GestureCommand()
        self._candidate_action: DroneAction = DroneAction.IDLE
        self._candidate_start: float = 0.0
        self._instant_actions = {
            DroneAction.EMERGENCY_STOP,
            DroneAction.TAKEOFF,
            DroneAction.LAND,
            DroneAction.HOVER,
        }

    def process_frame(self, frame: np.ndarray) -> Tuple[GestureCommand, np.ndarray]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        observations: List[HandObservation] = []
        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_landmarks, handedness in zip(
                result.multi_hand_landmarks, result.multi_handedness
            ):
                observations.append(
                    HandObservation(
                        landmarks=hand_landmarks.landmark,
                        handedness=handedness.classification[0].label.lower(),
                    )
                )
        command = self._interpret(observations)
        annotated = self._draw_annotations(frame.copy(), observations, command)
        return command, annotated

    def _interpret(self, observations: List[HandObservation]) -> GestureCommand:
        now = time()
        if not observations:
            return self._lock_command(DroneAction.HOVER, 0.0, now)

        if len(observations) >= 2:
            dist = abs(observations[0].wrist.x - observations[1].wrist.x)
            if dist < config.PALM_DISTANCE_THRESHOLD:
                return self._apply_command(DroneAction.HOVER, 0.0, now)

        best_action = DroneAction.IDLE
        best_intensity = 0.0

        for obs in observations:
            pinch = self._pinch_level(obs)
            if pinch < config.PINCH_THRESHOLD:
                return self._apply_command(DroneAction.EMERGENCY_STOP, 1.0, now)

            depth = self._depth_value(obs)
            tilt_lr = self._tilt_left_right(obs)
            tilt_ud = self._tilt_up_down(obs)
            palm_up, palm_down = self._palm_orientation(obs)

            if palm_up:
                return self._apply_command(DroneAction.TAKEOFF, 1.0, now)
            if palm_down:
                return self._apply_command(DroneAction.LAND, 1.0, now)

            if depth < -config.DEPTH_THRESHOLD:
                cand_action = DroneAction.MOVE_FORWARD
                cand_intensity = np.clip(abs(depth) * 2.0, 0.0, 1.0)
            elif depth > config.DEPTH_THRESHOLD:
                cand_action = DroneAction.MOVE_BACKWARD
                cand_intensity = np.clip(abs(depth) * 2.0, 0.0, 1.0)
            elif tilt_lr > config.TILT_THRESHOLD:
                cand_action = DroneAction.ROTATE_RIGHT
                cand_intensity = np.clip(tilt_lr, 0.0, 1.0)
            elif tilt_lr < -config.TILT_THRESHOLD:
                cand_action = DroneAction.ROTATE_LEFT
                cand_intensity = np.clip(abs(tilt_lr), 0.0, 1.0)
            elif tilt_ud > config.TILT_THRESHOLD:
                cand_action = DroneAction.PITCH_UP
                cand_intensity = np.clip(tilt_ud, 0.0, 1.0)
            elif tilt_ud < -config.TILT_THRESHOLD:
                cand_action = DroneAction.PITCH_DOWN
                cand_intensity = np.clip(abs(tilt_ud), 0.0, 1.0)
            else:
                cand_action = DroneAction.HOVER
                cand_intensity = 0.0

            if cand_intensity > best_intensity:
                best_action = cand_action
                best_intensity = cand_intensity

        return self._lock_command(best_action, best_intensity, now)

    def _lock_command(self, action: DroneAction, intensity: float, now: float) -> GestureCommand:
        if action in self._instant_actions:
            return self._apply_command(action, intensity, now)

        if action != self._candidate_action:
            self._candidate_action = action
            self._candidate_start = now
            return self._last_command

        if now - self._candidate_start >= config.GESTURE_HOLD_TIME:
            return self._apply_command(action, intensity, now)
        return self._last_command

    def _apply_command(self, action: DroneAction, intensity: float, now: float) -> GestureCommand:
        self._last_command = GestureCommand(action=action, intensity=float(intensity), timestamp=now)
        self._candidate_action = action
        self._candidate_start = now
        return self._last_command

    @staticmethod
    def _vector(hand: HandObservation, index: HandLandmark, reference: HandLandmark) -> np.ndarray:
        lms = hand.landmarks
        a = lms[index]
        b = lms[reference]
        return np.array([a.x - b.x, a.y - b.y, a.z - b.z], dtype=np.float32)

    def _palm_orientation(self, hand: HandObservation) -> Tuple[bool, bool]:
        v1 = self._vector(hand, HandLandmark.INDEX_FINGER_MCP, HandLandmark.WRIST)
        v2 = self._vector(hand, HandLandmark.PINKY_MCP, HandLandmark.WRIST)
        normal = np.cross(v1, v2)
        palm_up = normal[2] < -0.05
        palm_down = normal[2] > 0.05
        return palm_up, palm_down

    def _depth_value(self, hand: HandObservation) -> float:
        lms = hand.landmarks
        z_vals = [lms[idx].z for idx in (
            HandLandmark.INDEX_FINGER_MCP,
            HandLandmark.MIDDLE_FINGER_MCP,
            HandLandmark.RING_FINGER_MCP,
        )]
        return float(np.mean(z_vals))

    def _tilt_left_right(self, hand: HandObservation) -> float:
        index_mcp = hand.landmarks[HandLandmark.INDEX_FINGER_MCP]
        pinky_mcp = hand.landmarks[HandLandmark.PINKY_MCP]
        return float(index_mcp.y - pinky_mcp.y)

    def _tilt_up_down(self, hand: HandObservation) -> float:
        wrist = hand.landmarks[HandLandmark.WRIST]
        middle_mcp = hand.landmarks[HandLandmark.MIDDLE_FINGER_MCP]
        return float(wrist.x - middle_mcp.x)

    def _pinch_level(self, hand: HandObservation) -> float:
        thumb = hand.landmarks[HandLandmark.THUMB_TIP]
        index_tip = hand.landmarks[HandLandmark.INDEX_FINGER_TIP]
        dist = np.sqrt(
            (thumb.x - index_tip.x) ** 2
            + (thumb.y - index_tip.y) ** 2
            + (thumb.z - index_tip.z) ** 2
        )
        return float(dist)

    def _draw_annotations(
        self,
        frame: np.ndarray,
        observations: List[HandObservation],
        command: GestureCommand,
    ) -> np.ndarray:
        h, w, _ = frame.shape
        for obs in observations:
            for lm in obs.landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
        cv2.putText(
            frame,
            f"Gesture: {command.action.name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame
