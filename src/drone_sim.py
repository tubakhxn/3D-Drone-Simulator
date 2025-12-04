"""Entry point for the 3D gesture-controlled drone simulator."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import cv2
import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, RESIZABLE

import config
from gestures.hand_controller import HandGestureController
from physics.drone_state import DronePhysics
from render.drone_renderer import DroneRenderer
from ui.hud import draw_status_panel
from utils.timers import FrameTimer


def init_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Unable to access webcam")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv2.CAP_PROP_FPS, 60)
    return cap


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Gesture Drone Simulator")
    pygame.display.set_mode(
        (config.WINDOW_WIDTH, config.WINDOW_HEIGHT), DOUBLEBUF | OPENGL | RESIZABLE
    )
    clock = pygame.time.Clock()

    renderer = DroneRenderer((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    renderer.initialize()

    gesture_controller = HandGestureController()
    physics = DronePhysics()
    fps_timer = FrameTimer()

    try:
        camera = init_camera(config.CAMERA_INDEX)
    except RuntimeError as err:
        pygame.quit()
        raise SystemExit(err) from err

    running = True
    dt = 0.0
    try:
        while running:
            dt = clock.tick(config.MAX_FPS) / 1000.0
            if dt <= 0:
                dt = 1e-6
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    renderer.resize(event.w, event.h)

            ret, frame = camera.read()
            if not ret:
                print("Camera frame lost", file=sys.stderr)
                break
            frame = cv2.flip(frame, 1)

            command, annotated = gesture_controller.process_frame(frame)
            state = physics.step(command, dt)
            fps_value = fps_timer.tick()

            renderer.draw(state, fps_value, dt)
            pygame.display.flip()

            hud_frame = draw_status_panel(annotated, command, state, fps_value)
            cv2.imshow("Gesture Tracker", hud_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                running = False
    finally:
        camera.release()
        cv2.destroyAllWindows()
        pygame.quit()


if __name__ == "__main__":
    main()
