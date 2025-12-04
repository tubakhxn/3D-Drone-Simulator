"""OpenGL routines to render the drone and overlays."""
from __future__ import annotations

from typing import Tuple

from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_LINES,
    GL_MODELVIEW,
    GL_PROJECTION,
    GL_QUADS,
    glBegin,
    glClear,
    glClearColor,
    glColor3f,
    glDisable,
    glEnable,
    glEnd,
    glLineWidth,
    glLoadIdentity,
    glMatrixMode,
    glOrtho,
    glPopMatrix,
    glPushMatrix,
    glRotatef,
    glScalef,
    glTranslatef,
    glVertex3f,
    glViewport,
)
from OpenGL.GLU import gluLookAt, gluPerspective
import numpy as np

from physics.drone_state import DroneState


class DroneRenderer:
    """Handles OpenGL context setup and drawing."""

    def __init__(self, viewport: Tuple[int, int]) -> None:
        self.width, self.height = viewport
        self._prop_angle = 0.0

    def initialize(self) -> None:
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.04, 0.05, 0.08, 1.0)
        self._set_perspective(self.width, self.height)

    def _set_perspective(self, width: int, height: int) -> None:
        aspect = width / max(height, 1)
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(55.0, aspect, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def resize(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        self._set_perspective(width, height)

    def draw(self, state: DroneState, fps: float, dt: float) -> None:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        cam_target = state.position + np.array([0.0, 0.2, 0.0], dtype=np.float32)
        cam_eye = cam_target + np.array([2.2, 1.5, 4.5], dtype=np.float32)
        gluLookAt(
            cam_eye[0],
            cam_eye[1],
            cam_eye[2],
            cam_target[0],
            cam_target[1],
            cam_target[2],
            0.0,
            1.0,
            0.0,
        )
        self._draw_ground_grid()
        self._draw_drone(state, dt)
        self._draw_overlay(state, fps)

    def _draw_ground_grid(self) -> None:
        glColor3f(0.2, 0.25, 0.3)
        glBegin(GL_LINES)
        size = 20
        step = 1
        for i in range(-size, size + 1, step):
            glVertex3f(i, 0.0, -size)
            glVertex3f(i, 0.0, size)
            glVertex3f(-size, 0.0, i)
            glVertex3f(size, 0.0, i)
        glEnd()

    def _draw_drone(self, state: DroneState, dt: float) -> None:
        glPushMatrix()
        glTranslatef(state.position[0], state.position[1] + 0.5, state.position[2])
        glRotatef(np.degrees(state.rotation[2]), 0, 1, 0)  # yaw
        glRotatef(np.degrees(state.rotation[1]), 1, 0, 0)  # pitch
        glRotatef(np.degrees(state.rotation[0]), 0, 0, 1)  # roll

        # central body
        glColor3f(0.15, 0.65, 0.85)
        self._draw_box(0.4, 0.1, 0.8)

        arm_offsets = [
            (0.7, 0.0, 0.0),
            (-0.7, 0.0, 0.0),
            (0.0, 0.0, 0.7),
            (0.0, 0.0, -0.7),
        ]
        glColor3f(0.8, 0.8, 0.8)
        for ox, oy, oz in arm_offsets:
            glPushMatrix()
            glTranslatef(ox * 0.4, oy, oz * 0.4)
            glScalef(0.8 if ox == 0 else 0.08, 0.03, 0.8 if oz == 0 else 0.08)
            self._draw_box(1.0, 1.0, 1.0)
            glPopMatrix()

        self._prop_angle += state.propeller_speed * dt * 0.05
        glColor3f(0.95, 0.95, 0.95)
        prop_positions = [
            (0.6, 0.05, 0.6),
            (-0.6, 0.05, 0.6),
            (-0.6, 0.05, -0.6),
            (0.6, 0.05, -0.6),
        ]
        for px, py, pz in prop_positions:
            glPushMatrix()
            glTranslatef(px, py, pz)
            glRotatef(self._prop_angle, 0, 1, 0)
            glScalef(0.5, 0.02, 0.05)
            self._draw_box(1.0, 1.0, 1.0)
            glRotatef(90, 0, 1, 0)
            self._draw_box(1.0, 1.0, 1.0)
            glPopMatrix()

        glPopMatrix()

    def _draw_box(self, sx: float, sy: float, sz: float) -> None:
        glPushMatrix()
        glScalef(sx, sy, sz)
        glBegin(GL_QUADS)
        # top
        glVertex3f(-0.5, 0.5, -0.5)
        glVertex3f(0.5, 0.5, -0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        # bottom
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(-0.5, -0.5, 0.5)
        # front
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        # back
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(0.5, 0.5, -0.5)
        glVertex3f(-0.5, 0.5, -0.5)
        # left
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        glVertex3f(-0.5, 0.5, -0.5)
        # right
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(0.5, 0.5, -0.5)
        glEnd()
        glPopMatrix()

    def _draw_overlay(self, state: DroneState, fps: float) -> None:
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glLineWidth(2.0)
        glColor3f(0.3, 0.85, 0.8)
        cx, cy = self.width - 120, 120
        radius = 80
        glBegin(GL_LINES)
        for angle in range(0, 360, 15):
            rad = np.radians(angle)
            glVertex3f(cx, cy, 0)
            glVertex3f(cx + np.cos(rad) * radius, cy + np.sin(rad) * radius, 0)
        glEnd()

        yaw = np.degrees(state.rotation[2])
        heading = np.radians(yaw)
        glBegin(GL_LINES)
        glVertex3f(cx, cy, 0)
        glVertex3f(cx + np.cos(heading) * radius, cy + np.sin(heading) * radius, 0)
        glEnd()

        glColor3f(1.0, 1.0, 1.0)
        text_lines = [
            f"FPS: {fps:05.1f}",
            f"State: {state.status}",
            f"Alt: {state.position[1]:.2f}m",
        ]
        self._draw_bitmap_text(10, 30, text_lines)

        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def _draw_bitmap_text(self, x: int, start_y: int, lines: Tuple[str, ...] | list[str]) -> None:
        # basic bitmap font rendering via GL lines
        glLineWidth(1.5)
        for i, text in enumerate(lines):
            y = start_y + i * 18
            glBegin(GL_LINES)
            for j, ch in enumerate(text):
                ox = x + j * 9
                self._draw_glyph(ch, ox, y)
            glEnd()

    def _draw_glyph(self, ch: str, x: int, y: int) -> None:
        # draw extremely rough glyphs using segments (placeholder to avoid pygame font dependency here)
        segments = {
            "0": ((0, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 1), (0, 1, 0, 0)),
            "1": ((0.5, 0, 0.5, 1),),
            "2": ((0, 0, 1, 0), (1, 0, 1, 0.5), (1, 0.5, 0, 0.5), (0, 0.5, 0, 1), (0, 1, 1, 1)),
            "3": ((0, 0, 1, 0), (1, 0, 1, 1), (0, 0.5, 1, 0.5), (0, 1, 1, 1)),
            "4": ((0, 0, 0, 0.5), (0, 0.5, 1, 0.5), (1, 0, 1, 1)),
            "5": ((1, 0, 0, 0), (0, 0, 0, 0.5), (0, 0.5, 1, 0.5), (1, 0.5, 1, 1), (1, 1, 0, 1)),
            "6": ((1, 0, 0, 0), (0, 0, 0, 1), (0, 1, 1, 1), (1, 1, 1, 0.5), (1, 0.5, 0, 0.5)),
            "7": ((0, 0, 1, 0), (1, 0, 1, 1)),
            "8": ((0, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 1), (0, 1, 0, 0), (0, 0.5, 1, 0.5)),
            "9": ((1, 1, 1, 0), (1, 0, 0, 0), (0, 0, 0, 0.5), (0, 0.5, 1, 0.5)),
            "F": ((0, 0, 0, 1), (0, 0.5, 1, 0.5), (0, 1, 1, 1)),
            "P": ((0, 0, 0, 1), (0, 1, 1, 1), (1, 1, 1, 0.5), (1, 0.5, 0, 0.5)),
            "S": ((1, 0, 0, 0), (0, 0, 0, 0.5), (0, 0.5, 1, 0.5), (1, 0.5, 1, 1), (1, 1, 0, 1)),
            "T": ((0, 0, 1, 0), (0.5, 0, 0.5, 1)),
            "A": ((0, 1, 0.5, 0), (1, 1, 0.5, 0), (0.25, 0.5, 0.75, 0.5)),
            "E": ((1, 0, 0, 0), (0, 0, 0, 1), (0, 1, 1, 1), (0, 0.5, 0.8, 0.5)),
            "L": ((0, 0, 0, 1), (0, 1, 1, 1)),
            "H": ((0, 0, 0, 1), (1, 0, 1, 1), (0, 0.5, 1, 0.5)),
            "O": ((0, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 1), (0, 1, 0, 0)),
            "V": ((0, 0, 0.5, 1), (1, 0, 0.5, 1)),
            "R": ((0, 0, 0, 1), (0, 1, 1, 1), (1, 1, 1, 0.5), (1, 0.5, 0, 0.5), (0, 0.5, 1, 0)),
            " ": tuple(),
            ":": ((0.5, 0.2, 0.5, 0.25), (0.5, 0.75, 0.5, 0.8)),
            ".": ((0.5, 0.85, 0.5, 0.9),),
            "m": ((0, 1, 0, 0.5), (0, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 1), (0.5, 1, 1, 1), (1, 1, 1, 0.5)),
        }
        for seg in segments.get(ch, tuple()):
            x0, y0, x1, y1 = seg
            glVertex3f(x + x0 * 8, y + y0 * 12, 0)
            glVertex3f(x + x1 * 8, y + y1 * 12, 0)
