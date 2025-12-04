"""Runtime configuration constants for the drone simulator."""

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
MAX_FPS = 60
CAMERA_INDEX = 0

DRONE_MASS = 1.2  # kg
GRAVITY = -9.81  # m/s^2 applied on Y axis
THROTTLE_FORCE = 16.0
LINEAR_DAMPING = 0.85
ANGULAR_DAMPING = 0.75
POSITION_SMOOTHING = 0.15
ROTATION_SMOOTHING = 0.2

PROPELLER_BASE_SPEED = 400.0
PROPELLER_MAX_SPEED = 1600.0

GESTURE_HOLD_TIME = 0.18  # quicker lock-in for responsive feedback
PINCH_THRESHOLD = 0.075  # slightly wider pinch gap to avoid false stops
TILT_THRESHOLD = 0.18  # lower tilt requirement for wrist-heavy gestures
DEPTH_THRESHOLD = 0.06  # shallower push/pull motion needed
PALM_DISTANCE_THRESHOLD = 0.12  # tighter hover trigger for dual-hands

HUD_FONT = "arial"
HUD_FONT_SIZE = 20
