# Gesture-Driven 3D Drone Simulator

A full Python experience that combines OpenCV + MediaPipe gesture recognition with a PyOpenGL-rendered quadcopter. Control the drone's six degrees of freedom using only your hands in real-time while watching both the 3D scene and the annotated webcam feed.

## Features
- **6-DOF Drone Physics** with smooth interpolation, gravity, auto-hover, and propeller RPM feedback.
- **MediaPipe Gesture Map**
  - Palm up → Takeoff
  - Palm down → Land
  - Push forward / Pull back → Translate along the forward axis
  - Tilt left/right → Yaw rotation
  - Tilt up/down → Pitch
  - Pinch → Emergency stop
  - Two palms together → Hover/lock
- **OpenGL Visualization** with animated propellers, ground grid, compass/radar overlay, and live telemetry.
- **Webcam HUD** overlays gesture, state, position, and FPS data onto the camera feed.

## Quick Start
1. **Create a virtual environment (recommended)**
   ```powershell
   cd "c:\Users\Tuba Khan\Downloads\3D Drone"
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
2. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```
3. **Run the simulator**
   ```powershell
   python -m src.drone_sim
   ```

> The script opens two windows: the PyOpenGL drone scene and the "Gesture Tracker" webcam feed. Press `Esc` or close either window to exit.

## Creator
- `tubakhxn`

## Fork & Stay in Sync
1. Fork the repository on GitHub under your account.
2. Clone the fork:
  ```powershell
  git clone https://github.com/<your-user>/3D-Drone-Simulator.git
  cd 3D-Drone-Simulator
  ```
3. Track the original project so you can pull upstream fixes:
  ```powershell
  git remote add upstream https://github.com/<original-owner>/3D-Drone-Simulator.git
  ```
4. Periodically sync with upstream before opening PRs:
  ```powershell
  git fetch upstream
  git checkout main
  git merge upstream/main
  git push origin main
  ```

## Tips for Best Results
- Use a well-lit background that contrasts with your hands.
- Keep both hands within the camera frame for hover/pinch gestures.
- Aim for 60 FPS webcam capture; the renderer clamps to 60 FPS for stability.
- If the camera feed inverts controls, disable `cv2.flip` in `src/drone_sim.py`.

## Project Layout
```
src/
  config.py                # runtime constants
  drone_sim.py             # entry point
  gestures/
    commands.py
    hand_controller.py     # MediaPipe gesture logic
  physics/
    drone_state.py         # motion + propellers
  render/
    drone_renderer.py      # OpenGL scene + overlays
  ui/
    hud.py                 # webcam HUD overlay
  utils/
    timers.py
```

## Troubleshooting
- Install the latest GPU drivers; PyOpenGL relies on OpenGL 3.0+.
- MediaPipe uses CPU by default—close other webcam apps if frame rate drops.
- To switch cameras, change `CAMERA_INDEX` in `src/config.py`.

Enjoy piloting your virtual drone hands-free!
