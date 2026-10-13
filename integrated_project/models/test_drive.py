import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
model = mujoco.MjModel.from_xml_path(str(HERE / "scene_room.xml"))
data = mujoco.MjData(model)

R_WHEEL, TRACK = 0.033, 0.288            # wheel radius, distance between wheels [m]
START = (0.6, 0.5, np.pi / 2)            # same start pose as Part D's scenario.py


def place(x, y, th):
    data.qpos[0:3] = [x, y, 0.0]
    data.qpos[3:7] = [np.cos(th / 2), 0, 0, np.sin(th / 2)]
    mujoco.mj_forward(model, data)


def yaw_pitch():
    w, x, y, z = data.qpos[3:7]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return yaw, pitch


def drive(v, w):
    """Body velocity command -> wheel speed commands [rad/s]."""
    data.ctrl[model.actuator("wheel_left").id] = (v - w * TRACK / 2) / R_WHEEL
    data.ctrl[model.actuator("wheel_right").id] = (v + w * TRACK / 2) / R_WHEEL


# (v [m/s], w [rad/s], seconds): up the west side, turn right, a bit east
PLAN = [(0.0, 0.0, 0.5), (0.26, 0.0, 6.0), (0.0, -1.0, np.pi / 2), (0.2, 0.0, 3.0), (0.0, 0.0, 0.5)]

viewer = None
if "--no-viewer" not in sys.argv:
    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(model, data)

place(*START)
max_pitch, speeds = 0.0, []
for v, w, T in PLAN:
    drive(v, w)
    for _ in range(int(T / model.opt.timestep)):
        t0 = time.time()
        mujoco.mj_step(model, data)
        max_pitch = max(max_pitch, abs(yaw_pitch()[1]))
        if v == 0.26:
            speeds.append(np.hypot(*data.qvel[:2]))
        if viewer is not None:
            if not viewer.is_running():
                sys.exit()
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - t0)))

print(f"top speed: {np.max(speeds):.3f} m/s   (commanded 0.26, the real TB3 max)")
print(f"worst tilt while driving: {np.rad2deg(max_pitch):.2f} deg")
print(f"final pose: x {data.qpos[0]:.2f}  y {data.qpos[1]:.2f}  heading {np.rad2deg(yaw_pitch()[0]):.0f} deg")

# Camera check: look at the dock marker from 1.5 m away and save what the robot sees
place(3.5, 0.9, 0.0)
cam = model.camera("camera").id
fwd = -data.cam_xmat[cam].reshape(3, 3)[:, 2]
print(f"camera looks along {np.round(fwd, 2)}  (should be ~[1, 0, 0], same as the robot's heading)")
r = mujoco.Renderer(model, 480, 640)
r.update_scene(data, camera=cam)
cv2.imwrite(str(HERE / "test_drive_camera.png"), cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
print("saved what the camera sees to", HERE / "test_drive_camera.png")

if viewer is not None:
    print("close the viewer window to exit")
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.05)