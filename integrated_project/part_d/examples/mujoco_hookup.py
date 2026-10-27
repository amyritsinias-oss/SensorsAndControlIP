"""
How to drive the team's MuJoCo TurtleBot3 with the Part D controller.

Runs with MuJoCo 3.x on turtlebot3_waffle_pi/scene.xml. It uses only the core
MuJoCo Python API from the course's mujoco_UR3_tutorial
(MjModel.from_xml_path, MjData, mj_name2id, data.ctrl, mj_step, viewer).
Edit the four names under CONFIG to match the team's MJCF.

Current status: waypoints are the fixed ones from sim/scenario.py (not yet
from the LiDAR planner) and the estimator is encoder-only dead reckoning (no
camera updates yet), so the true pose drifts away from the estimate over a run.

Loop at the control rate T:
  wheel joint angles -> encoder ticks -> EKF predict
  camera ArUco detections            -> EKF update   (team's Part B/C code)
  EKF estimate x_hat, P              -> PathFollower.step -> (v, w)
  (v, w) -> wheel rates -> velocity actuators
Ground truth (the base body's pose) is read ONLY for logging.
"""
import sys
import os
import time
from pathlib import Path as FsPath

import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tb3_control import ControllerParams, RobotParams, PathFollower, wheel_rates
from sim.scenario import build_path, START

# ---------------- CONFIG: match these to the team's MJCF ------------------
MODEL_PATH = FsPath(__file__).parent / "turtlebot3_waffle_pi" / "scene.xml"
LEFT_ACT, RIGHT_ACT = "wheel_left_vel", "wheel_right_vel"          # <velocity> actuators
LEFT_JOINT, RIGHT_JOINT = "wheel_left_joint", "wheel_right_joint"  # hinge joints (encoders)
BASE_BODY = "base_footprint"                                         # logging only
# ---------------------------------------------------------------------------


class EncoderOnlyEKF:
    """Placeholder until the team's EKF is plugged in: dead-reckoning from the
    wheel encoders with a growing covariance. Replace with the real EKF
    (needs .x and .P, and an update() with the camera detections)."""

    def __init__(self, x0, rp):
        self.x, self.P, self.rp, self.prev = np.array(x0, float), np.diag([1e-4, 1e-4, 1e-4]), rp, None

    def predict(self, phiL, phiR):
        if self.prev is not None:
            dL, dR = phiL - self.prev[0], phiR - self.prev[1]
            r, L = self.rp.wheel_radius, self.rp.track_width
            ds, dth = 0.5 * r * (dL + dR), r * (dR - dL) / L
            th = self.x[2] + 0.5 * dth
            self.x += [ds * np.cos(th), ds * np.sin(th), dth]
            self.P += np.diag([1e-6, 1e-6, 1e-6]) * (1 + 100 * abs(ds))
        self.prev = (phiL, phiR)


def main():
    rp, cp = RobotParams(), ControllerParams()
    path, _ = build_path()
    ctrl = PathFollower(path, rp, cp)

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    aL = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, LEFT_ACT)
    aR = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, RIGHT_ACT)
    qL = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, LEFT_JOINT)]
    qR = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, RIGHT_JOINT)]
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
    n_sub = max(1, int(round(cp.T / model.opt.timestep)))

    ekf = EncoderOnlyEKF(START, rp)    # <- swap for the team's EKF
    log = []
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and not ctrl.done:
            t0 = time.time()
            ekf.predict(data.qpos[qL], data.qpos[qR])
            # ekf.update(camera_detections)   <- team's ArUco code goes here
            v, w, info = ctrl.step(ekf.x, ekf.P)
            wl, wr = wheel_rates(v, w, rp)
            data.ctrl[aL], data.ctrl[aR] = wl, wr
            for _ in range(n_sub):
                mujoco.mj_step(model, data)
            viewer.sync()
            # ---- evaluation only ----
            pos = data.xpos[base]
            R = data.xmat[base].reshape(3, 3)
            log.append((data.time, pos[0], pos[1], np.arctan2(R[1, 0], R[0, 0]), *ekf.x, info["e_y"]))
            time.sleep(max(0.0, cp.T - (time.time() - t0)))
    np.save("mujoco_run_log.npy", np.array(log))
    print("saved mujoco_run_log.npy  (t, x, y, th, x_hat, y_hat, th_hat, e_y_est)")


if __name__ == "__main__":
    main()
