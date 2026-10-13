import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from lidar import Lidar
from navigator import Navigator
from occupancy_grid import OccupancyGrid

PART_D = Path(__file__).resolve().parents[1] / "part_d"
sys.path.insert(0, str(PART_D))
from tb3_control import RobotParams, ControllerParams, PathFollower, Path as RefPath, dock_poses  # noqa: E402
from sim.scenario import MARKERS, DOCK_ID, START, DOCK_STANDOFF, APPROACH_LEN  # noqa: E402
from sim.plant import TB3Plant  # noqa: E402
from sim.camera import ArucoCamera  # noqa: E402
from sim.ekf_standin import OdomArucoEKF  # noqa: E402

SURPRISE = (3.4, 1.15, 0.15)       # (x, y, half-size) of an extra box on the planned route

# ---- MuJoCo scene with the surprise obstacle added (for the LiDAR only) ----
sx, sy, sh = SURPRISE
spec = mujoco.MjSpec.from_file(str(Path(__file__).resolve().parents[1] / "models" / "scene_room.xml"))
spec.worldbody.add_geom(name="surprise", type=mujoco.mjtGeom.mjGEOM_BOX, size=[sh, sh, 0.25],
                        pos=[sx, sy, 0.25], rgba=[1, 0.5, 0, 1])
model = spec.compile()
data = mujoco.MjData(model)
lidar = Lidar(model, site="lidar", robot_body="base", noise_std=0.01, seed=0)

# ---- Part D rig: plant, camera, EKF, controller ----
rp, cp = RobotParams(), ControllerParams()
rng = np.random.default_rng(0)
plant = TB3Plant(START.copy(), rp, wheel_err=0.03, T=cp.T, rng=rng)
cam = ArucoCamera(MARKERS, rng=np.random.default_rng(1))
ekf = OdomArucoEKF(START, np.diag([0.05**2, 0.05**2, np.deg2rad(5)**2]), rp, MARKERS, cam.sigma)
m = MARKERS[DOCK_ID]
pre, dock, dock_heading = dock_poses(m[:2], m[2], DOCK_STANDOFF, APPROACH_LEN)

nav = Navigator(lidar, OccupancyGrid(), RefPath, pre, dock, dock_heading)
ctrl = None

# ---- optional live viewer: shows the robot and draws the current plan as a green line ----
viewer = None
if "--no-viewer" not in sys.argv:
    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(model, data)


def draw_path(scn, path, step=10):
    """Draw the controller's current reference path as green line segments in the viewer."""
    scn.ngeom = 0
    for i in range(0, len(path.x) - step, step):
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.array([0.1, 0.8, 0.2, 1.0], dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.01,
                             np.array([path.x[i], path.y[i], 0.01]),
                             np.array([path.x[i + step], path.y[i + step], 0.01]))
        scn.ngeom += 1

LIDAR_EVERY = 4                     # 20 Hz control / 4 = 5 Hz LiDAR
truth, est, paths, stops = [], [], [], 0
ekf.predict(plant.step(0.0, 0.0))
t, k = 0.0, 0
while t < 150:
    t0 = time.time()
    # put the MuJoCo robot at the rig's true pose: the LiDAR scans from there and the viewer shows it
    xt = plant.truth()
    data.qpos[0:3] = [xt[0], xt[1], 0.0]
    data.qpos[3:7] = [np.cos(xt[2] / 2), 0, 0, np.sin(xt[2] / 2)]
    mujoco.mj_forward(model, data)
    # --- sensing + estimation (no ground truth below this line except to SIMULATE sensors) ---
    ekf.update(cam.detect(plant.truth(), t))
    if k % LIDAR_EVERY == 0:
        new_path, stop = nav.step(lidar.scan(data), ekf.x, ctrl.i if ctrl else 0)   # EKF pose only
        if new_path is not None:
            ctrl = PathFollower(new_path, rp, cp)
            paths.append((t, new_path))
            print(f"t = {t:5.1f} s  replanned ({len(nav.waypoints)} waypoints, {new_path.length:.2f} m)")
            if viewer is not None:
                with viewer.lock():
                    draw_path(viewer.user_scn, new_path)
    # --- control on the estimate ---
    v, w, info = ctrl.step(ekf.x, ekf.P)
    if stop:
        v, w = 0.0, 0.0
        stops += 1
    truth.append(plant.truth()); est.append(ekf.x.copy())
    if info["done"]:
        break
    ekf.predict(plant.step(v, w))
    t += cp.T
    k += 1
    if viewer is not None:
        if not viewer.is_running():
            break
        viewer.sync()
        time.sleep(max(0.0, cp.T - (time.time() - t0)))      # run in real time

truth, est = np.array(truth), np.array(est)
d = truth[-1, :2] - dock
print(f"docked: {info['done']} after {t:.1f} s, {nav.n_replans} plans, safety stops: {stops}, "
      f"final error to dock {np.hypot(*d) * 100:.1f} cm")

# ---- plot ----
g = nav.grid
extent = [g.x_min, g.x_min + g.nx * g.res, g.y_min, g.y_min + g.ny * g.res]
plt.figure(figsize=(9, 7))
plt.imshow(g.probability(), origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=1)
for j, (tp, p) in enumerate(paths):
    plt.plot(p.x, p.y, color=plt.cm.autumn(j / max(len(paths) - 1, 1)), lw=1, alpha=0.7,
             label=f"plan at t={tp:.0f} s" if j in (0, len(paths) - 1) else None)
plt.plot(truth[:, 0], truth[:, 1], "C0", lw=2, label="robot (ground truth, evaluation only)")
plt.gca().add_patch(plt.Rectangle((sx - sh, sy - sh), 2 * sh, 2 * sh, fill=False, ec="C1", lw=2,
                                  label="surprise obstacle"))
plt.plot(*dock, "k*", ms=12, label="dock")
plt.legend(loc="upper right", fontsize=8); plt.gca().set_aspect("equal")
plt.title("LiDAR navigation: map built while driving, replanned when blocked")
plt.savefig(Path(__file__).with_name("test_navigator.png"), dpi=120)
if viewer is not None:
    viewer.close()
plt.show()