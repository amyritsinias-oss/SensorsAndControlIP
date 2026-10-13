#this test imports and obstacle that is not in the original scene to test object recongition and path updating
import sys
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
from sim.plant import TB3Plant  
from sim.camera import ArucoCamera  
from sim.ekf_standin import OdomArucoEKF  

SURPRISE = (3.4, 1.15, 0.15)       

xml = (PART_D / "examples" / "turtlebot3_waffle_pi" / "scene.xml").read_text()
sx, sy, sh = SURPRISE
xml = xml.replace("</worldbody>", f'<geom name="surprise" type="box" size="{sh} {sh} 0.25" '
                                  f'pos="{sx} {sy} 0.25" rgba="1 0.5 0 1"/></worldbody>')
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
lidar = Lidar(model, noise_std=0.01, seed=0)

rp, cp = RobotParams(), ControllerParams()
rng = np.random.default_rng(0)
plant = TB3Plant(START.copy(), rp, wheel_err=0.03, T=cp.T, rng=rng)
cam = ArucoCamera(MARKERS, rng=np.random.default_rng(1))
ekf = OdomArucoEKF(START, np.diag([0.05**2, 0.05**2, np.deg2rad(5)**2]), rp, MARKERS, cam.sigma)
m = MARKERS[DOCK_ID]
pre, dock, dock_heading = dock_poses(m[:2], m[2], DOCK_STANDOFF, APPROACH_LEN)

nav = Navigator(lidar, OccupancyGrid(), RefPath, pre, dock, dock_heading)
ctrl = None
LIDAR_EVERY = 4                     
truth, est, paths, stops = [], [], [], 0
ekf.predict(plant.step(0.0, 0.0))
t, k = 0.0, 0
while t < 150:

    ekf.update(cam.detect(plant.truth(), t))
    if k % LIDAR_EVERY == 0:
        xt = plant.truth()  # simulate the scan from the true pose
        data.qpos[0:3] = [xt[0], xt[1], 0.0]
        data.qpos[3:7] = [np.cos(xt[2] / 2), 0, 0, np.sin(xt[2] / 2)]
        mujoco.mj_forward(model, data)
        new_path, stop = nav.step(lidar.scan(data), ekf.x, ctrl.i if ctrl else 0)   # EKF pose only
        if new_path is not None:
            ctrl = PathFollower(new_path, rp, cp)
            paths.append((t, new_path))
            print(f"t = {t:5.1f} s  replanned ({len(nav.waypoints)} waypoints, {new_path.length:.2f} m)")
    
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

truth, est = np.array(truth), np.array(est)
d = truth[-1, :2] - dock
print(f"docked: {info['done']} after {t:.1f} s, {nav.n_replans} plans, safety stops: {stops}, "
      f"final error to dock {np.hypot(*d) * 100:.1f} cm")


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
plt.show()