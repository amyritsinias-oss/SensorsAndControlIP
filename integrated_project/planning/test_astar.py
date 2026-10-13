import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from astar import plan
from lidar import Lidar
from occupancy_grid import OccupancyGrid

PART_D = Path(__file__).resolve().parents[1] / "part_d"
sys.path.insert(0, str(PART_D))
from tb3_control import Path as RefPath, dock_poses          # noqa: E402
from sim.scenario import MARKERS, DOCK_ID, START, DOCK_STANDOFF, APPROACH_LEN  # noqa: E402

SCENE = Path(__file__).resolve().parents[1] / "models" / "scene_room.xml"


model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)
lidar = Lidar(model, site="lidar", robot_body="base", noise_std=0.01)
grid = OccupancyGrid()


for pose in [(0.6, 0.5, 1.57), (0.7, 1.5, 1.57), (0.7, 2.4, 1.2), (1.3, 3.15, 0.0),
             (2.4, 2.9, -1.2), (2.5, 1.4, -1.57), (3.0, 0.9, 0.0), (4.0, 0.9, 0.0)]:
    data.qpos[0:3] = [pose[0], pose[1], 0.0]
    data.qpos[3:7] = [np.cos(pose[2] / 2), 0, 0, np.sin(pose[2] / 2)]
    mujoco.mj_forward(model, data)
    grid.update(lidar, lidar.scan(data), pose)

m = MARKERS[DOCK_ID]
pre, dock, dock_heading = dock_poses(m[:2], m[2], DOCK_STANDOFF, APPROACH_LEN)


waypoints, cells = plan(grid, START[:2], pre, inflate=0.30)
print("waypoints (m):\n", np.round(waypoints, 2))

ref = RefPath.from_waypoints(waypoints[:-1], START[2], pre, dock, dock_heading)

occ_r, occ_c = np.nonzero(grid.occupied())
ox, oy = grid.to_world(occ_r, occ_c)
d = np.hypot(ref.x[:, None] - ox[None, :], ref.y[:, None] - oy[None, :]).min(axis=1)
print(f"path length {ref.length:.2f} m, closest approach to an obstacle {d.min():.2f} m "
      f"(robot radius ~0.22 m)")

extent = [grid.x_min, grid.x_min + grid.nx * grid.res, grid.y_min, grid.y_min + grid.ny * grid.res]
plt.figure(figsize=(8, 6.5))
plt.imshow(grid.probability(), origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=1)
plt.imshow(np.ma.masked_where(~grid.inflated(0.30), np.ones_like(grid.log_odds)), origin="lower",
           extent=extent, cmap="Reds", alpha=0.25, vmin=0, vmax=1)
cx, cy = grid.to_world(*np.array(cells).T)
plt.plot(cx, cy, "C0", lw=1, alpha=0.6, label="raw A* cells")
plt.plot(waypoints[:, 0], waypoints[:, 1], "C1o--", label="pruned waypoints")
plt.plot(ref.x, ref.y, "C2", lw=2, label="Part D reference path")
plt.plot(*dock, "k*", ms=12, label="dock")
plt.legend(loc="upper right", fontsize=8); plt.gca().set_aspect("equal")
plt.title("A* on the LiDAR map -> waypoints -> controller path")
plt.savefig(Path(__file__).with_name("test_astar.png"), dpi=120)
plt.show()