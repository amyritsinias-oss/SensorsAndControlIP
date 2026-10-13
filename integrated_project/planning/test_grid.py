
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from lidar import Lidar
from occupancy_grid import OccupancyGrid

SCENE = Path(__file__).resolve().parents[1] / "part_d" / "examples" / "turtlebot3_waffle_pi" / "scene.xml"

model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)
lidar = Lidar(model, noise_std=0.01)          
grid = OccupancyGrid()


def place(pose):
    data.qpos[0:3] = [pose[0], pose[1], 0.0]
    data.qpos[3:7] = [np.cos(pose[2] / 2), 0, 0, np.sin(pose[2] / 2)]
    mujoco.mj_forward(model, data)


# poses along roughly the route the robot drives 
route = [(0.6, 0.5, 1.57), (0.7, 1.5, 1.57), (0.7, 2.4, 1.2), (1.3, 3.15, 0.0),
         (2.4, 2.9, -1.2), (2.5, 1.4, -1.57), (3.0, 0.9, 0.0), (4.0, 0.9, 0.0)]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for i, pose in enumerate(route):
    place(pose)
    grid.update(lidar, lidar.scan(data), pose)    # using a known pose here
    if i == 0:
        first = grid.probability().copy()

extent = [grid.x_min, grid.x_min + grid.nx * grid.res, grid.y_min, grid.y_min + grid.ny * grid.res]
for ax, img, title in [(axes[0], first, "After 1 scan (from start)"),
                       (axes[1], grid.probability(), f"After {len(route)} scans"),
                       (axes[2], grid.inflated(0.25).astype(float), "Inflated by 0.25 m (no-go for robot centre)")]:
    ax.imshow(img, origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_aspect("equal")
axes[1].plot(*np.array(route)[:, :2].T, "r.-", label="scan poses")
axes[1].legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig(Path(__file__).with_name("test_grid.png"), dpi=120)
print(f"occupied cells: {grid.occupied().sum()}, unknown cells: {(grid.log_odds == 0).sum()}")
plt.show()