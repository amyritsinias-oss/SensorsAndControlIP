#takes scan of room and tests if lidar actually works
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from lidar import Lidar

SCENE = Path(__file__).resolve().parents[1] / "models" / "scene_room.xml"

model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)
lidar = Lidar(model, site="lidar", robot_body="base")

# Place the robot (x, y, heading). The free joint's qpos is [x, y, z, qw, qx, qy, qz].
pose = np.array([1.0, 3.0, 0.3])
data.qpos[0:3] = [pose[0], pose[1], 0.0]
data.qpos[3:7] = [np.cos(pose[2] / 2), 0, 0, np.sin(pose[2] / 2)]
mujoco.mj_forward(model, data)          # update positions without simulating

ranges = lidar.scan(data)
pts = lidar.to_world(ranges, pose)      # here pose is known exactly; in the real loop use the EKF estimate
print(f"{np.isfinite(ranges).sum()} / {len(ranges)} beams hit something")

plt.figure(figsize=(7, 6))
plt.plot([0, 5, 5, 0, 0], [0, 0, 4, 4, 0], "k-", lw=1, label="room")
for x0, y0, x1, y1 in [(1.5, 0.0, 2.0, 2.2), (3.0, 1.8, 3.5, 4.0)]:
    plt.fill([x0, x1, x1, x0], [y0, y0, y1, y1], color="0.85")
plt.scatter(pts[:, 0], pts[:, 1], s=4, c="C2", label="LiDAR hits")
plt.plot(pose[0], pose[1], "ro", label="robot")
plt.arrow(pose[0], pose[1], 0.3 * np.cos(pose[2]), 0.3 * np.sin(pose[2]), color="r", width=0.02)
plt.axis("equal"); plt.legend(); plt.title("LiDAR scan in the world frame")
plt.savefig(Path(__file__).with_name("test_lidar.png"), dpi=120)
plt.show()