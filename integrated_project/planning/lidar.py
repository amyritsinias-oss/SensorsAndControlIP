
import mujoco
import numpy as np


class Lidar:
    def __init__(self, model, site="lidar_site", robot_body="base_footprint",
                 n_beams=360, range_min=0.12, range_max=3.5, noise_std=0.0, seed=None):
        self.model = model
        self.site_id = model.site(site).id
        self.body_id = model.body(robot_body).id
        self.n = n_beams
        self.range_min, self.range_max = range_min, range_max
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        # beam angles in the robot frame, counter-clockwise from +x (forward)
        self.angles = np.linspace(0, 2 * np.pi, n_beams, endpoint=False)
        # LiDAR position in the robot frame (needed to project points correctly)
        self.offset = model.site_pos[self.site_id][:2].copy()

    def scan(self, data): #return n_beam ranges, beams that hit nothing within the range are np.inf
        
        origin = data.site_xpos[self.site_id].copy()
        R = data.site_xmat[self.site_id].reshape(3, 3)
        dirs_robot = np.column_stack([np.cos(self.angles), np.sin(self.angles), np.zeros(self.n)])
        dirs_world = (R @ dirs_robot.T).T

        dist = np.zeros(self.n)
        geom_hit = np.zeros(self.n, dtype=np.int32)
        mujoco.mj_multiRay(self.model, data, origin, dirs_world.ravel(), None, 1, self.body_id,
                           geom_hit, dist, None, self.n, self.range_max)

        if self.noise_std > 0:
            dist = dist + self.rng.normal(0, self.noise_std, self.n)
        invalid = (geom_hit < 0) | (dist < self.range_min) | (dist > self.range_max)
        dist[invalid] = np.inf
        return dist

    def to_world(self, ranges, pose): #validbeams (N,2) point in the world frame, using the robots pose x,y,theta
        #passing the ekf estimate, not ground truth

        ok = np.isfinite(ranges)
        r, a = ranges[ok], self.angles[ok]
        pts_robot = self.offset + np.column_stack([r * np.cos(a), r * np.sin(a)])
        x, y, th = pose
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        return pts_robot @ R.T + np.array([x, y])