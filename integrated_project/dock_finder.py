"""Find the charging station from what the robot sees - no hard-coded dock position.

    camera : WHERE the dock marker (ArUco id 0) is  -> marker centre in the robot frame (solvePnP)
    LiDAR  : HOW to line up with it                 -> line fitted to the wall points around the marker

The camera's position estimate is good, but its estimate of which way a far-away marker faces is
not (seen almost head-on, solvePnP's orientation is ambiguous: up to ~40 deg off at 4 m). The wall
the marker is stuck on is a long, straight LiDAR feature, so its normal gives the facing direction
to about a degree. The marker centre is then snapped onto that wall line.

Everything is in the world frame via the EKF estimate - never ground truth.
"""
import numpy as np

from tb3_control import dock_poses, wrap


class DockFinder:
    def __init__(self, standoff, approach_len, freeze_dist=1.0, wall_radius=0.5):
        self.standoff, self.approach_len = standoff, approach_len
        self.freeze_dist = freeze_dist      # stop updating the target once this close (approach in progress)
        self.wall_radius = wall_radius      # LiDAR points within this of the marker are used for the wall fit
        self.sum_w, self.sum_xy = 0.0, np.zeros(2)
        self.marker_xy = None               # current best estimate of the marker centre (world)
        self.facing = None                  # direction the marker faces, out of the wall (world)
        self.frozen = False
        self.n_sightings = 0
        self.wall_fitted = False            # False = provisional facing (wall still beyond LiDAR range)
        self._published = None              # the estimate the robot is currently driving to

    @property
    def found(self):
        return self.marker_xy is not None and self.facing is not None

    def update(self, obs, pose, scan_world):
        """obs = camera detection of id 0 (dict with p_base), pose = EKF [x, y, th],
        scan_world = (N, 2) LiDAR hits in the world. Returns True if the target moved noticeably."""
        if self.frozen:
            return False
        x, y, th = pose
        c, s = np.cos(th), np.sin(th)
        p = obs["p_base"][:2]
        seen_xy = np.array([x + c * p[0] - s * p[1], y + s * p[0] + c * p[1]])
        w = 1.0 / max(obs["range"], 0.3) ** 2                     # closer sightings count for more
        self.sum_w += w
        self.sum_xy += w * seen_xy
        self.n_sightings += 1
        guess = self.sum_xy / self.sum_w

        facing = self._wall_normal(guess, scan_world, pose)
        if facing is not None:
            if not self.wall_fitted:                              # first proper wall fit replaces the guess
                self.facing, self.wall_fitted = facing, True
            else:
                self.facing = wrap(self.facing + 0.3 * wrap(facing - self.facing))
            n = np.array([np.cos(self.facing), np.sin(self.facing)])
            self.marker_xy = guess - n * np.dot(guess - self._wall_point, n)  # snap the centre onto the wall
        else:
            # wall not in LiDAR range yet (3.5 m): provisional facing = from the marker towards the robot,
            # good enough to start driving; replaced by the wall fit once the wall is in range
            self.marker_xy = guess
            if not self.wall_fitted:
                self.facing = float(np.arctan2(y - guess[1], x - guess[0]))

        if self.wall_fitted and np.hypot(*(np.array(pose[:2]) - self.marker_xy)) < self.freeze_dist + self.standoff:
            self.frozen = True                                    # close: keep the target steady for the approach
        if not self.found:
            return False
        if self._published is None:
            changed = True
        else:                                     # compare with the target in use, not the last update
            changed = (np.hypot(*(self.marker_xy - self._published[0])) > 0.03
                       or abs(wrap(self.facing - self._published[1])) > np.deg2rad(2))
        if changed:
            self._published = (self.marker_xy.copy(), self.facing)
        return changed

    def _wall_normal(self, guess, pts, pose):
        if pts is None or len(pts) == 0:
            return None
        near = pts[np.hypot(pts[:, 0] - guess[0], pts[:, 1] - guess[1]) < self.wall_radius]
        if len(near) < 8:
            return None
        centre = near.mean(axis=0)
        _, sv, vt = np.linalg.svd(near - centre)
        if sv[1] > 0.2 * sv[0]:                                   # not a straight line (corner / clutter)
            return None
        normal = np.array([-vt[0, 1], vt[0, 0]])                  # perpendicular to the wall direction
        if np.dot(np.array(pose[:2]) - centre, normal) < 0:       # point it into the room, towards the robot
            normal = -normal
        self._wall_point = centre
        return float(np.arctan2(normal[1], normal[0]))

    def targets(self):
        """(pre_dock_xy, dock_xy, dock_heading) for the navigator / controller."""
        return dock_poses(self.marker_xy, self.facing, self.standoff, self.approach_len)