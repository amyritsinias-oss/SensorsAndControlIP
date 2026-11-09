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
    """Two-stage target:
       beacon stage - a marker seen from far away (e.g. mounted high) gives a rough target to drive towards.
                      It never locks the target.
       dock stage   - the first sighting of the docking marker itself (e.g. low, at the dock) throws the
                      beacon estimate away; from then on only the docking marker is used, and the target
                      locks once the robot is freeze_dist from the dock.
    With a single marker (beacon id == dock id), every sighting is a dock-stage sighting."""

    def __init__(self, standoff, approach_len, freeze_dist=0.5, wall_radius=0.5, min_dock_sightings=5):
        self.standoff, self.approach_len = standoff, approach_len
        self.freeze_dist = freeze_dist      # lock the target once the robot is this close to the dock pose
        self.wall_radius = wall_radius      # LiDAR points within this of the marker are used for the wall fit
        self.sum_w, self.sum_xy = 0.0, np.zeros(2)
        self.marker_xy = None               # current best estimate of the marker centre (world)
        self.facing = None                  # direction the marker faces, out of the wall (world)
        self.frozen = False
        self.n_sightings = 0
        self.wall_fitted = False            # False = provisional facing (wall still beyond LiDAR range)
        self._published = None              # the estimate the robot is currently driving to
        self.stage = None                   # None -> "beacon" -> "dock"
        self.min_dock_sightings = min_dock_sightings   # never lock on fewer sightings of the docking marker
        self.n_dock_sightings = 0
        self._published_stage = None

    @property
    def found(self):
        return self.marker_xy is not None and self.facing is not None

    def update(self, obs, pose, scan_world, is_dock=True):
        """obs = camera detection (dict with p_base, range), pose = EKF [x, y, th],
        scan_world = (N, 2) LiDAR hits in the world, is_dock = True for the docking marker,
        False for the beacon. Returns True if the target moved noticeably."""
        if is_dock and self.stage != "dock":
            # first sighting of the docking marker: forget everything learned from the beacon
            self.stage, self.frozen = "dock", False
            self.sum_w, self.sum_xy = 0.0, np.zeros(2)
            self.n_dock_sightings = 0
            self.marker_xy, self.facing, self.wall_fitted = None, None, False
        elif not is_dock:
            if self.stage == "dock":
                return False                                      # docking marker seen: ignore the beacon
            self.stage = "beacon"
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
        self.n_dock_sightings += self.stage == "dock"
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
            self.marker_xy = guess
            if not self.wall_fitted:
                n_b = obs.get("normal_base")
                if n_b is not None and obs["range"] < 1.5:
                    # no wall to fit (e.g. marker on a free-standing dock), but close enough that the
                    # camera's own estimate of the marker's facing direction is reliable
                    f = wrap(th + np.arctan2(n_b[1], n_b[0]))
                    self.facing = f if self.facing is None else wrap(self.facing + 0.3 * wrap(f - self.facing))
                else:
                    # far away and no wall in LiDAR range (3.5 m) yet: provisional facing = from the marker
                    # towards the robot, good enough to start driving; refined as the robot gets closer
                    self.facing = float(np.arctan2(y - guess[1], x - guess[0]))

        if (self.stage == "dock" and self.wall_fitted and self.n_dock_sightings >= self.min_dock_sightings
                and np.hypot(*(np.array(pose[:2]) - self.marker_xy)) < self.freeze_dist + self.standoff):
            self.frozen = True                                    # close to the dock: keep the target steady
        if not self.found:
            return False
        if self._published is None or self._published_stage != self.stage:
            changed = True                                        # first target, or switched beacon -> dock
        else:                                     # compare with the target in use, not the last update
            changed = (np.hypot(*(self.marker_xy - self._published[0])) > 0.03
                       or abs(wrap(self.facing - self._published[1])) > np.deg2rad(2))
        if changed:
            self._published = (self.marker_xy.copy(), self.facing)
            self._published_stage = self.stage
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