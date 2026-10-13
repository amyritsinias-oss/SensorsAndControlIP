#lidar scans at 5hz and constantly updates the occupancy grid, replanning if the current path is blocked
import numpy as np

from astar import plan


class Navigator:
    def __init__(self, lidar, grid, path_class, pre_dock, dock, dock_heading,
                 inflate=0.30, robot_radius=0.22, stop_dist=0.25, stop_half_angle=np.deg2rad(30),
                 cooldown_scans=5):
        self.lidar, self.grid = lidar, grid
        self.Path = path_class          
        self.pre, self.dock, self.dock_heading = np.asarray(pre_dock), np.asarray(dock), dock_heading
        self.inflate = inflate   # planning margin (robot radius + spline cut-in)
        self.robot_radius = robot_radius  # collision check margin for the current path
        self.stop_dist = stop_dist
        self.front = np.abs((lidar.angles + np.pi) % (2 * np.pi) - np.pi) < stop_half_angle
        self.path = None
        self.waypoints = None
        self.n_replans = 0
        self.cooldown_scans = cooldown_scans # min scans between replans (5 scans = 1 s at 5 Hz)
        self.scans_since_replan = cooldown_scans

    def replan(self, pose):
        waypoints, _ = plan(self.grid, pose[:2], self.pre, self.inflate)
        if waypoints is None:
            return False      # no route, keep the old path, the safety stop protects it
        # from_waypoints appends the pre-dock point itself and adds the straight approach to the dock
        self.path = self.Path.from_waypoints(waypoints[:-1], pose[2], self.pre, self.dock, self.dock_heading)
        self.waypoints = waypoints
        self.n_replans += 1
        return True

    def path_blocked(self, i_now):
        
        if self.path is None:
            return True
        blocked = self.grid.inflated(self.robot_radius)
        ahead = slice(i_now, int(np.searchsorted(self.path.s, self.path.s_dock_start)))   # final approach excluded
        rows, cols = self.grid.to_cell(self.path.x[ahead], self.path.y[ahead])
        ok = self.grid.in_bounds(rows, cols)
        return bool(blocked[rows[ok], cols[ok]].any())

    def step(self, ranges, pose, i_now=0):

        self.grid.update(self.lidar, ranges, pose)
        self.scans_since_replan += 1
        new_path = None
        # cooldown, an obstacle is revealed over several scans, and replanning on every one of them
        # would reset the controller repeatedly. The safety stop covers anything urgent in the meanwhile
        if self.scans_since_replan >= self.cooldown_scans and self.path_blocked(i_now) and self.replan(pose):
            new_path = self.path
            self.scans_since_replan = 0
        stop = bool(np.any(ranges[self.front] < self.stop_dist))
        return new_path, stop