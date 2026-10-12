"""
Reference path: a smooth curve through the planner's waypoints, joined
tangentially to a straight final approach onto the ArUco docking pose.

The LiDAR/planning part of the pipeline supplies the waypoints; this module
only turns them into something a tracking controller can use (position,
heading and curvature as functions of arc length s).
"""
import numpy as np
from scipy.interpolate import CubicSpline

from .model import wrap


def dock_poses(marker_xy, marker_normal_angle, standoff, approach_len):
    """Dock and pre-dock poses from the ArUco marker pose (world frame).

    marker_normal_angle : direction the marker FACES (out of the wall) [rad]
    standoff            : distance from marker to robot centre when docked [m]
    approach_len        : length of the straight final approach [m]
    The robot must arrive facing the marker, i.e. heading = normal + pi.
    """
    n = np.array([np.cos(marker_normal_angle), np.sin(marker_normal_angle)])
    dock_xy = np.asarray(marker_xy, float) + standoff * n
    pre_xy = dock_xy + approach_len * n
    heading = wrap(marker_normal_angle + np.pi)
    return pre_xy, dock_xy, heading


class Path:
    """Densely sampled path with arrays s, x, y, th, kappa (all length N)."""

    def __init__(self, x, y, s_dock_start=None):
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        dx, dy = np.diff(self.x), np.diff(self.y)
        ds = np.hypot(dx, dy)
        self.s = np.concatenate([[0.0], np.cumsum(ds)])
        th = np.arctan2(dy, dx)
        self.th = np.concatenate([th, [th[-1]]])
        # curvature = d(th)/ds, central difference on the unwrapped heading
        thu = np.unwrap(self.th)
        self.kappa = np.gradient(thu, self.s)
        self.length = self.s[-1]
        self.s_dock_start = self.length if s_dock_start is None else s_dock_start

    # ------------------------------------------------------------------
    @classmethod
    def from_waypoints(cls, waypoints, start_heading, pre_dock_xy, dock_xy,
                       dock_heading, ds=0.01):
        """Clamped cubic spline through [waypoints..., pre_dock], start heading
        and end heading fixed, then a straight segment pre_dock -> dock.

        Clamping the end tangent to the dock heading is what makes the join
        to the straight approach smooth in heading (no step in e_th when the
        robot switches segments).
        """
        P = np.vstack([np.asarray(waypoints, float), pre_dock_xy])
        chord = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        t0 = np.array([np.cos(start_heading), np.sin(start_heading)])
        t1 = np.array([np.cos(dock_heading), np.sin(dock_heading)])
        # chord-length parameterisation -> |dP/du| ~ 1, so unit tangents are the right scale
        csx = CubicSpline(chord, P[:, 0], bc_type=((1, t0[0]), (1, t1[0])))
        csy = CubicSpline(chord, P[:, 1], bc_type=((1, t0[1]), (1, t1[1])))
        u = np.linspace(0, chord[-1], int(chord[-1] / ds * 3) + 2)   # oversample, then resample by arc length
        xs, ys = csx(u), csy(u)
        # resample the spline uniformly in arc length
        sl = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
        s_new = np.arange(0.0, sl[-1], ds)
        xs, ys = np.interp(s_new, sl, xs), np.interp(s_new, sl, ys)
        # straight final approach
        L_app = np.linalg.norm(np.asarray(dock_xy) - np.asarray(pre_dock_xy))
        n_app = max(int(L_app / ds), 2)
        a = np.linspace(0, 1, n_app + 1)[1:]
        xa = pre_dock_xy[0] + a * (dock_xy[0] - pre_dock_xy[0])
        ya = pre_dock_xy[1] + a * (dock_xy[1] - pre_dock_xy[1])
        path = cls(np.concatenate([xs, xa]), np.concatenate([ys, ya]),
                   s_dock_start=sl[-1])
        # the straight segment has zero curvature by construction; kill the
        # numerical blip that np.gradient makes at the join
        path.kappa[path.s >= path.s_dock_start] = 0.0
        return path

    # ------------------------------------------------------------------
    def project(self, px, py, i_hint=0, back=50, fwd=300):
        """Closest path sample to (px, py), searched in a window around i_hint.

        The window stops the projection jumping to a different part of the
        path where the path passes close to itself. Returns
            i, s, e_y (left +), th_path, kappa
        with s and e_y refined by projecting onto the local tangent.
        """
        lo, hi = max(0, i_hint - back), min(len(self.s), i_hint + fwd)
        d2 = (self.x[lo:hi] - px) ** 2 + (self.y[lo:hi] - py) ** 2
        i = lo + int(np.argmin(d2))
        c, sn = np.cos(self.th[i]), np.sin(self.th[i])
        dx, dy = px - self.x[i], py - self.y[i]
        s = self.s[i] + c * dx + sn * dy        # along-track
        e_y = -sn * dx + c * dy                 # cross-track, + = left
        return i, s, e_y, self.th[i], self.kappa[i]
