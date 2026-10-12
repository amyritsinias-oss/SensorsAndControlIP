"""
Part D controller: speed-scheduled LQI path follower with ArUco docking.

Interface (what the rest of the team's pipeline sees):

    ctrl = PathFollower(path)                 # path from tb3_control.path
    v, w, info = ctrl.step(x_hat, P)          # once per control period T
    # x_hat = EKF estimate [x, y, th] in the world frame
    # P     = EKF covariance (3x3), optional, only used to slow down when unsure
    # v, w  = applied (already saturated) linear / angular velocity command
    # IMPORTANT: give the EKF the same (v, w) or, better, the wheel encoders,
    #            so the estimator predicts with what the robot actually got.

The controller NEVER sees ground truth: its only state input is x_hat.

Control law (per step k):
    e_y, e_th, kappa  <- project x_hat onto the path
    v                  <- speed profile (curvature, stopping distance, uncertainty)
    u_fb  = -K(v) [e_y, e_th, z]          (LQI, gain scheduled on v)
    omega = v kappa + u_fb                 (curvature feed-forward + feedback)
    (v, omega) scaled together to respect wheel / turn-rate limits
    z    <- z + T e_y   unless the command saturated (conditional anti-windup)
"""
import numpy as np

from .params import RobotParams, ControllerParams
from .lqr_design import GainSchedule
from .model import wrap


def wheel_rates(v, w, rp):
    """(v, omega) -> (w_L, w_R) wheel angular rates [rad/s]."""
    half = 0.5 * rp.track_width
    return (v - w * half) / rp.wheel_radius, (v + w * half) / rp.wheel_radius


def saturate(v, w, rp):
    """Scale v and omega by the SAME factor so the commanded curvature w/v,
    i.e. the path geometry, is kept, while respecting the turn-rate limit
    and the per-wheel speed limit. Returns (v, w, saturated_flag)."""
    f = 1.0
    if abs(w) > rp.w_max:
        f = min(f, rp.w_max / abs(w))
    wl, wr = wheel_rates(v, w, rp)
    peak = max(abs(wl), abs(wr))
    if peak > rp.wheel_rate_max:
        f = min(f, rp.wheel_rate_max / peak)
    return f * v, f * w, f < 1.0 - 1e-9


class PathFollower:
    def __init__(self, path, rp=None, cp=None, integral=True, schedule=True):
        self.path = path
        self.rp = rp or RobotParams()
        self.cp = cp or ControllerParams()
        self.integral = integral
        self.schedule = schedule
        self.gains = GainSchedule(self.cp, self.rp.v_max, integral=integral)
        self.K_fixed = self.gains.K(self.cp.v_nom)      # used only when schedule=False
        self.reset()

    def reset(self):
        self.z = 0.0          # integral of e_y [m s]
        self.i = 0            # projection hint (index on path)
        self.v_prev = 0.0     # last applied speed, for the accel limit
        self.done = False

    # ------------------------------------------------------------------
    def _speed_ref(self, s, P):
        cp, path = self.cp, self.path
        # curvature preview: look ~1 s ahead so the robot slows BEFORE a bend
        ds = path.s[1] - path.s[0]
        n_ahead = int(max(cp.v_nom * 1.0, 0.2) / ds)
        k_ahead = np.max(np.abs(path.kappa[self.i:self.i + n_ahead + 1]))
        v_curve = np.sqrt(cp.a_lat_max / max(k_ahead, 1e-6))
        # can still stop at the dock with the decel limit
        s_rem = max(path.length - s, 0.0)
        v_stop = np.sqrt(2.0 * cp.a_long_max * s_rem)
        v = min(cp.v_nom, v_curve, v_stop)
        # slow down when the EKF says it is unsure of position (covariance, not truth)
        if P is not None:
            sig = np.sqrt(max(np.linalg.eigvalsh(P[:2, :2]).max(), 0.0))
            if sig > cp.sigma_slow:
                v *= cp.sigma_slow / sig
        # creep speed so the robot never stalls before the dock
        v = max(v, cp.v_creep)
        # longitudinal acceleration limit
        dv = cp.a_long_max * cp.T
        return float(np.clip(v, self.v_prev - dv, self.v_prev + dv))

    # ------------------------------------------------------------------
    def step(self, x_hat, P=None):
        cp, rp = self.cp, self.rp
        px, py, th = float(x_hat[0]), float(x_hat[1]), float(x_hat[2])
        self.i, s, e_y, th_p, kappa = self.path.project(px, py, self.i)
        e_th = wrap(th - th_p)
        s_rem = self.path.length - s

        if self.done or s_rem < cp.dock_tol:
            self.done = True
            self.v_prev = 0.0
            return 0.0, 0.0, dict(e_y=e_y, e_th=e_th, s=s, kappa=kappa, z=self.z,
                                  w_fb=0.0, sat=False, done=True,
                                  docking=True, K=np.zeros(3))

        v = self._speed_ref(s, P)
        K = self.gains.K(v) if self.schedule else self.K_fixed
        u_fb = -float(K @ np.array([e_y, e_th, self.z]))
        w = v * kappa + u_fb

        v_app, w_app, sat = saturate(v, w, rp)

        # conditional integration (anti-windup): freeze z while saturated,
        # unless integrating would pull z back towards zero
        if self.integral and (not sat or np.sign(e_y) != np.sign(self.z)):
            self.z = float(np.clip(self.z + cp.T * e_y, -cp.zint_limit, cp.zint_limit))

        self.v_prev = v_app
        info = dict(e_y=e_y, e_th=e_th, s=s, kappa=kappa, z=self.z,
                    w_fb=w_app - v_app * kappa,     # feedback part actually applied
                    sat=sat, done=False,
                    docking=s >= self.path.s_dock_start, K=K)
        return v_app, w_app, info
