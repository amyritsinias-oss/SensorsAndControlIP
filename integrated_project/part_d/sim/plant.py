"""
Test rig plant: a TurtleBot3 that is deliberately NOT the model the
controller was designed on.

Differences from the design model (each one is something the real robot or
the MuJoCo scene will also do):
  * worn wheel   : one wheel radius off by a few %, so commanding a straight
                   line produces a yaw rate proportional to speed
  * motor lag    : wheel rates follow the command through a first-order lag
  * latency      : commands take effect `delay_steps` control periods late
  * encoders     : quantised (4096 ticks/rev, XM430 Dynamixel) plus slip noise

Ground truth lives ONLY in this class. The controller and the EKF get
encoder ticks and camera detections, never self.x.
"""
import numpy as np

from tb3_control.params import RobotParams
from tb3_control.controller import wheel_rates
from tb3_control.model import wrap


class TB3Plant:
    TICKS_PER_REV = 4096

    def __init__(self, x0, rp=None, wheel_err=0.03, err_side="R", motor_tau=0.08,
                 delay_steps=1, T=0.05, dt=0.005, slip_std=0.01, rng=None):
        self.rp = rp or RobotParams()
        self.x = np.array(x0, float)               # TRUE pose [x, y, th]
        self.T, self.dt = T, dt
        self.n_sub = int(round(T / dt))
        self.motor_tau = motor_tau
        self.rng = rng or np.random.default_rng(0)
        self.slip_std = slip_std
        r = self.rp.wheel_radius
        # true wheel radii; the controller/EKF assume r for both (they don't know wheel_err)
        self.rL = r * (1 + wheel_err) if err_side == "L" else r
        self.rR = r * (1 + wheel_err) if err_side == "R" else r
        self.wl = 0.0
        self.wr = 0.0                               # actual wheel rates
        self.queue = [(0.0, 0.0)] * delay_steps     # command latency FIFO
        self.phiL = 0.0
        self.phiR = 0.0                             # true wheel angles [rad]

    def step(self, v_cmd, w_cmd):
        """Advance one control period. Returns cumulative encoder ticks (L, R)."""
        self.queue.append((v_cmd, w_cmd))
        v_c, w_c = self.queue.pop(0)
        wl_c, wr_c = wheel_rates(v_c, w_c, self.rp)
        a = self.dt / (self.motor_tau + self.dt)
        for _ in range(self.n_sub):
            self.wl += a * (wl_c - self.wl)
            self.wr += a * (wr_c - self.wr)
            # ground contact with small random slip
            sl = 1 + self.slip_std * self.rng.standard_normal()
            sr = 1 + self.slip_std * self.rng.standard_normal()
            vl, vr = self.rL * self.wl * sl, self.rR * self.wr * sr
            v = 0.5 * (vl + vr)
            w = (vr - vl) / self.rp.track_width
            th_mid = self.x[2] + 0.5 * w * self.dt
            self.x[0] += v * np.cos(th_mid) * self.dt
            self.x[1] += v * np.sin(th_mid) * self.dt
            self.x[2] = wrap(self.x[2] + w * self.dt)
            self.phiL += self.wl * self.dt
            self.phiR += self.wr * self.dt
        k = self.TICKS_PER_REV / (2 * np.pi)
        return int(np.floor(self.phiL * k)), int(np.floor(self.phiR * k))

    def kick(self, dx, dy, dth):
        """External disturbance: the robot is bumped (truth moves, sensors don't know)."""
        self.x += np.array([dx, dy, dth])
        self.x[2] = wrap(self.x[2])

    def truth(self):
        """Ground truth - EVALUATION ONLY."""
        return self.x.copy()
