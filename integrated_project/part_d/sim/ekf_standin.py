"""
STAND-IN EKF (Week 7 structure) so the controller can be tested in closed
loop before the team's estimator (Part C) is ready. Swap it for the team's
EKF: the controller only needs .x (3,) and .P (3x3).

Predict : wheel-encoder odometry (the motion the wheels actually made)
Update  : ArUco range-bearing to known marker positions
It knows the NOMINAL wheel radius only - never the plant's wheel error.
"""
import numpy as np

from tb3_control.model import wrap


class OdomArucoEKF:
    # k_odo = 0.15 is deliberately inflated: the worn wheel is a BIAS, not white
    # noise, and with k_odo = 0.03 the filter became overconfident in heading,
    # gated out correct marker updates and diverged (NEES inside 95% bound: 17%).
    # At 0.15 it is ~89% over 20 seeds. A proper fix (estimation lead's call) is
    # to add the wheel-radius ratio as an EKF state.
    def __init__(self, x0, P0, rp, markers, camera_sigma, k_odo=0.15, sig_odo_floor=1e-4,
                 gate=9.21):
        self.x = np.array(x0, float)
        self.P = np.array(P0, float)
        self.rp = rp
        self.markers = markers
        self.camera_sigma = camera_sigma        # function r -> (sig_r, sig_b)
        self.k_odo = k_odo                      # wheel-angle noise grows with |dphi|
        self.sig_odo_floor = sig_odo_floor
        self.gate = gate                        # chi2(2) 99% innovation gate
        self.ticks_prev = None
        self.k = 2 * np.pi / 4096

    def predict(self, ticks):
        if self.ticks_prev is None:
            self.ticks_prev = ticks
            return
        dphiL = (ticks[0] - self.ticks_prev[0]) * self.k
        dphiR = (ticks[1] - self.ticks_prev[1]) * self.k
        self.ticks_prev = ticks
        r, L = self.rp.wheel_radius, self.rp.track_width
        ds = 0.5 * r * (dphiL + dphiR)
        dth = r * (dphiR - dphiL) / L
        th_m = self.x[2] + 0.5 * dth
        c, s = np.cos(th_m), np.sin(th_m)
        self.x = self.x + np.array([ds * c, ds * s, dth])
        self.x[2] = wrap(self.x[2])
        # Jacobians wrt state and wrt wheel increments
        F = np.array([[1, 0, -ds * s], [0, 1, ds * c], [0, 0, 1]])
        # d(th_m)/d(dphiL) = -r/(2L), d(th_m)/d(dphiR) = +r/(2L)
        a = r / (2 * L)
        G = np.array([[0.5 * r * c + ds * s * a, 0.5 * r * c - ds * s * a],
                      [0.5 * r * s - ds * c * a, 0.5 * r * s + ds * c * a],
                      [-r / L, r / L]])
        sL = self.k_odo * abs(dphiL) + self.sig_odo_floor
        sR = self.k_odo * abs(dphiR) + self.sig_odo_floor
        Qu = np.diag([sL ** 2, sR ** 2])
        self.P = F @ self.P @ F.T + G @ Qu @ G.T

    def update(self, detections):
        nis = []
        for mid, r_m, b_m in detections:
            mx, my, _ = self.markers[mid]
            dx, dy = mx - self.x[0], my - self.x[1]
            q = dx * dx + dy * dy
            r = np.sqrt(q)
            zhat = np.array([r, wrap(np.arctan2(dy, dx) - self.x[2])])
            H = np.array([[-dx / r, -dy / r, 0.0],
                          [dy / q, -dx / q, -1.0]])
            sr, sb = self.camera_sigma(r_m)
            R = np.diag([sr ** 2, sb ** 2])
            nu = np.array([r_m - zhat[0], wrap(b_m - zhat[1])])
            S = H @ self.P @ H.T + R
            d2 = float(nu @ np.linalg.solve(S, nu))
            if d2 > self.gate:            # reject outliers / wrong associations
                continue
            K = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + K @ nu
            self.x[2] = wrap(self.x[2])
            IKH = np.eye(3) - K @ H
            self.P = IKH @ self.P @ IKH.T + K @ R @ K.T     # Joseph form
            nis.append(d2)
        return nis
