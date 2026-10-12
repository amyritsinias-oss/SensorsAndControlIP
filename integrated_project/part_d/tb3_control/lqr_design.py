"""
LQI gain design (Week 6): Bryson's rule weights, discrete Riccati solution,
and a gain schedule over forward speed.
"""
import numpy as np
from scipy.linalg import solve_discrete_are

from .model import augmented_dt


def bryson(*maxima):
    """Bryson's rule: weight = 1 / (largest acceptable value)^2, diagonal."""
    return np.diag([1.0 / m ** 2 for m in maxima])


def dlqr(A, B, Q, R):
    """Infinite-horizon discrete LQR.
    P solves the DARE, K = (R + B'PB)^-1 B'PA, control u = -K x."""
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


def design_weights(cp):
    """Q and R for state [e_y, e_th, z] and input omega_fb, from ControllerParams."""
    Q = bryson(cp.ey_max, cp.eth_max, cp.zint_max)
    R = cp.rho * bryson(cp.u_max)
    return Q, R


class GainSchedule:
    """K(v) designed on a grid of speeds and linearly interpolated.

    Why schedule: the linear model's A has v in it. A gain designed at
    0.2 m/s and used at 0.03 m/s (the docking creep speed) is far too
    aggressive in heading relative to how slowly the lateral error can move,
    and a gain designed at low speed is sluggish at cruise. Re-solving the
    Riccati equation per speed keeps the closed-loop poles where the weights
    put them across the whole speed range.
    """

    def __init__(self, cp, v_max, integral=True):
        self.cp = cp
        self.integral = integral
        self.v_grid = np.linspace(cp.v_sched_min, v_max, cp.n_sched)
        Q, R = design_weights(cp)
        self.Q, self.R = Q, R
        self.K_grid = np.zeros((cp.n_sched, 3))     # k_z stays 0 when integral=False
        self.poles = []
        for i, v in enumerate(self.v_grid):
            A, B = augmented_dt(v, cp.T)
            if integral:
                K, _ = dlqr(A, B, Q, R)
                self.K_grid[i] = K.ravel()
            else:
                # plain LQR on [e_y, e_th] only (used for the integral-action ablation)
                K, _ = dlqr(A[:2, :2], B[:2], Q[:2, :2], R)
                self.K_grid[i, :2] = K.ravel()
            self.poles.append(self.closed_loop_poles(v))

    def K(self, v):
        """Gain row [k_y, k_th, k_z] at speed v (clamped to the grid)."""
        v = float(np.clip(abs(v), self.v_grid[0], self.v_grid[-1]))
        return np.array([np.interp(v, self.v_grid, self.K_grid[:, j]) for j in range(3)])

    def closed_loop_poles(self, v_design, v_actual=None):
        """Poles of the augmented loop when the gain for v_design runs at v_actual.
        Used to show what goes wrong WITHOUT scheduling."""
        v_actual = v_design if v_actual is None else v_actual
        A, B = augmented_dt(v_actual, self.cp.T)
        K = self.K(v_design).reshape(1, 3)
        if not self.integral:
            A, B, K = A[:2, :2], B[:2], K[:, :2]
        return np.linalg.eigvals(A - B @ K)
