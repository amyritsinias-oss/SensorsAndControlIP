"""
State-space model used for control design (Week 5 material, applied to the TB3).

Plant: differential-drive robot, treated as a unicycle
    x_dot  = v cos(th)
    y_dot  = v sin(th)
    th_dot = omega
with the wheel speeds mapped from (v, omega) by
    w_R = (v + omega L/2) / r,   w_L = (v - omega L/2) / r.

Path-error coordinates (Frenet frame of the reference path at the closest point):
    e_y  : signed cross-track error, + when the robot is LEFT of the path   [m]
    e_th : heading error th - th_path, wrapped to (-pi, pi]                   [rad]
Exact error dynamics, path curvature kappa:
    e_y_dot  = v sin(e_th)
    e_th_dot = omega - kappa * v cos(e_th) / (1 - kappa e_y)

Linearise about e_y = e_th = 0 with the feed-forward omega_ff = v kappa removed
from the input (u = omega - v kappa is the FEEDBACK turn rate):
    d/dt [e_y ]   [0 v] [e_y ]   [0]
         [e_th] = [0 0] [e_th] + [1] u
This is a double integrator in distance travelled. A depends on v, so the
design is repeated over a grid of speeds (gain scheduling, see lqr_design.py).

Integral action is added in DISCRETE time exactly as the controller code
implements it (forward-Euler accumulation, Week 6 slide "offset-free tracking"):
    z(k+1) = z(k) + T e_y(k)
so the augmented model the gain is designed on is the model the code runs.
"""
import numpy as np
from scipy.linalg import expm


def error_model_ct(v):
    """Continuous-time linearised path-error model (A, B) at speed v."""
    A = np.array([[0.0, v],
                  [0.0, 0.0]])
    B = np.array([[0.0],
                  [1.0]])
    return A, B


def c2d_zoh(A, B, T):
    """Zero-order-hold discretisation via the block-matrix exponential.
    expm([[A, B],[0, 0]] T) = [[Ad, Bd],[0, I]]."""
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * T)
    return Md[:n, :n], Md[:n, n:]


def augmented_dt(v, T):
    """Discrete augmented model, state [e_y, e_th, z], input u = omega_fb.

    Plant part is the exact ZOH of the linear model:
        Ad = [[1, vT], [0, 1]],  Bd = [[v T^2 / 2], [T]]
    Integrator row matches the code: z(k+1) = z(k) + T e_y(k).
    """
    A, B = error_model_ct(v)
    Ad, Bd = c2d_zoh(A, B, T)
    Aa = np.zeros((3, 3))
    Aa[:2, :2] = Ad
    Aa[2, 0] = T          # z += T * e_y
    Aa[2, 2] = 1.0
    Ba = np.zeros((3, 1))
    Ba[:2, :] = Bd
    return Aa, Ba


def ctrb(A, B):
    """Controllability matrix [B AB A^2B ... A^(n-1)B]."""
    n = A.shape[0]
    cols = [B]
    for _ in range(n - 1):
        cols.append(A @ cols[-1])
    return np.hstack(cols)


def controllability_report(v, T):
    """Rank and conditioning of the augmented discrete pair at speed v.
    At v = 0 the rank drops: a stationary robot can turn but cannot reduce
    a lateral error (Week 5 unicycle result)."""
    Aa, Ba = augmented_dt(v, T)
    C = ctrb(Aa, Ba)
    s = np.linalg.svd(C, compute_uv=False)
    return {"v": v, "rank": int(np.linalg.matrix_rank(C, tol=1e-12)),
            "n": Aa.shape[0], "sigma_min": float(s[-1]), "sigma_max": float(s[0])}


def wrap(a):
    """Wrap an angle (or array of angles) to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi
