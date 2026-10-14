"""
TurtleBot3 pose EKF: wheel-encoder odometry + ArUco range-bearing, with the
two wheel-radius scale errors estimated as extra states.

Replaces sim/ekf_standin.py. Same drop-in interface the controller needs:
    ekf.x  -> (3,)  pose estimate [x, y, th]       (world frame)
    ekf.P  -> (3,3) pose covariance
plus ekf.xa / ekf.Pa for the full 5-state vector and covariance.

State
    xa = [x, y, th, bL, bR]
    bL, bR : fractional wheel-radius errors, true radius = r_nom * (1 + b)
Why the bias states: the stand-in treated a worn wheel (a constant BIAS) as
white noise, so it had to inflate the odometry noise 5x (k_odo = 0.15) to stay
consistent, and the leftover bias in the estimate caused the D2 mean
cross-track failures. Here the bias is estimated from the marker updates, so
the odometry noise can be set to what the encoders and slip actually do.

Predict : wheel increments dphiL, dphiR (encoder ticks OR joint angles)
          ds_i = r (1 + b_i) dphi_i
          ds   = (dsL + dsR) / 2,  dth = (dsR - dsL) / L,  midpoint heading
Update  : ArUco detection (id, range, bearing) to a marker at a known (x, y),
          measured from the camera, which may sit cam_offset ahead of the
          base centre (Waffle Pi camera is ~0.07 m forward; the sim uses 0).
          Chi-square innovation gate, Joseph-form covariance update, and a
          recovery step that re-opens the pose covariance if every recent
          detection has been gated out (e.g. the robot was bumped).

Run this file directly to tune it:  python -m sim.ekf  [--seeds 20]
"""
from dataclasses import dataclass, asdict

import numpy as np

from tb3_control.model import wrap


# =============================================================================
#  TUNING - every knob lives here.  See tune() at the bottom for the sweep.
# =============================================================================
@dataclass
class EKFTuning:
    # ---- odometry (predict) noise --------------------------------------------
    # Per-wheel angle noise per step:  sigma_i = k_odo * |dphi_i| + sig_odo_floor  [rad]
    #   k_odo         : slip / scrub, proportional to how far the wheel turned.
    #                   Plant slip is 1% per 5 ms substep -> ~0.3% per 50 ms step,
    #                   so 0.01-0.03 is realistic with margin for unmodelled
    #                   effects (motor lag between ticks, carpet, bumps).
    #                   Stand-in needed 0.15 because it had no bias states.
    #   sig_odo_floor : encoder quantisation. One tick = 2*pi/4096 = 1.5e-3 rad;
    #                   the difference of two quantised readings has
    #                   sigma ~ 1 tick / sqrt(6) ~ 6e-4 rad.
    k_odo: float = 0.01               # tuned (sweep: flat 0.01-0.05 once bias states are on)
    sig_odo_floor: float = 6e-4
    # Extra additive pose noise per step (unmodelled: latency between ticks and
    # pose, chassis flex). Keep small; raise it if NEES is high on straights.
    q_xy: float = 1e-4                 # [m]   per step, 1 sigma
    q_th: float = 0.0                  # [rad] per step, 1 sigma (tuned: 0 gives NEES95 closest to 95%)

    # ---- wheel-radius bias states -------------------------------------------
    estimate_bias: bool = True         # False -> plain 3-state EKF (for ablation)
    sig_b0: float = 0.05               # initial 1 sigma on each wheel scale (covers +/-4.5% wear)
    sig_b_rw: float = 5e-4             # tuned. bias random walk, 1 sigma per sqrt(m) travelled by that wheel
                                       #   (tyre pressure / load / floor changes). Too small -> bias
                                       #   freezes early and can't track; too big -> noisy bias.
    b_clip: float = 0.15               # hard limit on |b| (physically a wheel isn't 15% off)

    # ---- camera (update) ----------------------------------------------------
    r_scale: float = 1.0               # multiplies the camera's sigma (both range and bearing).
                                       #   >1 if the real detector is worse than its datasheet model.
    sig_r_min: float = 0.005           # floor on range sigma [m]
    sig_b_min: float = np.deg2rad(0.5) # floor on bearing sigma [rad]
    cam_offset: float = 0.0            # camera ahead of base centre [m] (MuJoCo/real Waffle Pi ~0.073)

    # ---- outlier gate + recovery --------------------------------------------
    gate: float = 9.21                 # chi2(2) 99%. 13.8 = 99.9%. Lower = stricter.
    recover_after: int = 4             # this many gated detections in a row with none accepted ...
    recover_sigma_xy: float = 0.10     # ... re-opens pose covariance by this much (1 sigma, m)
    recover_sigma_th: float = np.deg2rad(10.0)   # ... and this (rad), then retries the detection


# Last sweep (20 closed-loop seeds 5000-5019, python -m sim.ekf):
#   setting                          rms pos err   NEES95   |mean e_y|   dock
#   stand-in equivalent (no bias)    2.80 cm       88 %     0.89 cm      100 %
#   3-state, realistic noise         3.08 cm       32 %     1.01 cm       90 %   <- overconfident
#   THIS (defaults above)            2.43 cm       96 %     0.84 cm      100 %
# Estimated wheel bias converges to within ~0.2 % of the true wheel error.

# Named presets so the report can quote exactly what was run.
PRESETS = {
    "default": EKFTuning(),
    # What the stand-in effectively was: no bias states, inflated odometry noise.
    "standin_equiv": EKFTuning(estimate_bias=False, k_odo=0.15, sig_odo_floor=1e-4,
                               q_xy=0.0, q_th=0.0, recover_after=10 ** 9),
    # Plain 3-state EKF with realistic noise - shows why the bias states are needed.
    "no_bias": EKFTuning(estimate_bias=False),
    # For MuJoCo / the real robot: camera offset and a bit more caution.
    "mujoco": EKFTuning(cam_offset=0.073, r_scale=1.5, k_odo=0.04),
}


class TB3EKF:
    TICKS_PER_REV = 4096

    def __init__(self, x0, P0, rp, markers, camera_sigma, tuning=None, b0=(0.0, 0.0)):
        """
        x0, P0       : initial pose (3,) and pose covariance (3x3)
        rp           : RobotParams (nominal wheel radius / track width ONLY)
        markers      : dict id -> (x, y, facing_angle)
        camera_sigma : function range -> (sig_r, sig_b)
        tuning       : EKFTuning (or a preset name); default EKFTuning()
        b0           : initial wheel-scale guess (e.g. from a calibration run)
        """
        if isinstance(tuning, str):
            tuning = PRESETS[tuning]
        self.tp = tuning or EKFTuning()
        self.rp = rp
        self.markers = markers
        self.camera_sigma = camera_sigma

        self.xa = np.zeros(5)
        self.xa[:3] = np.asarray(x0, float)
        self.xa[3:] = b0
        self.Pa = np.zeros((5, 5))
        self.Pa[:3, :3] = np.asarray(P0, float)
        sb0 = self.tp.sig_b0 if self.tp.estimate_bias else 0.0
        self.Pa[3, 3] = self.Pa[4, 4] = sb0 ** 2

        self.ticks_prev = None
        self.phi_prev = None
        self.k_tick = 2 * np.pi / self.TICKS_PER_REV
        self.n_rejected = 0            # consecutive gated detections
        self.n_recoveries = 0
        self.last_nis = []

    # ---- interface the controller uses -----------------------------------------
    @property
    def x(self):
        return self.xa[:3].copy()

    @x.setter
    def x(self, value):
        self.xa[:3] = np.asarray(value, float)
        self.xa[2] = wrap(self.xa[2])

    @property
    def P(self):
        return self.Pa[:3, :3].copy()

    @property
    def bias(self):
        """Estimated wheel-radius errors (bL, bR) and their 1 sigma."""
        return self.xa[3:].copy(), np.sqrt(np.diag(self.Pa)[3:])

    # ---- predict ---------------------------------------------------------------
    def predict(self, ticks):
        """Cumulative encoder ticks (L, R), as TB3Plant.step returns."""
        if self.ticks_prev is None:
            self.ticks_prev = ticks
            return
        dL = (ticks[0] - self.ticks_prev[0]) * self.k_tick
        dR = (ticks[1] - self.ticks_prev[1]) * self.k_tick
        self.ticks_prev = ticks
        self._propagate(dL, dR)

    def predict_angles(self, phiL, phiR):
        """Cumulative wheel joint angles [rad] (MuJoCo data.qpos of the wheel hinges)."""
        if self.phi_prev is None:
            self.phi_prev = (phiL, phiR)
            return
        dL, dR = phiL - self.phi_prev[0], phiR - self.phi_prev[1]
        self.phi_prev = (phiL, phiR)
        self._propagate(dL, dR)

    def _propagate(self, dphiL, dphiR):
        tp = self.tp
        r, L = self.rp.wheel_radius, self.rp.track_width
        th, bL, bR = self.xa[2], self.xa[3], self.xa[4]
        dsL, dsR = r * (1 + bL) * dphiL, r * (1 + bR) * dphiR
        ds = 0.5 * (dsL + dsR)
        dth = (dsR - dsL) / L
        th_m = th + 0.5 * dth
        c, s = np.cos(th_m), np.sin(th_m)

        self.xa[0] += ds * c
        self.xa[1] += ds * s
        self.xa[2] = wrap(th + dth)

        # d(pose)/d(dsL) and d(pose)/d(dsR)  (midpoint heading depends on both)
        JL = np.array([0.5 * c + ds * s / (2 * L), 0.5 * s - ds * c / (2 * L), -1.0 / L])
        JR = np.array([0.5 * c - ds * s / (2 * L), 0.5 * s + ds * c / (2 * L), 1.0 / L])

        F = np.eye(5)
        F[0, 2], F[1, 2] = -ds * s, ds * c
        if tp.estimate_bias:
            F[:3, 3] = JL * r * dphiL          # dsL/dbL = r dphiL
            F[:3, 4] = JR * r * dphiR

        # noise on the wheel angles -> pose, via dsi/dphii = r (1 + bi)
        G = np.zeros((5, 2))
        G[:3, 0] = JL * r * (1 + bL)
        G[:3, 1] = JR * r * (1 + bR)
        sL = tp.k_odo * abs(dphiL) + tp.sig_odo_floor
        sR = tp.k_odo * abs(dphiR) + tp.sig_odo_floor

        Q = G @ np.diag([sL ** 2, sR ** 2]) @ G.T
        Q[0, 0] += tp.q_xy ** 2
        Q[1, 1] += tp.q_xy ** 2
        Q[2, 2] += tp.q_th ** 2
        if tp.estimate_bias:
            Q[3, 3] += tp.sig_b_rw ** 2 * abs(dsL)
            Q[4, 4] += tp.sig_b_rw ** 2 * abs(dsR)

        self.Pa = F @ self.Pa @ F.T + Q
        self.Pa = 0.5 * (self.Pa + self.Pa.T)

    # ---- update ----------------------------------------------------------------
    def _measure(self, mx, my):
        """Predicted (range, bearing) from the camera and its 2x5 Jacobian."""
        x, y, th = self.xa[:3]
        co = self.tp.cam_offset
        c, s = np.cos(th), np.sin(th)
        dx, dy = mx - (x + co * c), my - (y + co * s)
        q = dx * dx + dy * dy
        rng = np.sqrt(q)
        zhat = np.array([rng, wrap(np.arctan2(dy, dx) - th)])
        H = np.zeros((2, 5))
        H[0, :3] = [-dx / rng, -dy / rng, co * (dx * s - dy * c) / rng]
        H[1, :3] = [dy / q, -dx / q, -co * (dy * s + dx * c) / q - 1.0]
        return zhat, H

    def update(self, detections):
        """detections: list of (marker_id, range, bearing). Returns NIS of accepted ones."""
        tp = self.tp
        nis = []
        for mid, r_m, b_m in detections:
            if mid not in self.markers:
                continue
            mx, my = self.markers[mid][:2]
            sr, sb = self.camera_sigma(r_m)
            sr = max(sr * tp.r_scale, tp.sig_r_min)
            sb = max(sb * tp.r_scale, tp.sig_b_min)
            R = np.diag([sr ** 2, sb ** 2])

            for attempt in (0, 1):
                zhat, H = self._measure(mx, my)
                nu = np.array([r_m - zhat[0], wrap(b_m - zhat[1])])
                S = H @ self.Pa @ H.T + R
                d2 = float(nu @ np.linalg.solve(S, nu))
                if d2 <= tp.gate:
                    break
                self.n_rejected += 1
                if attempt == 0 and self.n_rejected >= tp.recover_after:
                    self._reopen()             # lost: widen pose covariance and retry once
                    continue
                d2 = None
                break
            if d2 is None:
                continue

            K = self.Pa @ H.T @ np.linalg.inv(S)
            self.xa = self.xa + K @ nu
            self.xa[2] = wrap(self.xa[2])
            self.xa[3:] = np.clip(self.xa[3:], -tp.b_clip, tp.b_clip)
            IKH = np.eye(5) - K @ H
            self.Pa = IKH @ self.Pa @ IKH.T + K @ R @ K.T          # Joseph form
            self.Pa = 0.5 * (self.Pa + self.Pa.T)
            self.n_rejected = 0
            nis.append(d2)
        self.last_nis = nis
        return nis

    def _reopen(self):
        tp = self.tp
        self.Pa[0, 0] += tp.recover_sigma_xy ** 2
        self.Pa[1, 1] += tp.recover_sigma_xy ** 2
        self.Pa[2, 2] += tp.recover_sigma_th ** 2
        self.n_rejected = 0
        self.n_recoveries += 1


# =============================================================================
#  TUNING SWEEP
#  Closed-loop trials (plant -> EKF -> controller) over seeds disjoint from
#  the Part D Monte Carlo. Scores each setting on:
#    rms_pos  : RMS true-vs-estimated position error   (accuracy)
#    nees95   : fraction of steps with NEES < 7.81      (consistency, want ~0.95;
#               much lower = overconfident, ~1.00 = too cautious)
#    |mean_ey|: mean true cross-track (D2 - the bias the stand-in left behind)
#    success  : docked within tolerance
#  Selection rule is in tune(); results of the last sweep are recorded on EKFTuning.
# =============================================================================
def _trial(args):
    seed, tuning = args
    from sim.scenario import run_trial
    L = run_trial(seed=seed, ekf_tuning=tuning)
    return dict(rms_pos=L["rms_est_pos"], nees95=L["nees_in95"], mean_ey=abs(L["mean_ey"]),
                rms_ey=L["rms_ey"], success=float(L["success"]))


def evaluate(tuning, seeds, pool=None):
    jobs = [(s, tuning) for s in seeds]
    rows = list(pool.map(_trial, jobs)) if pool else [_trial(j) for j in jobs]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} | \
           {"worst_mean_ey": float(np.max([r["mean_ey"] for r in rows]))}


def tune(seeds=range(5000, 5020), grid=None, verbose=True):
    """Grid search over the main knobs. Returns (best EKFTuning, table rows)."""
    import itertools
    from dataclasses import replace
    from multiprocessing import Pool

    grid = grid or {
        "k_odo": [0.01, 0.03, 0.05],
        "sig_b_rw": [5e-4, 2e-3],
        "q_th": [0.0, np.deg2rad(0.02), np.deg2rad(0.05)],
        "r_scale": [1.0, 1.3],
    }
    keys = list(grid)
    rows = []
    with Pool() as pool:
        for name in ("standin_equiv", "no_bias"):
            res = evaluate(PRESETS[name], seeds, pool)
            rows.append(({"preset": name}, res))
            if verbose:
                _print_row({"preset": name}, res)
        for vals in itertools.product(*(grid[k] for k in keys)):
            setting = dict(zip(keys, vals))
            res = evaluate(replace(EKFTuning(), **setting), seeds, pool)
            rows.append((setting, res))
            if verbose:
                _print_row(setting, res)

    # Selection: every trial docks, accuracy within 3% of the best, then the most
    # CONSISTENT one (NEES95 closest to 0.95). Accuracy differences under ~3% are
    # within seed-to-seed noise, while an over- or under-confident P misleads the
    # controller's uncertainty-aware speed and the outlier gate.
    cand = [(s, r) for s, r in rows if "preset" not in s and r["success"] == 1.0] \
        or [sr for sr in rows if "preset" not in sr[0]]
    best_rms = min(r["rms_pos"] for _, r in cand)
    near = [(s, r) for s, r in cand if r["rms_pos"] <= 1.03 * best_rms]
    pick = min(near, key=lambda sr: abs(sr[1]["nees95"] - 0.95))
    best = replace(EKFTuning(), **pick[0])
    if verbose:
        print("\nBEST:", pick[0], " ->  rms_pos %.2f cm, NEES95 %.0f%%, |mean e_y| %.2f cm"
              % (100 * pick[1]["rms_pos"], 100 * pick[1]["nees95"], 100 * pick[1]["mean_ey"]))
        print("Full tuning:", asdict(best))
    return best, rows


def _print_row(setting, r):
    setting = {k: (round(float(np.rad2deg(v)), 3) if k == "q_th" else v) for k, v in setting.items()}
    print("%-70s rms_pos %5.2f cm | NEES95 %4.0f%% | |mean e_y| %4.2f cm (worst %4.2f) | "
          "rms e_y %4.2f cm | dock %3.0f%%"
          % (setting, 100 * r["rms_pos"], 100 * r["nees95"], 100 * r["mean_ey"],
             100 * r["worst_mean_ey"], 100 * r["rms_ey"], 100 * r["success"]))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Tune the TB3 EKF in closed loop.")
    ap.add_argument("--seeds", type=int, default=20, help="trials per setting")
    a = ap.parse_args()
    tune(seeds=range(5000, 5000 + a.seeds))
