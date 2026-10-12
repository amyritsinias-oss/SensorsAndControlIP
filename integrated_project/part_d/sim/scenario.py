"""
Test scenario and closed-loop trial runner.

Room 5 m x 4 m with two shelving obstacles. The waypoints stand in for the
output of the team's LiDAR planner; the charging station is an ArUco marker
on the east wall. Other markers on the walls let the EKF correct drift.

run_trial() wires:  plant -> encoders/camera -> EKF -> controller -> plant
and logs ground truth SEPARATELY, only for evaluation.
"""
import numpy as np

from tb3_control import RobotParams, ControllerParams, Path, PathFollower, dock_poses, wrap
from .plant import TB3Plant
from .camera import ArucoCamera
from .ekf_standin import OdomArucoEKF

ROOM = (5.0, 4.0)
OBSTACLES = [(1.5, 0.0, 2.0, 2.2),          # (x0, y0, x1, y1) shelving blocks
             (3.0, 1.8, 3.5, 4.0)]
DOCK_ID = 0
MARKERS = {                                  # id: (x, y, angle the marker faces)
    DOCK_ID: (5.0, 0.9, np.pi),              # charging station, east wall
    1: (0.0, 2.0, 0.0),                      # west wall
    2: (1.0, 4.0, -np.pi / 2),               # north wall
    3: (2.5, 4.0, -np.pi / 2),
    4: (5.0, 3.0, np.pi),                    # east wall, north end
    5: (2.5, 0.0, np.pi / 2),                # south wall
    6: (0.6, 0.0, np.pi / 2),
}
START = np.array([0.6, 0.5, np.pi / 2])
WAYPOINTS = [(0.6, 0.5), (0.7, 2.4), (1.3, 3.15), (2.4, 2.9), (2.5, 1.4), (3.0, 0.9)]
DOCK_STANDOFF = 0.35      # marker -> robot centre when docked [m] (robot front ~0.2 m + contacts)
APPROACH_LEN = 0.70       # straight final approach [m]
DOCK_TOL = dict(lat=0.02, lon=0.03, head=np.deg2rad(5.0))   # success criteria at the dock


def build_path(approach_len=APPROACH_LEN):
    m = MARKERS[DOCK_ID]
    pre, dock, head = dock_poses(m[:2], m[2], DOCK_STANDOFF, approach_len)
    path = Path.from_waypoints(WAYPOINTS, START[2], pre, dock, head)
    return path, np.array([dock[0], dock[1], head])


def run_trial(seed=0, cp=None, rp=None, wheel_err=None, x0_err=None, integral=True,
              schedule=True, delay_steps=1, motor_tau=0.08, kick=None, kick_at_dock=None, path=None,
              t_max=120.0, controller_state_source="ekf"):
    """One closed-loop run. Returns a dict of logged arrays + summary metrics.

    controller_state_source="truth" is an EVALUATION BASELINE ONLY (the Week 8
    'cheating with ground truth' comparison). The delivered pipeline uses "ekf".
    """
    rng = np.random.default_rng(seed)
    rp = rp or RobotParams()
    cp = cp or ControllerParams()
    if path is None:
        path, dock = build_path()
    else:
        dock = np.array([path.x[-1], path.y[-1], path.th[-1]])
    # personalised disturbances for this trial
    if wheel_err is None:
        wheel_err = rng.choice([-1, 1]) * rng.uniform(0.02, 0.045)
    if x0_err is None:
        x0_err = np.array([rng.normal(0, 0.05), rng.normal(0, 0.05), rng.normal(0, np.deg2rad(5))])
    x_true0 = START + x0_err

    plant = TB3Plant(x_true0, rp, wheel_err=wheel_err, err_side="R", motor_tau=motor_tau,
                     delay_steps=delay_steps, T=cp.T, rng=np.random.default_rng(seed + 1000))
    cam = ArucoCamera(MARKERS, rng=np.random.default_rng(seed + 2000))
    # EKF starts at the NOMINAL start pose; P0 covers the start uncertainty
    P0 = np.diag([0.05 ** 2, 0.05 ** 2, np.deg2rad(5) ** 2])
    ekf = OdomArucoEKF(START, P0, rp, MARKERS, cam.sigma)
    ctrl = PathFollower(path, rp, cp, integral=integral, schedule=schedule)

    keys = ["t", "x", "y", "th", "xh", "yh", "thh", "Pxx", "Pyy", "Ptt", "ey_true", "eth_true",
            "ey_est", "eth_est", "v", "w", "w_fb", "sat", "dock", "s", "z", "nees"]
    log = {k: [] for k in keys}
    i_true = 0
    kicked = False
    ticks = plant.step(0.0, 0.0)
    ekf.predict(ticks)
    t = 0.0
    while t < t_max:
        # ---- sensing + estimation (no truth) ---------------------------
        dets = cam.detect(plant.x, t)          # the camera itself sees the world
        ekf.update(dets)
        # ---- control on the estimate ------------------------------------
        xc = plant.truth() if controller_state_source == "truth" else ekf.x
        v, w, info = ctrl.step(xc, ekf.P)
        # ---- evaluation only: truth ------------------------------------
        xt = plant.truth()
        i_true, _, ey_t, thp_t, _ = path.project(xt[0], xt[1], i_true)
        e = xt - ekf.x
        e[2] = wrap(e[2])
        nees = float(e @ np.linalg.solve(ekf.P, e))
        for k, val in zip(keys, [t, xt[0], xt[1], xt[2], ekf.x[0], ekf.x[1], ekf.x[2],
                                 ekf.P[0, 0], ekf.P[1, 1], ekf.P[2, 2], ey_t, wrap(xt[2] - thp_t),
                                 info["e_y"], info["e_th"], v, w, info["w_fb"], info["sat"],
                                 info["docking"], info["s"], info["z"], nees]):
            log[k].append(val)
        if info["done"]:
            break
        # ---- actuate -----------------------------------------------------
        if kick is not None and abs(t - kick[0]) < cp.T / 2:
            plant.kick(*kick[1:])
        if kick_at_dock is not None and info["docking"] and not kicked:
            # Reference jump as the final approach starts: e.g. the camera re-estimates
            # the dock pose when close. Shifting truth AND estimate by the same amount is
            # equivalent to moving the path, so the estimator stays consistent and the
            # test isolates the controller's ability to correct at low speed.
            plant.kick(*kick_at_dock)
            ekf.x = ekf.x + np.asarray(kick_at_dock)
            ekf.x[2] = wrap(ekf.x[2])
            kicked = True
        ticks = plant.step(v, w)
        ekf.predict(ticks)
        t += cp.T

    log = {k: np.asarray(v) for k, v in log.items()}
    log.update(metrics(log, plant.truth(), dock, ctrl.done, cp))
    log["wheel_err"] = wheel_err
    log["path"] = path
    log["dock_pose"] = dock
    return log


def metrics(log, x_final, dock, done, cp):
    """Tracking metrics on TRUE cross-track error, plus docking accuracy."""
    follow = ~log["dock"]                     # path-following phase (before final approach)
    ey = log["ey_true"]
    out = {}
    out["rms_ey"] = float(np.sqrt(np.mean(ey[follow] ** 2)))
    out["mean_ey"] = float(np.mean(ey[follow]))
    out["mean_ey_est"] = float(np.mean(log["ey_est"][follow]))   # what the integrator sees
    out["max_ey"] = float(np.max(np.abs(ey)))
    out["rms_wfb"] = float(np.sqrt(np.mean(log["w_fb"][follow] ** 2)))
    out["sat_frac"] = float(np.mean(log["sat"]))
    out["rms_est_pos"] = float(np.sqrt(np.mean((log["x"] - log["xh"]) ** 2 + (log["y"] - log["yh"]) ** 2)))
    out["nees_in95"] = float(np.mean(log["nees"] < 7.81))
    # docking error expressed in the dock frame
    c, s = np.cos(dock[2]), np.sin(dock[2])
    d = x_final[:2] - dock[:2]
    out["dock_lon"] = float(c * d[0] + s * d[1])
    out["dock_lat"] = float(-s * d[0] + c * d[1])
    out["dock_head"] = float(wrap(x_final[2] - dock[2]))
    out["docked"] = bool(done)
    out["success"] = bool(done and abs(out["dock_lat"]) <= DOCK_TOL["lat"]
                          and abs(out["dock_lon"]) <= DOCK_TOL["lon"]
                          and abs(out["dock_head"]) <= DOCK_TOL["head"])
    out["t_dock"] = float(log["t"][-1])
    return out
