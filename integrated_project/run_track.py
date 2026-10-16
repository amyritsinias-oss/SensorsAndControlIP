import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent                     # integrated_project/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "planning"))
sys.path.insert(0, str(ROOT / "part_d"))

from lidar import Lidar                                     # noqa: E402
from navigator import Navigator                             # noqa: E402
from occupancy_grid import OccupancyGrid                    # noqa: E402
from astar import plan                                      # noqa: E402
from tb3_control import RobotParams, ControllerParams, PathFollower, Path as RefPath, dock_poses, wrap  # noqa: E402
from sim.scenario import MARKERS, DOCK_ID, DOCK_STANDOFF, APPROACH_LEN, ROOM  # noqa: E402
from sim.ekf_standin import OdomArucoEKF                    # noqa: E402   <- swap for the Part C EKF
from camera_view import ArucoCameraView, MARKER_SIZES       # noqa: E402
from dock_finder import DockFinder                          # noqa: E402

SCENE = ROOT / "models" / "scene_room.xml"
WHEEL_R, TRACK, TICKS_PER_REV = 0.033, 0.288, 4096         # must match models/tb3/turtlebot3_waffle_pi.xml
LIDAR_EVERY, CAMERA_EVERY = 4, 2                            # 20 Hz control -> LiDAR 5 Hz, camera 10 Hz
START = np.array([0.6, 2.0, 0.0])                           # west side, facing the east wall (tape mark on the floor)
DOCK_Y_RANGE = (0.6, 2.4)                                   # where on the east wall the dock may be placed
DOCK_MARKER_Z = 0.17                                        # dock tag centre height (bigger 0.24 m tag)
LANDMARKS = {k: v for k, v in MARKERS.items() if k != DOCK_ID}   # known markers for localisation (not the dock)


def ekf_sigma(r):
    return 0.01 + 0.02 * r, np.deg2rad(1.0)                 # camera noise the EKF assumes


# ---------------------------------------------------------------- scenario (world setup, not robot knowledge)
def random_layout(rng, n, tries=300):
    """Random dock position on the east wall + n obstacles that (a) leave the start and dock area clear,
    (b) don't block the camera's first view of the dock, (c) still leave a route. Checked with the true
    layout - this is scenario SETUP, the robot never sees it."""
    dock_y = rng.uniform(*DOCK_Y_RANGE)
    marker = np.array([ROOM[0], dock_y])
    pre, _, _ = dock_poses(marker, np.pi, DOCK_STANDOFF, APPROACH_LEN)
    cam0 = START[:2] + np.array([0.086, 0.0])
    sight = marker - cam0
    for _ in range(tries):
        obs = []
        while len(obs) < n:
            r = rng.uniform(0.08, 0.15)
            x, y = rng.uniform(1.2, ROOM[0] - 1.0), rng.uniform(0.4, ROOM[1] - 0.4)
            if np.hypot(x - START[0], y - START[1]) < 0.8:
                continue
            if x > pre[0] - 0.5 and abs(y - dock_y) < 0.6:                       # charging area kept clear
                continue
            t = np.clip(np.dot([x - cam0[0], y - cam0[1]], sight) / np.dot(sight, sight), 0, 1)
            if np.hypot(*(cam0 + t * sight - [x, y])) < r + 0.3:                 # keep the first view clear
                continue
            if any(np.hypot(x - ox, y - oy) < r + orr + 0.45 for ox, oy, orr, _ in obs):
                continue
            obs.append((x, y, r, rng.choice(["box", "cylinder"])))
        if route_exists(obs, pre):
            return dock_y, obs
    raise RuntimeError("could not find a layout with a route to the dock")


def route_exists(obs, pre):
    g = OccupancyGrid()
    yy, xx = np.mgrid[0:g.ny, 0:g.nx]
    cx, cy = g.to_world(yy, xx)
    occ = (cx < 0) | (cx > ROOM[0]) | (cy < 0) | (cy > ROOM[1])
    for x, y, r, _ in obs:
        occ |= np.hypot(cx - x, cy - y) < r * 1.42
    g.log_odds[occ] = g.l_max
    return plan(g, START[:2], pre, inflate=0.30)[0] is not None


def build_model(dock_y, obs):
    spec = mujoco.MjSpec.from_file(str(SCENE))
    for name in ["obs_1", "obs_2", "obs_crossing"]:                              # open room: shelves out
        try:
            spec.body(name).pos = [20.0, 20.0, -5.0]
        except Exception:
            pass
    half = MARKER_SIZES[DOCK_ID] * 1.5 / 2                                      # tag + white border
    spec.body(f"marker_{DOCK_ID}").pos = [ROOM[0] - 0.002, dock_y, DOCK_MARKER_Z]
    spec.geom(f"aruco_{DOCK_ID}").size = [half, half, 0.01]
    spec.geom("dock_pad").pos = [ROOM[0] - 0.3, dock_y, 0.001]
    for i, (x, y, r, kind) in enumerate(obs):
        body = spec.worldbody.add_body(name=f"rand_{i}", pos=[x, y, 0.2])
        if kind == "box":
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[r, r, 0.2], rgba=[0.95, 0.6, 0.1, 1])
        else:
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, 0.2, 0], rgba=[0.95, 0.6, 0.1, 1])
    return spec.compile()


def draw_path(scn, path, step=10):
    scn.ngeom = 0
    for i in range(0, len(path.x) - step, step):
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.array([0.1, 0.8, 0.2, 1.0], dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.01,
                             np.array([path.x[i], path.y[i], 0.01]),
                             np.array([path.x[i + step], path.y[i + step], 0.01]))
        scn.ngeom += 1


# ---------------------------------------------------------------- one mission
def run(seed=0, n_obstacles=5, use_viewer=True, t_max=150.0, verbose=True):
    rng = np.random.default_rng(seed)
    dock_y, obs = random_layout(rng, n_obstacles)
    model = build_model(dock_y, obs)
    data = mujoco.MjData(model)
    true_dock = dock_poses(np.array([ROOM[0], dock_y]), np.pi, DOCK_STANDOFF, APPROACH_LEN)  # evaluation only

    rp = RobotParams(wheel_radius=WHEEL_R, track_width=TRACK)
    cp = ControllerParams()
    n_sub = max(1, round(cp.T / model.opt.timestep))
    act_l, act_r = model.actuator("wheel_left").id, model.actuator("wheel_right").id
    enc_l, enc_r = model.sensor("enc_left").adr[0], model.sensor("enc_right").adr[0]
    robot_bodies = {model.body(n).id for n in ["base", "wheel_left", "wheel_right"]}
    floor_id = model.geom("floor").id

    x0 = START + np.array([rng.normal(0, 0.03), rng.normal(0, 0.03), rng.normal(0, np.deg2rad(3))])
    data.qpos[0:3] = [x0[0], x0[1], 0.0]                    # placed by hand on the tape mark (small error)
    data.qpos[3:7] = [np.cos(x0[2] / 2), 0, 0, np.sin(x0[2] / 2)]
    mujoco.mj_forward(model, data)

    def read_ticks():
        k = TICKS_PER_REV / (2 * np.pi)
        return int(np.floor(data.sensordata[enc_l] * k)), int(np.floor(data.sensordata[enc_r] * k))

    def true_pose():                                        # evaluation only
        w, x, y, z = data.qpos[3:7]
        return np.array([data.qpos[0], data.qpos[1], np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))])

    lidar = Lidar(model, site="lidar", robot_body="base", noise_std=0.01, seed=seed)
    camera = ArucoCameraView(model)
    ekf = OdomArucoEKF(START, np.diag([0.05 ** 2, 0.05 ** 2, np.deg2rad(5) ** 2]), rp, LANDMARKS, ekf_sigma)
    ekf.predict(read_ticks())
    grid = OccupancyGrid()
    nav = Navigator(lidar, grid, RefPath, None, None, None)
    dock = DockFinder(DOCK_STANDOFF, APPROACH_LEN)

    viewer = None
    if use_viewer:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)

    ctrl, stop, info, scan_world = None, False, {"done": False}, None
    log = {"truth": [], "est": [], "plans": [], "dock_est": []}
    collisions, stops, k, t = 0, 0, 0, 0.0
    stopped_for, backing_up, retarget = 0, 0, False
    t_found = None
    while t < t_max:
        t_wall = time.time()
        # ---------------- sensing + estimation (no ground truth) ----------------
        if k % CAMERA_EVERY == 0:
            dets = camera.read(data)
            ekf.update([d for d in dets if d[0] in LANDMARKS])          # known markers -> localisation
            dock_obs = [o for o in camera.last_obs if o["id"] == DOCK_ID]
            if dock_obs and dock.update(dock_obs[0], ekf.x, scan_world):  # dock marker -> target
                retarget = True
                if t_found is None and dock.found:
                    t_found = t
                    if verbose:
                        print(f"t = {t:5.1f} s  dock spotted: marker at ({dock.marker_xy[0]:.2f}, "
                              f"{dock.marker_xy[1]:.2f}), facing {np.rad2deg(dock.facing):.0f} deg")
            if viewer is not None:
                camera.overlay(viewer)
        if k % LIDAR_EVERY == 0:
            ranges = lidar.scan(data)
            scan_world = lidar.to_world(ranges, ekf.x)
            if not dock.found:
                grid.update(lidar, ranges, ekf.x)                       # keep mapping while searching
                stop = bool(np.any(ranges[nav.front] < nav.stop_dist))
            else:
                if retarget and nav.scans_since_replan >= nav.cooldown_scans:
                    nav.pre, nav.dock, nav.dock_heading = dock.targets()
                    nav.path = None                                     # force a plan to the new target
                    retarget = False
                new_path, stop = nav.step(ranges, ekf.x, ctrl.i if ctrl else 0)
                if new_path is not None:
                    ctrl = PathFollower(new_path, rp, cp)
                    log["plans"].append((t, new_path))
                    if verbose:
                        print(f"t = {t:5.1f} s  planned ({len(nav.waypoints)} waypoints, {new_path.length:.2f} m)"
                              + ("  [dock target locked]" if dock.frozen else ""))
                    if viewer is not None:
                        with viewer.lock():
                            draw_path(viewer.user_scn, new_path)
        # ---------------- decide the command (estimate only) ----------------
        if not dock.found:
            v, w = 0.0, 0.4                                             # search: turn on the spot until seen
        elif ctrl is None:
            v, w = 0.0, 0.0
        else:
            v, w, info = ctrl.step(ekf.x, ekf.P)
        if backing_up:
            v, w = -0.08, 0.0
            backing_up -= 1
            if backing_up == 0:
                nav.path = None
                nav.scans_since_replan = nav.cooldown_scans
        elif stop and dock.found:
            v, w = 0.0, 0.0
            stops += 1
            stopped_for += 1
            if stopped_for >= 40:                                       # blocked 2 s: back off and replan
                backing_up, stopped_for = 20, 0
        else:
            stopped_for = 0
        # ---------------- evaluation only ----------------
        log["truth"].append(true_pose())
        log["est"].append(ekf.x.copy())
        if info.get("done"):
            break
        # ---------------- actuate ----------------
        data.ctrl[act_l] = (v - w * TRACK / 2) / WHEEL_R
        data.ctrl[act_r] = (v + w * TRACK / 2) / WHEEL_R
        hit = False
        for _ in range(n_sub):
            mujoco.mj_step(model, data)
            for c in data.contact[:data.ncon]:                          # evaluation only
                b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
                if (b1 in robot_bodies) != (b2 in robot_bodies) and floor_id not in (c.geom1, c.geom2):
                    hit = True
        collisions += hit
        ekf.predict(read_ticks())
        t += cp.T
        k += 1
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(max(0.0, cp.T - (time.time() - t_wall)))

    # ---------------- evaluation only: score against the TRUE dock ----------------
    xt = true_pose()
    _, dock_true_xy, dock_true_head = true_dock
    c, s = np.cos(dock_true_head), np.sin(dock_true_head)
    d = xt[:2] - dock_true_xy
    res = dict(seed=seed, dock_y=dock_y, docked=bool(info.get("done")), t=t, plans=nav.n_replans,
               t_found=t_found, collision_steps=collisions,
               dock_lat=-s * d[0] + c * d[1], dock_lon=c * d[0] + s * d[1], dock_head=wrap(xt[2] - dock_true_head),
               est_marker=None if not dock.found else dock.marker_xy.copy(),
               est_facing=None if not dock.found else dock.facing,
               obstacles=obs, log=log, grid=grid)
    res["success"] = (res["docked"] and collisions == 0 and abs(res["dock_lat"]) < 0.03
                      and abs(res["dock_lon"]) < 0.03 and abs(res["dock_head"]) < np.deg2rad(5))
    if viewer is not None:
        viewer.close()
    camera.close()
    return res


def plot(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = res["grid"]
    extent = [g.x_min, g.x_min + g.nx * g.res, g.y_min, g.y_min + g.ny * g.res]
    plt.figure(figsize=(9, 7))
    plt.imshow(g.probability(), origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=1)
    plans = res["log"]["plans"]
    for j, (tp, p) in enumerate(plans):
        plt.plot(p.x, p.y, color=plt.cm.autumn(j / max(len(plans) - 1, 1)), lw=1, alpha=0.6)
    tr, es = np.array(res["log"]["truth"]), np.array(res["log"]["est"])
    plt.plot(tr[:, 0], tr[:, 1], "C0", lw=2, label="robot (ground truth, evaluation only)")
    plt.plot(es[:, 0], es[:, 1], "C1--", lw=1, label="EKF estimate")
    for x, y, r, kind in res["obstacles"]:
        patch = plt.Rectangle((x - r, y - r), 2 * r, 2 * r) if kind == "box" else plt.Circle((x, y), r)
        patch.set(fill=False, ec="C1", lw=2)
        plt.gca().add_patch(patch)
    plt.plot(ROOM[0], res["dock_y"], "ks", ms=10, label="dock marker (true, evaluation only)")
    if res["est_marker"] is not None:
        plt.plot(*res["est_marker"], "g+", ms=16, mew=2, label="dock marker (robot's estimate)")
    plt.legend(loc="upper left", fontsize=8); plt.gca().set_aspect("equal")
    plt.title(f"seed {res['seed']}: docked={res['docked']}  error lat {res['dock_lat'] * 100:+.1f} cm, "
              f"lon {res['dock_lon'] * 100:+.1f} cm, head {np.rad2deg(res['dock_head']):+.1f} deg")
    plt.savefig(out, dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None, help="layout seed (default: a new random one each run)")
    ap.add_argument("--n-obstacles", type=int, default=5)
    ap.add_argument("--trials", type=int, default=0, help="run N random layouts headless and report")
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args()
    if args.seed is None:
        args.seed = int(np.random.default_rng().integers(0, 10000))
        if not args.trials:
            print(f"random layout: seed {args.seed}  (rerun with --seed {args.seed} to repeat it)")
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    if args.trials:
        rows = []
        for s in range(args.seed, args.seed + args.trials):
            r = run(seed=s, n_obstacles=args.n_obstacles, use_viewer=False, verbose=False)
            rows.append(r)
            print(f"seed {s:5d}: success={r['success']!s:5}  dock y={r['dock_y']:.2f}  t={r['t']:5.1f} s  "
                  f"collisions={r['collision_steps']}  err lat {r['dock_lat'] * 100:+.1f} cm "
                  f"lon {r['dock_lon'] * 100:+.1f} cm head {np.rad2deg(r['dock_head']):+.1f} deg")
            if not r["success"]:
                plot(r, out_dir / f"track_fail_seed{s}.png")
        ok = np.mean([r["success"] for r in rows]) * 100
        print(f"\nSUCCESS {ok:.0f}% over {args.trials} random layouts "
              f"(random dock position, {args.n_obstacles} obstacles each)")
        return

    r = run(seed=args.seed, n_obstacles=args.n_obstacles, use_viewer=not args.no_viewer)
    print(f"docked: {r['docked']} after {r['t']:.1f} s, {r['plans']} plans, collisions: {r['collision_steps']}")
    print(f"dock error vs TRUE dock: lateral {r['dock_lat'] * 100:+.2f} cm, longitudinal {r['dock_lon'] * 100:+.2f} cm, "
          f"heading {np.rad2deg(r['dock_head']):+.2f} deg   success: {r['success']}")
    plot(r, out_dir / f"track_seed{args.seed}.png")
    print("saved", out_dir / f"track_seed{args.seed}.png")


if __name__ == "__main__":
    main()