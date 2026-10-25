"""Real-robot version of run_track.py: start zone -> dock zone over rosbridge.

Same pipeline as the simulation, unchanged:
    DockFinder (dock ArUco + LiDAR wall fit) -> OccupancyGrid + A* Navigator -> LQI PathFollower
Only the sensors and motors are swapped:
    MuJoCo LiDAR rays      -> robot /scan          (RealLidar)
    MuJoCo camera render   -> robot camera image    (RealCamera)
    MuJoCo encoder sensors -> robot /joint_states   (read_ticks)
    data.ctrl + mj_step    -> robot.send_command(v, w)

World frame = where the robot starts: (0, 0), facing +x. The dock marker must be visible
from the start. There are no landmark markers, so the EKF runs on wheel odometry and the
dock target is refined every time the dock marker is seen again.

Put this file in integrated_project/ (next to run_track.py) and run:
    python integrated_project/run_real.py 192.168.1.213
    python integrated_project/run_real.py 192.168.1.213 --v-nom 0.10 --show
Ctrl+C stops the robot.

>>> Fill in the MEASURE values below before the first run. <<<
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent                     # integrated_project/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "planning"))
sys.path.insert(0, str(ROOT / "part_d"))

from navigator import Navigator                             # noqa: E402
from occupancy_grid import OccupancyGrid                    # noqa: E402
from tb3_control import RobotParams, ControllerParams, PathFollower, Path as RefPath  # noqa: E402
from sim.ekf_standin import OdomArucoEKF                    # noqa: E402   <- swap for the Part C EKF
from dock_finder import DockFinder                          # noqa: E402
from rosbridge_api import RosbridgeRobot                    # noqa: E402

# ============================ MEASURE / SET THESE ============================
DOCK_ID = 0
MARKER_SIZE = 0.15           # m, BLACK square side of the dock marker (measure to the mm)
# Optional high "beacon" marker, mounted DIRECTLY ABOVE the dock marker (same wall, same x/y),
# so the robot can see it over the obstacles from far away. Set BEACON_ID = None to disable.
BEACON_ID = 3
BEACON_SIZE = 0.10           # m, BLACK square side of the beacon marker
MARKER_SIZES = {DOCK_ID: MARKER_SIZE, BEACON_ID: BEACON_SIZE}
CALIB_FILE = None             # e.g. "calib.npz" with arrays K (3x3) and dist (from Part B). None = guessed K
GUESS_HFOV_DEG = 29.0         # Pi Camera v2 horizontal FOV, only used when CALIB_FILE is None

# Camera position in the robot base frame (origin = middle of the wheel axle on the floor,
# x forward, y left, z up) [m]
CAM_POS = np.array([0.073, 0.0, 0.10])     # MEASURE: forward, left, height of the lens
LIDAR_OFFSET = np.array([-0.064, 0.0])     # MEASURE: LiDAR centre forward/left of the axle middle

# Robot: Waffle Pi defaults. Burger: track 0.160, v_max 0.22, w_max 2.84
WHEEL_R, TRACK = 0.033, 0.287
V_MAX, W_MAX = 0.26, 0.8          # W_MAX lowered from 1.82: fast turns blur the camera
LEFT_JOINT, RIGHT_JOINT = "wheel_left_joint", "wheel_right_joint"

DOCK_STANDOFF = 0.35          # MEASURE/SET: marker -> robot centre when docked [m]
APPROACH_LEN = 0.50           # straight final approach [m]; start must be > STANDOFF + APPROACH_LEN + ~0.3 from the dock

# Occupancy grid around the start (robot starts at (0,0) facing +x, the dock is ahead)
GRID = dict(x_min=-0.5, y_min=-3.0, width=8.0, height=6.0, res=0.05)   # covers 7.5 m ahead, 3 m each side
# ============================================================================

TICKS_PER_REV = 4096          # what the stand-in EKF expects (k = 2*pi/4096)
CAMERA_EVERY = 2              # run ArUco every 2nd control step (10 Hz at T = 0.05)
SCAN_STALE_S = 1.0            # stop if no new LiDAR scan for this long


def ekf_sigma(r):
    return 0.01 + 0.02 * r, np.deg2rad(1.0)


class RealLidar:
    """Same attributes the Navigator / OccupancyGrid use from the sim Lidar."""

    def __init__(self, n_beams=360, range_min=0.12, range_max=3.5, offset=LIDAR_OFFSET):
        self.n = n_beams
        self.angles = np.linspace(0, 2 * np.pi, n_beams, endpoint=False)
        self.range_min, self.range_max = range_min, range_max
        self.offset = np.asarray(offset, float)

    def from_scan(self, scan):
        """Robot LaserScan -> n_beams ranges at self.angles; no return / invalid -> inf (like the sim)."""
        r = np.asarray(scan.ranges, float)
        a = np.mod(np.asarray(scan.angles, float), 2 * np.pi)
        good = np.isfinite(r) & (r >= self.range_min) & (r <= self.range_max)
        out = np.full(self.n, np.inf)
        idx = np.round(a[good] / (2 * np.pi / self.n)).astype(int) % self.n
        np.minimum.at(out, idx, r[good])       # nearest beam; keep the closest if two land together
        return out

    def to_world(self, ranges, pose):
        ok = np.isfinite(ranges)
        r, a = ranges[ok], self.angles[ok]
        pts = self.offset + np.column_stack([r * np.cos(a), r * np.sin(a)])
        x, y, th = pose
        c, s = np.cos(th), np.sin(th)
        return np.column_stack([x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]])


class RealCamera:
    """ArUco detection on the robot image; same outputs as camera_view.ArucoCameraView.read()."""

    def __init__(self):
        self.K = self.dist = None
        if CALIB_FILE:
            c = np.load(CALIB_FILE)
            self.K, self.dist = c["K"], c["dist"]
        # OpenCV camera (x right, y down, z forward) -> robot base (x forward, y left, z up)
        self.R_bc = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
        self.t_bc = CAM_POS
        self.obj = {}
        for mid, size in MARKER_SIZES.items():
            if mid is None:
                continue
            s = size / 2
            self.obj[mid] = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)
        self.last_obs, self.last_frame = [], None

    def read(self, img):
        h, w = img.shape[:2]
        if self.K is None:
            f = (w / 2) / np.tan(np.deg2rad(GUESS_HFOV_DEG) / 2)
            self.K, self.dist = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]]), np.zeros(5)
            print(f"camera {w}x{h}, no calibration file: using a guessed K (f = {f:.0f} px)")
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        dets, self.last_obs = [], []
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            for c, mid in zip(corners, ids.ravel()):
                if int(mid) not in self.obj:
                    continue
                ok, rvec, tvec = cv2.solvePnP(self.obj[int(mid)], c[0], self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if not ok:
                    continue
                p = self.R_bc @ tvec.ravel() + self.t_bc
                rng, brg = float(np.hypot(p[0], p[1])), float(np.arctan2(p[1], p[0]))
                dets.append((int(mid), rng, brg))
                self.last_obs.append(dict(id=int(mid), range=rng, bearing=brg, p_base=p,
                                          pixels=float(np.ptp(c[0][:, 0]))))
        self.last_frame = img
        return dets


def read_ticks(robot):
    js = robot.query_joint_states()
    if not js or "name" not in js:
        return None
    pos = dict(zip(js["name"], js["position"]))
    if LEFT_JOINT not in pos or RIGHT_JOINT not in pos:
        raise RuntimeError(f"joint names on this robot: {js['name']} - set LEFT_JOINT / RIGHT_JOINT")
    k = TICKS_PER_REV / (2 * np.pi)
    return int(np.floor(pos[LEFT_JOINT] * k)), int(np.floor(pos[RIGHT_JOINT] * k))


def run(host, v_nom, t_max, show):
    rp = RobotParams(wheel_radius=WHEEL_R, track_width=TRACK, v_max=V_MAX, w_max=W_MAX)
    # sigma_slow off: with no landmark markers the EKF position uncertainty only grows, which would
    # slow the robot to creep speed; the dock target is re-measured by the camera instead
    cp = ControllerParams(v_nom=v_nom, sigma_slow=1e9)
    lidar, camera = RealLidar(), RealCamera()
    grid = OccupancyGrid(**GRID)
    nav = Navigator(lidar, grid, RefPath, None, None, None)
    dock = DockFinder(DOCK_STANDOFF, APPROACH_LEN)
    ekf = OdomArucoEKF(np.zeros(3), np.diag([1e-4, 1e-4, np.deg2rad(1) ** 2]), rp, {}, ekf_sigma)
    log = {"t": [], "est": [], "cmd": [], "plans": []}

    with RosbridgeRobot(host) as robot:
        print("connected - waiting for encoders and LiDAR ...")
        t_wait = time.time()
        while read_ticks(robot) is None or robot.query_laser_scan(timeout=0.5) is None:
            if time.time() - t_wait > 10:
                raise SystemExit("no /joint_states or /scan from the robot after 10 s")
        ekf.predict(read_ticks(robot))
        print("running. Ctrl+C to stop.")

        ctrl, stop, info, scan_world, ranges = None, False, {"done": False}, None, None
        stopped_for, backing_up, retarget = 0, 0, False
        t0 = last_scan_t = time.time()
        k, slow_steps = 0, 0
        last_seen = None
        try:
            while time.time() - t0 < t_max:
                t_wall = time.time()
                t = t_wall - t0
                # ---------------- sensing + estimation ----------------
                if k % CAMERA_EVERY == 0:
                    img = robot.query_image(timeout=0, wait_for_new=True)
                    if img is not None:
                        camera.read(img)
                        # prefer the dock marker; use the beacon (directly above it) when the dock one is hidden
                        dock_obs = ([o for o in camera.last_obs if o["id"] == DOCK_ID]
                                    or [o for o in camera.last_obs if o["id"] == BEACON_ID])
                        if dock_obs:
                            seen = "dock" if dock_obs[0]["id"] == DOCK_ID else "beacon"
                            if seen != last_seen:
                                print(f"t = {t:5.1f} s  following the {seen} marker")
                                last_seen = seen
                        if dock_obs and dock.update(dock_obs[0], ekf.x, scan_world):
                            retarget = True
                            if dock.n_sightings == 1:
                                print(f"t = {t:5.1f} s  dock spotted at ({dock.marker_xy[0]:.2f}, "
                                      f"{dock.marker_xy[1]:.2f}), {dock_obs[0]['range']:.2f} m away")
                        if show:
                            cv2.imshow("robot camera", camera.last_frame)
                            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                                break
                scan = robot.query_laser_scan(timeout=0, wait_for_new=True)
                if scan is not None:
                    last_scan_t = time.time()
                    ranges = lidar.from_scan(scan)
                    scan_world = lidar.to_world(ranges, ekf.x)
                    if not dock.found:
                        grid.update(lidar, ranges, ekf.x)
                        stop = bool(np.any(ranges[nav.front] < nav.stop_dist))
                    else:
                        if retarget and nav.scans_since_replan >= nav.cooldown_scans:
                            nav.pre, nav.dock, nav.dock_heading = dock.targets()
                            rr, cc = grid.to_cell(*nav.pre)
                            if not grid.in_bounds(rr, cc):
                                raise SystemExit(f"dock target ({nav.pre[0]:.2f}, {nav.pre[1]:.2f}) is outside "
                                                 "the map - make GRID bigger at the top of run_real.py")
                            nav.path = None
                            retarget = False
                        new_path, stop = nav.step(ranges, ekf.x, ctrl.i if ctrl else 0)
                        if new_path is not None:
                            ctrl = PathFollower(new_path, rp, cp)
                            log["plans"].append((t, new_path))
                            print(f"t = {t:5.1f} s  planned ({len(nav.waypoints)} waypoints, "
                                  f"{new_path.length:.2f} m)" + ("  [dock locked]" if dock.frozen else ""))
                # ---------------- decide the command ----------------
                if not dock.found:
                    v, w = 0.0, 0.0                                     # wait still until the dock marker is seen
                    if k % 40 == 0:
                        print(f"t = {t:5.1f} s  waiting for the dock marker - point the camera at it")
                elif ctrl is None:
                    v, w = 0.0, 0.0
                else:
                    v, w, info = ctrl.step(ekf.x, ekf.P)
                if backing_up:
                    v, w = -0.06, 0.0
                    backing_up -= 1
                    if backing_up == 0:
                        nav.path = None
                        nav.scans_since_replan = nav.cooldown_scans
                elif stop and dock.found:
                    v, w = 0.0, 0.0
                    stopped_for += 1
                    if stopped_for >= 40:                               # blocked ~2 s: back off and replan
                        backing_up, stopped_for = 20, 0
                else:
                    stopped_for = 0
                if time.time() - last_scan_t > SCAN_STALE_S:            # fail safe: no LiDAR -> no motion
                    v, w = 0.0, 0.0
                if info.get("done"):
                    robot.stop()
                    print(f"t = {t:5.1f} s  DOCKED at estimate ({ekf.x[0]:.2f}, {ekf.x[1]:.2f}, "
                          f"{np.rad2deg(ekf.x[2]):.0f} deg)")
                    break
                # ---------------- actuate + odometry ----------------
                robot.send_command(float(v), float(w))
                log["t"].append(t)
                log["est"].append(ekf.x.copy())
                log["cmd"].append((v, w))
                time.sleep(max(0.0, cp.T - (time.time() - t_wall)))
                ticks = read_ticks(robot)
                if ticks is not None:
                    ekf.predict(ticks)
                if time.time() - t_wall > 1.5 * cp.T:
                    slow_steps += 1
                k += 1
        except KeyboardInterrupt:
            print("\nCtrl+C - stopping")
        finally:
            robot.stop()
            if show:
                cv2.destroyAllWindows()
    if k and slow_steps > 0.2 * k:
        print(f"warning: {100 * slow_steps / k:.0f}% of steps overran T = {cp.T} s - "
              "consider a larger T in part_d/tb3_control/params.py")
    return log, grid, dock


def plot(log, grid, dock, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    occ = 1 / (1 + np.exp(-grid.log_odds))
    ax.imshow(occ, origin="lower", cmap="Greys", vmin=0, vmax=1,
              extent=[grid.x_min, grid.x_min + grid.nx * grid.res, grid.y_min, grid.y_min + grid.ny * grid.res])
    for _, p in log["plans"]:
        ax.plot(p.x, p.y, "c-", lw=0.8, alpha=0.6)
    if log["est"]:
        e = np.array(log["est"])
        ax.plot(e[:, 0], e[:, 1], "b-", lw=2, label="EKF estimate")
    if dock.found:
        ax.plot(*dock.marker_xy, "rs", ms=8, label="dock marker (est.)")
    ax.plot(0, 0, "go", label="start")
    ax.set_aspect("equal")
    ax.legend(loc="best")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("saved", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("host", help="robot IP, e.g. 192.168.1.213")
    ap.add_argument("--v-nom", type=float, default=0.10, help="cruise speed [m/s] (sim used 0.20)")
    ap.add_argument("--t-max", type=float, default=120.0)
    ap.add_argument("--show", action="store_true", help="show the camera with detections")
    args = ap.parse_args()
    log, grid, dock = run(args.host, args.v_nom, args.t_max, args.show)
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    plot(log, grid, dock, out_dir / f"real_run_{time.strftime('%Y%m%d_%H%M%S')}.png")


if __name__ == "__main__":
    main()