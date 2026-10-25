"""Live ArUco test: detect markers, show ID, distance and bearing.

Uses the same settings as the project (DICT_4X4_50, dock id 0 = 0.24 m,
others 0.15 m black square), so if a marker reads correctly here, the
project's camera code will see it too.

Usage:
    python aruco_test.py 192.168.1.213          # robot camera over rosbridge
    python aruco_test.py --webcam               # Mac webcam, no robot needed
    python aruco_test.py 192.168.1.213 --calib calib.npz   # with real calibration

Keys (click the image window first):  Q / Esc = quit,  S = save a snapshot.

Check: hold a tape measure from the camera to a marker. Distance should match
within a few cm. Without calibration (--calib) expect ~5-15 % error; that's
the guessed intrinsics, not the marker.
"""

import argparse
import time

import cv2
import numpy as np

MARKER_DICT = cv2.aruco.DICT_4X4_50
MARKER_SIZE = 0.10            # m, black square side - must match the printouts
MARKER_SIZES = {0: 0.24}      # per-id overrides (dock)
GUESS_HFOV_DEG = 29.0       # Raspberry Pi Camera v2 horizontal FOV (used only without --calib)


def guess_intrinsics(w, h, hfov_deg=GUESS_HFOV_DEG):
    f = (w / 2) / np.tan(np.deg2rad(hfov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]]), np.zeros(5)


def object_points(size):
    s = size / 2
    return np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("host", nargs="?", help="robot IP, e.g. 192.168.1.213")
    p.add_argument("--webcam", action="store_true", help="use this computer's webcam instead of the robot")
    p.add_argument("--calib", help=".npz file with K (3x3) and dist arrays from your calibration")
    p.add_argument("--size", type=float, help="override marker size for all ids [m]")
    args = p.parse_args()
    if not args.webcam and not args.host:
        p.error("give the robot IP, or use --webcam")

    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(MARKER_DICT), params)

    K = dist = None
    if args.calib:
        c = np.load(args.calib)
        K, dist = c["K"], c["dist"]
        print("using calibration from", args.calib)

    robot = cap = None
    if args.webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise SystemExit("could not open webcam (check System Settings > Privacy > Camera)")
        get_frame = lambda: cap.read()[1]
    else:
        from rosbridge_api import RosbridgeRobot
        robot = RosbridgeRobot(args.host)
        robot.setup()
        get_frame = lambda: robot.query_image(timeout=1.0, wait_for_new=True)

    print("Running. Q/Esc to quit, S to save a snapshot.")
    last_print = 0.0
    try:
        while True:
            img = get_frame()
            if img is None:
                continue
            h, w = img.shape[:2]
            if K is None:
                K, dist = guess_intrinsics(w, h)
                print(f"image {w}x{h}, no calibration: using a guessed K (f = {K[0, 0]:.0f} px)")

            corners, ids, _ = detector.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            lines = []
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(img, corners, ids)
                for c, mid in zip(corners, ids.ravel()):
                    mid = int(mid)
                    size = args.size or MARKER_SIZES.get(mid, MARKER_SIZE)
                    ok, rvec, tvec = cv2.solvePnP(object_points(size), c[0], K, dist,
                                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    if not ok:
                        continue
                    x, _, z = tvec.ravel()                 # camera frame: x right, z forward
                    rng = float(np.hypot(x, z))
                    brg = float(np.degrees(np.arctan2(-x, z)))   # + = marker is to the left
                    cv2.drawFrameAxes(img, K, dist, rvec, tvec, size / 2)
                    px = int(c[0][:, 0].min()), int(c[0][:, 1].min()) - 10
                    cv2.putText(img, f"id {mid}: {rng:.2f} m  {brg:+.0f} deg", px,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    lines.append(f"id {mid}: range {rng:.3f} m, bearing {brg:+.1f} deg, "
                                 f"{np.ptp(c[0][:, 0]):.0f} px wide")
            else:
                cv2.putText(img, "no markers", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if time.time() - last_print > 1.0:           # print once a second
                print(" | ".join(lines) if lines else "no markers")
                last_print = time.time()

            cv2.imshow("ArUco test", img)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                name = f"aruco_snapshot_{int(time.time())}.png"
                cv2.imwrite(name, img)
                print("saved", name)
    finally:
        cv2.destroyAllWindows()
        if cap is not None:
            cap.release()
        if robot is not None:
            robot.cleanup()


if __name__ == "__main__":
    main()