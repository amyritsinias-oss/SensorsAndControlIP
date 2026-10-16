import cv2
import mujoco
import numpy as np

MARKER_DICT = cv2.aruco.DICT_4X4_50
MARKER_SIZE = 0.15          # m, black square side - must match models/build_scene.py
MARKER_SIZES = {0: 0.24}    # per-id overrides: the dock (id 0) carries a bigger tag so it is readable from across the room


class ArucoCameraView:
    def __init__(self, model, camera="camera", width=640, height=480, show=False,
                 window="Robot camera (ArUco)"):
        # show=True also opens a separate OpenCV window; overlay(viewer) puts the feed in the MuJoCo viewer
        self.model, self.show, self.window = model, show, window
        self.last_frame = None                                   # annotated RGB frame
        self.cam_id = model.camera(camera).id
        self.w, self.h = width, height
        self.renderer = mujoco.Renderer(model, height, width)

        # Intrinsics: ideal pinhole from the camera's vertical field of view in the MJCF.
        # (Part B: replace with the calibrated K and distortion from a checkerboard.)
        fy = (height / 2) / np.tan(np.deg2rad(model.cam_fovy[self.cam_id]) / 2)
        self.K = np.array([[fy, 0, width / 2], [0, fy, height / 2], [0, 0, 1.0]])
        self.dist = np.zeros(5)

        # Extrinsics: camera pose in the robot base frame, converted from MuJoCo camera axes
        # (x right, y up, looks along -z) to OpenCV camera axes (x right, y down, looks along +z).
        self.t_bc = model.cam_pos[self.cam_id].copy()
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.cam_quat[self.cam_id])
        self.R_bc = R.reshape(3, 3) @ np.diag([1, -1, -1])

        self.last_obs = []                                       # full detections from the last read()
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX   # sharper corners, less range bias
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(MARKER_DICT), params)

    @staticmethod
    def object_points(marker_id):
        s = MARKER_SIZES.get(marker_id, MARKER_SIZE) / 2         # marker corners in its own frame
        return np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], float)

    def render(self, data):
        self.renderer.update_scene(data, camera=self.cam_id)
        return self.renderer.render()                            # RGB, (h, w, 3)

    def read(self, data):
        """Render, detect, and return [(id, range, bearing)] in the robot base frame."""
        rgb = self.render(data)
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        detections, labels, self.last_obs = [], [], []
        if ids is not None:
            for c, mid in zip(corners, ids.ravel()):
                ok, rvec, tvec = cv2.solvePnP(self.object_points(int(mid)), c[0], self.K, self.dist,
                                              flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if not ok:
                    continue
                p = self.R_bc @ tvec.ravel() + self.t_bc         # marker centre in the base frame
                rng, brg = float(np.hypot(p[0], p[1])), float(np.arctan2(p[1], p[0]))
                detections.append((int(mid), rng, brg))
                labels.append((c, int(mid), rng, brg, rvec, tvec))
                self.last_obs.append(dict(id=int(mid), range=rng, bearing=brg, p_base=p,
                                          pixels=float(np.ptp(c[0][:, 0]))))
        self._draw(rgb, labels)
        return detections

    def _draw(self, rgb, labels):
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for c, mid, rng, brg, rvec, tvec in labels:
            pts = c[0].astype(int)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            cv2.drawFrameAxes(img, self.K, self.dist, rvec, tvec, MARKER_SIZES.get(mid, MARKER_SIZE) * 0.5, 2)
            tag = "DOCK" if mid == 0 else f"id {mid}"
            x, y = pts[:, 0].min(), max(pts[:, 1].min() - 10, 20)
            cv2.putText(img, f"{tag}  {rng:.2f} m  {np.rad2deg(brg):+.1f} deg", (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        status = f"{len(labels)} marker(s) detected" if labels else "no markers in view"
        cv2.putText(img, status, (10, self.h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if labels else (0, 0, 255), 2, cv2.LINE_AA)
        self.last_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.show:
            cv2.imshow(self.window, img)
            cv2.waitKey(1)

    def overlay(self, viewer, width=400, margin=10):
        """Show the latest annotated frame inside the MuJoCo viewer, top-right of the 3D view."""
        vp = viewer.viewport
        if self.last_frame is None or vp is None or vp.width < width + 2 * margin:
            return
        height = int(width * self.h / self.w)
        small = cv2.resize(self.last_frame, (width, height), interpolation=cv2.INTER_AREA)
        # the 3D view can start part-way across the window (UI panels), so offset by vp.left / vp.bottom
        rect = mujoco.MjrRect(vp.left + vp.width - width - margin, vp.bottom + vp.height - height - margin,
                              width, height)
        if not getattr(self, "_announced", False):
            print(f"[camera] feed overlay at x={rect.left}, y={rect.bottom} ({width}x{height}) "
                  f"in a {vp.width}x{vp.height} view")
            self._announced = True
        viewer.set_images((rect, small))

    def close(self):
        self.renderer.close()
        if self.show:
            cv2.destroyWindow(self.window)