"""
Simulated ArUco detections: range and bearing to wall markers, with the
geometric limits of the Waffle Pi's Raspberry Pi camera.

A marker is detected only if it is inside the horizontal field of view,
within range, and seen from in front (a marker viewed edge-on cannot be
decoded). Noise grows with range, as the marker gets fewer pixels.
"""
import numpy as np

from tb3_control.model import wrap


class ArucoCamera:
    def __init__(self, markers, hfov_deg=62.2, r_min=0.15, r_max=3.0,
                 max_view_deg=60.0, sig_r0=0.01, sig_r_k=0.02, sig_b=np.deg2rad(1.0),
                 rate_hz=10.0, p_drop=0.1, rng=None):
        # markers: dict id -> (x, y, normal_angle)
        self.markers = markers
        self.half_fov = np.deg2rad(hfov_deg) / 2
        self.r_min, self.r_max = r_min, r_max
        self.max_view = np.deg2rad(max_view_deg)
        self.sig_r0, self.sig_r_k, self.sig_b = sig_r0, sig_r_k, sig_b
        self.period = 1.0 / rate_hz
        self.p_drop = p_drop
        self.rng = rng or np.random.default_rng(1)
        self._t_last = -1e9

    def sigma(self, r):
        return self.sig_r0 + self.sig_r_k * r, self.sig_b

    def detect(self, x_true, t):
        """Returns list of (id, range, bearing). Uses truth to GENERATE the
        measurement (that is what a sensor does); nothing downstream sees truth."""
        if t - self._t_last < self.period - 1e-9:
            return []
        self._t_last = t
        out = []
        for mid, (mx, my, mn) in self.markers.items():
            dx, dy = mx - x_true[0], my - x_true[1]
            r = np.hypot(dx, dy)
            b = wrap(np.arctan2(dy, dx) - x_true[2])
            view = abs(wrap(np.arctan2(-dy, -dx) - mn))     # angle off the marker normal
            if not (self.r_min < r < self.r_max and abs(b) < self.half_fov and view < self.max_view):
                continue
            if self.rng.random() < self.p_drop:              # motion blur / missed detection
                continue
            sr, sb = self.sigma(r)
            out.append((mid, r + sr * self.rng.standard_normal(),
                        wrap(b + sb * self.rng.standard_normal())))
        return out
