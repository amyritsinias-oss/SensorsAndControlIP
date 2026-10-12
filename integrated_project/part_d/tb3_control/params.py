"""
Parameters for the TurtleBot3 path-following / docking controller (Part D).

Everything numeric lives here so the project page can quote one place.
Robot values are the TurtleBot3 Waffle Pi datasheet values (the Waffle Pi
is the TB3 variant that carries both the LiDAR and the camera needed for
Track A). Change ROBOT to the Burger values if the team's MJCF uses Burger.
"""
from dataclasses import dataclass


@dataclass
class RobotParams:
    wheel_radius: float = 0.033      # r  [m]
    track_width: float = 0.287       # L  [m]  distance between wheel contact points
    v_max: float = 0.26              # max linear speed  [m/s]  (datasheet)
    w_max: float = 1.82              # max angular speed [rad/s] (datasheet)

    @property
    def wheel_rate_max(self):
        # Max wheel angular rate [rad/s]. Driving straight at v_max uses all of it,
        # so any turning at full speed has to come out of v (see saturate()).
        return self.v_max / self.wheel_radius


@dataclass
class ControllerParams:
    # ---- timing -------------------------------------------------------
    T: float = 0.05                  # control period [s] (20 Hz, typical cmd_vel rate)

    # ---- speed profile -------------------------------------------------
    v_nom: float = 0.20              # cruise speed [m/s]   (~77% of v_max, leaves wheel headroom to steer)
    a_lat_max: float = 0.25          # max lateral accel on curves [m/s^2] -> v <= sqrt(a_lat/|kappa|)
    a_long_max: float = 0.30         # accel/decel limit [m/s^2]
    v_creep: float = 0.03            # min speed while docking, so the robot never stalls short of the dock
    dock_tol: float = 0.01           # stop when the remaining arc length < this [m]

    # ---- Bryson's rule maxima (LQI weights) ----------------------------
    # state z = [e_y, e_th, z_int]; input u = omega_fb = omega - v*kappa
    ey_max: float = 0.05             # largest acceptable cross-track error [m]
    eth_max: float = 0.15            # largest acceptable heading error [rad] (~8.6 deg)
    zint_max: float = 0.05           # largest acceptable integral of e_y [m s]
    u_max: float = 0.60              # largest acceptable feedback turn rate [rad/s]
    rho: float = 5.0                 # extra scale on R (rho>1 = gentler). Chosen from the sweep (E2): minimum of
                                     # the truth-fed tracking curve, ~40% less steering effort than rho=1 at the
                                     # same EKF-in-the-loop tracking (which is set by estimation error).

    # ---- gain schedule --------------------------------------------------
    v_sched_min: float = 0.03        # speeds below this use the gain designed at this speed
    n_sched: int = 15                # number of speeds the gain is designed at

    # ---- integrator ---------------------------------------------------
    zint_limit: float = 0.10         # hard clamp on the integrator state [m s] (anti-windup backstop)

    # ---- uncertainty-aware speed (uses the EKF covariance, never truth) --
    sigma_slow: float = 0.05         # if 1-sigma position uncertainty exceeds this [m], slow down
