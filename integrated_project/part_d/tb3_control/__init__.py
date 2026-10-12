"""TurtleBot3 Track A - Part D (control): LQI path following and ArUco docking."""
from .params import RobotParams, ControllerParams
from .path import Path, dock_poses
from .controller import PathFollower, saturate, wheel_rates
from .lqr_design import GainSchedule, dlqr, bryson, design_weights
from .model import error_model_ct, c2d_zoh, augmented_dt, ctrb, controllability_report, wrap

__all__ = ["RobotParams", "ControllerParams", "Path", "dock_poses", "PathFollower",
           "saturate", "wheel_rates", "GainSchedule", "dlqr", "bryson", "design_weights",
           "error_model_ct", "c2d_zoh", "augmented_dt", "ctrb", "controllability_report", "wrap"]
