"""Sanity checks for the Part D controller.
Run:  python -m pytest -q   (or without pytest:  python tests/test_controller.py)"""
import ast
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tb3_control import (RobotParams, ControllerParams, GainSchedule, PathFollower, augmented_dt,
                         c2d_zoh, error_model_ct, controllability_report, saturate, wrap)
from sim.scenario import build_path, run_trial

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_zoh_matches_closed_form():
    v, T = 0.2, 0.05
    Ad, Bd = c2d_zoh(*error_model_ct(v), T)
    assert np.allclose(Ad, [[1, v * T], [0, 1]])
    assert np.allclose(Bd, [[v * T ** 2 / 2], [T]])


def test_integrator_row_matches_code():
    # z(k+1) = z(k) + T e_y(k) is what PathFollower.step does
    A, _ = augmented_dt(0.2, 0.05)
    assert np.allclose(A[2], [0.05, 0.0, 1.0])


def test_controllability_needs_motion():
    cp = ControllerParams()
    assert controllability_report(0.0, cp.T)["rank"] < 3
    assert controllability_report(cp.v_creep, cp.T)["rank"] == 3


def test_closed_loop_stable_across_schedule():
    cp, rp = ControllerParams(), RobotParams()
    g = GainSchedule(cp, rp.v_max)
    for v in np.linspace(cp.v_sched_min, rp.v_max, 25):
        assert np.max(np.abs(g.closed_loop_poles(v))) < 1.0


def test_saturation_keeps_curvature_and_limits():
    rp = RobotParams()
    v, w, sat = saturate(0.25, 3.0, rp)
    assert sat and abs(w) <= rp.w_max + 1e-9
    assert np.isclose(w / v, 3.0 / 0.25)
    half = rp.track_width / 2
    assert max(abs(v - w * half), abs(v + w * half)) / rp.wheel_radius <= rp.wheel_rate_max + 1e-9


def test_controller_on_path_gives_feedforward_only():
    path, _ = build_path()
    ctrl = PathFollower(path)
    i = 400
    ctrl.i = i                       # projection hint (normally carried from the last step)
    v, w, info = ctrl.step(np.array([path.x[i], path.y[i], path.th[i]]))
    assert abs(info["e_y"]) < 1e-3 and abs(wrap(info["e_th"])) < 1e-2


def test_controller_never_reads_ground_truth():
    """Static check: nothing in the controller package touches the plant or truth."""
    banned = {"truth", "x_true", "plant", "wheel_err", "TB3Plant"}
    for f in os.listdir(os.path.join(ROOT, "tb3_control")):
        if f.endswith(".py"):
            tree = ast.parse(open(os.path.join(ROOT, "tb3_control", f)).read())
            names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
            assert not (names & banned), (f, names & banned)


def test_closed_loop_docks():
    L = run_trial(seed=1234)
    assert L["success"], {k: L[k] for k in ("dock_lat", "dock_lon", "dock_head")}


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:      # noqa: BLE001 - report every failure, keep going
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
