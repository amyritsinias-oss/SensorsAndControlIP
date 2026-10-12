"""
Part D experiments: controller design, tuning and tracking results.

    python experiments/run_part_d.py            # full run (~2-3 min)
    python experiments/run_part_d.py --quick    # fewer seeds, for a smoke test

Writes figures to results/ and every number quoted on the project page to
results/results.json and results/results.md.

Ground truth is read ONLY inside sim/scenario.py for evaluation. The one
place the controller is given truth is the clearly labelled 'truth-fed
baseline' in E6, which exists only to measure what estimation costs.
"""
import argparse
import json
import os
import sys
from dataclasses import replace
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tb3_control import (RobotParams, ControllerParams, GainSchedule, controllability_report,
                         augmented_dt)
from sim.scenario import run_trial, build_path, ROOM, OBSTACLES, MARKERS, DOCK_ID, DOCK_TOL

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

# ---- figure style (validated categorical slots 1-3, recessive grid) -------
C_TRUE, C_EST, C_ALT = "#2a78d6", "#eb6834", "#1baf7a"
C_REF, C_INK, C_MUTED, C_GRID = "#8a8985", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 9, "axes.titlesize": 10,
    "axes.edgecolor": C_MUTED, "axes.labelcolor": C_INK, "xtick.color": C_MUTED,
    "ytick.color": C_MUTED, "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "legend.frameon": False,
})

# ---- team-defined Part D specification (agree these with the team) -------
SPEC = {
    "D1_rms_ey": 0.05,      # RMS true cross-track error while path following [m]
    "D2_mean_ey": 0.02,     # |mean| true cross-track error (steady offset) [m]
    "D3_rms_wfb": 0.30,     # RMS feedback turn rate, omega - v*kappa [rad/s]
    "D4_success": 0.95,     # fraction of docks within 2 cm lat / 3 cm long / 5 deg
}


def _run(kw):
    """Pool worker: run one trial and keep only the scalar metrics."""
    L = run_trial(**kw)
    return {k: v for k, v in L.items() if isinstance(v, (float, bool, int, np.floating))}


def _low_speed_frac(kw):
    L = run_trial(**kw)
    return float(np.mean(L["v"] < 0.10))


def batch(pool, seeds, **kw):
    return pool.map(_run, [dict(seed=s, **kw) for s in seeds])


def agg(rows, key):
    a = np.array([r[key] for r in rows], float)
    return float(np.mean(a)), float(np.std(a))


# ===========================================================================
def e0_design(cp, rp):
    """Controllability across speed, gains and poles at cruise."""
    rep = [controllability_report(v, cp.T) for v in (0.0, cp.v_creep, cp.v_nom)]
    g = GainSchedule(cp, rp.v_max)
    K = g.K(cp.v_nom)
    A, B = augmented_dt(cp.v_nom, cp.T)
    p = np.linalg.eigvals(A - B @ K.reshape(1, 3))
    s = np.log(p.astype(complex)) / cp.T                  # z = e^{sT} -> s
    wn = np.abs(s)
    zeta = -np.real(s) / wn
    return {"controllability": rep, "Q": np.diag(g.Q).tolist(), "R": float(g.R[0, 0]),
            "K_vnom": K.tolist(), "poles_z": [complex(x).__repr__() for x in p],
            "poles_abs": np.abs(p).tolist(), "wn": wn.tolist(), "zeta": zeta.tolist()}


def fig_nominal(cp, seed=0):
    L = run_trial(seed=seed, cp=cp)
    path = L["path"]
    fig = plt.figure(figsize=(11, 7.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.05, 1.25], hspace=0.55, wspace=0.22)

    ax = fig.add_subplot(gs[:, 0])
    ax.add_patch(Rectangle((0, 0), *ROOM, fill=False, ec=C_INK, lw=1.2))
    for x0, y0, x1, y1 in OBSTACLES:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#d9d8d3", ec="none"))
    for mid, (mx, my, mn) in MARKERS.items():
        ax.plot(mx, my, "s", ms=7 if mid == DOCK_ID else 5, color=C_INK)
        ax.annotate("dock" if mid == DOCK_ID else f"M{mid}", (mx, my), xytext=(4, 4),
                    textcoords="offset points", color=C_MUTED, fontsize=8)
    ax.plot(path.x, path.y, "--", color=C_REF, lw=1.2, label="reference")
    ax.plot(L["x"], L["y"], color=C_TRUE, label="true (evaluation only)")
    ax.plot(L["xh"], L["yh"], color=C_EST, lw=1.1, label="EKF estimate (controller input)")
    i0 = np.searchsorted(path.s, path.s_dock_start)
    ax.plot(path.x[i0], path.y[i0], "o", color=C_INK, ms=4)
    ax.annotate("pre-dock", (path.x[i0], path.y[i0]), xytext=(-10, 8), textcoords="offset points",
                color=C_MUTED, fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-0.2, ROOM[0] + 0.2)
    ax.set_ylim(-0.2, ROOM[1] + 0.2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Path following and docking (seed %d, wheel error %+.1f%%)" % (seed, 100 * L["wheel_err"]))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)

    t = L["t"]
    dk = t[np.argmax(L["dock"])] if L["dock"].any() else t[-1]

    def shade(a, y=0.92):
        a.axvspan(dk, t[-1], color="#f1f0ec", zorder=0)
        a.text(dk - 0.3, y, "final approach", transform=a.get_xaxis_transform(),
               color=C_MUTED, fontsize=8, ha="right")

    a1 = fig.add_subplot(gs[0, 1])
    shade(a1)
    sig = np.sqrt(np.maximum(L["Pxx"], L["Pyy"]))
    a1.fill_between(t, -2 * sig * 100, 2 * sig * 100, color=C_EST, alpha=0.12, lw=0, label="±2σ EKF position")
    a1.plot(t, L["ey_true"] * 100, color=C_TRUE, label="true e_y")
    a1.plot(t, L["ey_est"] * 100, color=C_EST, lw=1.0, label="estimated e_y")
    a1.axhline(0, color=C_MUTED, lw=0.8)
    a1.set_ylabel("cross-track [cm]")
    a1.set_title("Cross-track error (true pose): RMS %.1f cm, mean %+.1f cm"
                 % (L["rms_ey"] * 100, L["mean_ey"] * 100))
    a1.legend(loc="lower left", ncol=3, fontsize=8)

    a2 = fig.add_subplot(gs[1, 1], sharex=a1)
    shade(a2)
    a2.plot(t, L["w"], color=C_TRUE, label="ω applied")
    a2.plot(t, L["w_fb"], color=C_EST, lw=1.0, label="ω feedback (ω − vκ)")
    a2.set_ylabel("turn rate [rad/s]")
    a2.set_title("Steering: RMS feedback %.2f rad/s" % L["rms_wfb"])
    a2.legend(loc="lower left", ncol=2, fontsize=8)

    a3 = fig.add_subplot(gs[2, 1], sharex=a1)
    shade(a3, y=0.45)
    a3.plot(t, L["v"], color=C_TRUE, label="v applied")
    a3.axhline(cp.v_nom, color=C_MUTED, lw=0.8, ls=":")
    a3.set_ylabel("speed [m/s]")
    a3.set_xlabel("time [s]")
    a3.set_title("Speed  (dock error %.1f cm lat, %.1f cm long, %.1f°)"
                 % (100 * L["dock_lat"], 100 * L["dock_lon"], np.rad2deg(L["dock_head"])))
    fig.savefig(os.path.join(OUT, "fig1_nominal_run.png"), bbox_inches="tight")
    plt.close(fig)
    return {k: L[k] for k in ("rms_ey", "mean_ey", "max_ey", "rms_wfb", "sat_frac", "dock_lat",
                              "dock_lon", "dock_head", "success", "t_dock", "wheel_err", "nees_in95")}


def e2_rho_sweep(pool, cp, seeds):
    """Designed sweep of the effort weight. Only the Q/R ratio matters, so
    one scalar rho on R spans the whole speed-vs-effort trade-off."""
    rhos = np.logspace(-1.5, 1.5, 9)
    res = []
    for r in rhos:
        rows = batch(pool, seeds, cp=replace(cp, rho=float(r)))
        g = GainSchedule(replace(cp, rho=float(r)), RobotParams().v_max)
        rows_t = batch(pool, seeds, cp=replace(cp, rho=float(r)), controller_state_source="truth")
        res.append(dict(rho=float(r), ky=float(g.K(cp.v_nom)[0]),
                        rms_ey_truth=agg(rows_t, "rms_ey"), rms_wfb_truth=agg(rows_t, "rms_wfb"),
                        rms_ey=agg(rows, "rms_ey"), rms_wfb=agg(rows, "rms_wfb"),
                        mean_ey=agg(rows, "mean_ey"), success=agg(rows, "success")[0],
                        sat=agg(rows, "sat_frac")[0]))
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
    x = [d["rms_wfb"][0] for d in res]
    y = [100 * d["rms_ey"][0] for d in res]
    axs[0].errorbar(x, y, xerr=[d["rms_wfb"][1] for d in res], yerr=[100 * d["rms_ey"][1] for d in res],
                    fmt="o-", color=C_TRUE, ms=5, lw=1.4, elinewidth=0.8, capsize=2,
                    label="EKF in the loop (delivered)")
    axs[0].plot([d["rms_wfb_truth"][0] for d in res], [100 * d["rms_ey_truth"][0] for d in res],
                "s--", color=C_EST, ms=4, lw=1.2, label="truth-fed baseline (evaluation only)")
    axs[0].legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    for j, (d, xi, yi) in enumerate(zip(res, x, y)):
        if j in (0, len(res) - 1) or abs(np.log(d["rho"] / cp.rho)) < 0.2:   # label ends + chosen only
            axs[0].annotate("ρ=%.2g" % d["rho"], (xi, yi), xytext=(0, 9), textcoords="offset points",
                            fontsize=8, color=C_INK, ha="center")
    axs[0].axvline(SPEC["D3_rms_wfb"], color=C_MUTED, ls=":", lw=1)
    axs[0].axhline(100 * SPEC["D1_rms_ey"], color=C_MUTED, ls=":", lw=1)
    axs[0].text(SPEC["D3_rms_wfb"], 0.3, " D3 limit", fontsize=8, color=C_MUTED)
    axs[0].text(0.6, 100 * SPEC["D1_rms_ey"], "D1 limit ", va="bottom", ha="right", fontsize=8, color=C_MUTED)
    axs[0].set_ylim(0, 100 * SPEC["D1_rms_ey"] + 0.6)
    axs[0].set_xlabel("RMS feedback turn rate [rad/s]")
    axs[0].set_ylabel("RMS true cross-track [cm]")
    axs[0].set_title("Tracking vs effort over ρ (mean ± sd, %d seeds)" % len(seeds))
    axs[1].semilogx([d["rho"] for d in res], [d["ky"] for d in res], "o-", color=C_TRUE, ms=5)
    axs[1].set_xlabel("ρ (scale on R)")
    axs[1].set_ylabel("cross-track gain k_y at v_nom [rad/s per m]")
    axs[1].set_title("Gain on cross-track error vs ρ")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_rho_sweep.png"), bbox_inches="tight")
    plt.close(fig)
    return res


def e3_integral(pool, cp, seeds):
    """Integral action ablation across worn-wheel size and sign."""
    errs = np.array([-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06])
    out = {"wheel_err": errs.tolist(), "with": [], "without": []}
    for name, integ in (("with", True), ("without", False)):
        for e in errs:
            rows = batch(pool, seeds, cp=cp, wheel_err=float(e), integral=integ)
            out[name].append(dict(mean_ey=agg(rows, "mean_ey"), mean_ey_est=agg(rows, "mean_ey_est"),
                                  rms_ey=agg(rows, "rms_ey"),
                                  dock_lat=agg(rows, "dock_lat"), success=agg(rows, "success")[0]))
    # prediction for the no-integral offset: steady state of the linear loop
    # with a yaw-rate disturbance d = v*eps/L gives e_y = d / k_y
    rp = RobotParams()
    ky = GainSchedule(cp, rp.v_max, integral=False).K(cp.v_nom)[0]
    out["pred_offset_no_int"] = (cp.v_nom * errs / rp.track_width / ky).tolist()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for name, col, lab in (("with", C_TRUE, "LQI (with integral)"), ("without", C_EST, "LQR (no integral)")):
        m = [100 * d["mean_ey"][0] for d in out[name]]
        s = [100 * d["mean_ey"][1] for d in out[name]]
        ax.errorbar(100 * errs, m, yerr=s, fmt="o-", color=col, ms=5, capsize=2, elinewidth=0.8, label=lab)
    ax.plot(100 * errs, [100 * d["mean_ey_est"][0] for d in out["with"]], "^:", color=C_ALT, ms=4,
            label="LQI, mean ESTIMATED e_y (what z sees)")
    ax.axhline(0, color=C_MUTED, lw=0.8)
    ax.set_xlabel("right-wheel radius error [%]")
    ax.set_ylabel("mean true cross-track [cm]")
    ax.set_title("Steady offset vs worn-wheel error (mean ± sd, %d seeds)" % len(seeds))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_integral_ablation.png"), bbox_inches="tight")
    plt.close(fig)
    out["ky_no_int"] = float(ky)
    return out


def e4_schedule(pool, cp, seeds):
    """Why the gain is scheduled on speed: closed-loop poles and docking."""
    rp = RobotParams()
    g = GainSchedule(cp, rp.v_max)
    vs = np.linspace(cp.v_creep, rp.v_max, 40)

    def damping(p):
        s = np.log(p.astype(complex)) / cp.T
        return np.min(-np.real(s) / np.abs(s)), np.max(np.abs(p))

    sch = np.array([damping(g.closed_loop_poles(v)) for v in vs])
    fix = np.array([damping(g.closed_loop_poles(cp.v_nom, v)) for v in vs])
    # designed study: how long must the straight final approach be to absorb
    # an error at its start, with and without scheduling? (bump at dock entry)
    bump = (0.0, 0.06, np.deg2rad(-8))      # 6 cm / 8 deg reference jump as the approach starts
    lens = [0.4, 0.7, 1.0, 1.4]
    study = {"approach_len": lens, "sched": [], "fixed": []}
    for Lapp in lens:
        path, _ = build_path(Lapp)
        for name, sched in (("sched", True), ("fixed", False)):
            rows = batch(pool, seeds, cp=cp, schedule=sched, kick_at_dock=bump, path=path)
            study[name].append(dict(success=agg(rows, "success")[0],
                                    abs_lat=float(np.mean([abs(r["dock_lat"]) for r in rows])),
                                    abs_head=float(np.mean([abs(r["dock_head"]) for r in rows]))))
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
    axs[0].plot(vs, sch[:, 0], color=C_TRUE, label="scheduled K(v)")
    axs[0].plot(vs, fix[:, 0], color=C_EST, label="fixed K(v_nom)")
    axs[0].axvline(cp.v_creep, color=C_MUTED, ls=":", lw=1)
    axs[0].text(cp.v_creep, 0.05, " creep speed", fontsize=8, color=C_MUTED)
    axs[0].set_xlabel("actual speed v [m/s]")
    axs[0].set_ylabel("minimum damping ratio ζ")
    axs[0].set_title("Closed-loop damping across speed")
    axs[0].set_ylim(0, 1.05)
    axs[0].legend(fontsize=8)
    axs[1].plot(vs, sch[:, 1], color=C_TRUE, label="scheduled K(v)")
    axs[1].plot(vs, fix[:, 1], color=C_EST, label="fixed K(v_nom)")
    axs[1].set_xlabel("actual speed v [m/s]")
    axs[1].set_ylabel("slowest pole |z|  (1 = no decay)")
    axs[1].set_title("Slowest closed-loop pole")
    axs[1].legend(fontsize=8)
    axs[2].plot(lens, [100 * d["success"] for d in study["sched"]], "o-", color=C_TRUE, ms=5, label="scheduled")
    axs[2].plot(lens, [100 * d["success"] for d in study["fixed"]], "s--", color=C_EST, ms=4, label="fixed")
    axs[2].set_xlabel("final approach length [m]")
    axs[2].set_ylabel("docking success [%]")
    axs[2].set_ylim(-5, 105)
    axs[2].set_title("6 cm / 8° dock-pose jump at approach start (%d seeds)" % len(seeds))
    axs[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_gain_schedule.png"), bbox_inches="tight")
    plt.close(fig)
    # how much of a normal run is actually spent at low speed, where scheduling matters?
    rows_v = pool.map(_low_speed_frac, [dict(seed=sd, cp=cp) for sd in seeds])
    return {"frac_time_v_below_0.1": float(np.mean(rows_v)),
            "zeta_min_creep_sched": float(sch[0, 0]), "zeta_min_creep_fixed": float(fix[0, 0]),
            "zmax_creep_sched": float(sch[0, 1]), "zmax_creep_fixed": float(fix[0, 1]),
            "approach_study": study}


def e5_robustness(pool, cp, seeds):
    """Command latency and motor lag (unmodelled in the design), plus a bump."""
    lat = []
    for d in (0, 1, 2, 4, 6):
        rows = batch(pool, seeds, cp=cp, delay_steps=d)
        lat.append(dict(delay_ms=d * cp.T * 1000, rms_ey=agg(rows, "rms_ey"),
                        rms_wfb=agg(rows, "rms_wfb"), success=agg(rows, "success")[0]))
    tau = []
    for m in (0.02, 0.08, 0.2, 0.4):
        rows = batch(pool, seeds, cp=cp, motor_tau=m)
        tau.append(dict(motor_tau=m, rms_ey=agg(rows, "rms_ey"), success=agg(rows, "success")[0]))
    # bump: robot shoved 10 cm sideways and 10 deg at t = 12 s (sensors not told)
    L = run_trial(seed=3, cp=cp, kick=(12.0, 0.07, -0.07, np.deg2rad(10)))
    t = L["t"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(t, 100 * L["ey_true"], color=C_TRUE, label="true e_y")
    ax.plot(t, 100 * L["ey_est"], color=C_EST, lw=1.0, label="estimated e_y")
    ax.axvline(12.0, color=C_MUTED, ls=":", lw=1)
    ax.text(12.2, 0.92, "bump", transform=ax.get_xaxis_transform(), fontsize=8, color=C_MUTED)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("cross-track [cm]")
    ax.set_title("Recovery from a 10 cm / 10° bump (docked: %s, final lateral %.1f cm)"
                 % ("yes" if L["success"] else "no", 100 * L["dock_lat"]))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_bump.png"), bbox_inches="tight")
    plt.close(fig)
    return {"latency": lat, "motor_tau": tau, "bump": {"success": L["success"], "max_ey": L["max_ey"],
                                                        "dock_lat": L["dock_lat"]}}


def e6_monte_carlo(pool, cp, seeds):
    """Monte Carlo over start pose, wheel error and noise; EKF vs truth-fed baseline."""
    rows = batch(pool, seeds, cp=cp)
    base = batch(pool, seeds, cp=cp, controller_state_source="truth")   # EVALUATION BASELINE ONLY
    lat = 100 * np.array([r["dock_lat"] for r in rows])
    lon = 100 * np.array([r["dock_lon"] for r in rows])
    hd = np.rad2deg([r["dock_head"] for r in rows])
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for a, data, lim, lab in ((axs[0], lat, 100 * DOCK_TOL["lat"], "lateral [cm]"),
                              (axs[1], lon, 100 * DOCK_TOL["lon"], "longitudinal [cm]"),
                              (axs[2], hd, np.rad2deg(DOCK_TOL["head"]), "heading [deg]")):
        a.hist(data, bins=np.linspace(-lim, lim, 33), color=C_TRUE, edgecolor="white", linewidth=1.0)
        for s in (-lim, lim):
            a.axvline(s, color=C_MUTED, ls=":", lw=1)
        a.set_xlabel("dock error, " + lab)
        a.grid(axis="x")
    axs[0].set_ylabel("trials")
    succ = np.mean([r["success"] for r in rows])
    fig.suptitle("Docking accuracy over %d randomised trials - success %.0f%% (dotted = tolerance)"
                 % (len(seeds), 100 * succ), fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_monte_carlo.png"), bbox_inches="tight")
    plt.close(fig)
    keys = ("rms_ey", "mean_ey", "max_ey", "rms_wfb", "sat_frac", "rms_est_pos", "nees_in95", "t_dock")
    return {"n": len(seeds), "success": float(succ),
            "ekf": {k: agg(rows, k) for k in keys},
            "truth_baseline": {k: agg(base, k) for k in keys},
            "truth_baseline_success": float(np.mean([r["success"] for r in base])),
            "dock_abs_p95": {"lat_cm": float(np.percentile(np.abs(lat), 95)),
                             "lon_cm": float(np.percentile(np.abs(lon), 95)),
                             "head_deg": float(np.percentile(np.abs(hd), 95))},
            "worst_mean_ey_abs": float(np.max(np.abs([r["mean_ey"] for r in rows]))),
            "frac_rms_ey_ok": float(np.mean([r["rms_ey"] <= SPEC["D1_rms_ey"] for r in rows])),
            "frac_wfb_ok": float(np.mean([r["rms_wfb"] <= SPEC["D3_rms_wfb"] for r in rows])),
            "frac_mean_ok": float(np.mean([abs(r["mean_ey"]) <= SPEC["D2_mean_ey"] for r in rows])),
            "frac_mean_ok_truth": float(np.mean([abs(r["mean_ey"]) <= SPEC["D2_mean_ey"] for r in base]))}


def write_markdown(res):
    """results/results.md - every table on the project page, generated from this run."""
    d, m, cp = res["design"], res["mc"], res["params"]["controller"]
    e, b = m["ekf"], m["truth_baseline"]
    cm = lambda x: "%.2f" % (100 * x)
    L = ["# Part D results (generated by experiments/run_part_d.py - do not edit by hand)", ""]
    L += ["## Design at v_nom = %.2f m/s, T = %.2f s, rho = %g" % (cp["v_nom"], cp["T"], cp["rho"]), "",
          "| quantity | value |", "|---|---|",
          "| Q = diag(1/e_y,max^2, 1/e_th,max^2, 1/z_max^2) | diag(%.0f, %.1f, %.0f) |" % tuple(d["Q"]),
          "| R = rho / u_max^2 | %.2f |" % d["R"],
          "| K = [k_y, k_th, k_z] | [%.2f, %.2f, %.2f] |" % tuple(d["K_vnom"]),
          "| closed-loop \\|z\\| | %s |" % ", ".join("%.4f" % x for x in d["poles_abs"]),
          "| natural freq. w_n [rad/s] | %s |" % ", ".join("%.2f" % x for x in d["wn"]),
          "| damping ratio | %s |" % ", ".join("%.2f" % x for x in d["zeta"]),
          "| ctrb rank at v = 0 / creep / v_nom | %s |" % " / ".join(str(c["rank"]) for c in d["controllability"]), ""]
    L += ["## Specification check - %d randomised trials (mean ± sd)" % m["n"], "",
          "| req. | metric | limit | EKF in loop | trials passing | truth-fed baseline* |", "|---|---|---|---|---|---|",
          "| D1 | RMS true cross-track [cm] | %s | %s ± %s | %.0f%% | %s |" % (cm(SPEC["D1_rms_ey"]), cm(e["rms_ey"][0]), cm(e["rms_ey"][1]), 100 * m["frac_rms_ey_ok"], cm(b["rms_ey"][0])),
          "| D2 | \\|mean\\| true cross-track [cm] | %s | %s ± %s (worst %s) | %.0f%% | %.0f%% pass |" % (cm(SPEC["D2_mean_ey"]), cm(e["mean_ey"][0]), cm(e["mean_ey"][1]), cm(m["worst_mean_ey_abs"]), 100 * m["frac_mean_ok"], 100 * m["frac_mean_ok_truth"]),
          "| D3 | RMS feedback turn rate [rad/s] | %.2f | %.3f ± %.3f | %.0f%% | %.3f |" % (SPEC["D3_rms_wfb"], e["rms_wfb"][0], e["rms_wfb"][1], 100 * m["frac_wfb_ok"], b["rms_wfb"][0]),
          "| D4 | docking success (2 cm / 3 cm / 5°) | %.0f%% | %.0f%% | - | %.0f%% |" % (100 * SPEC["D4_success"], 100 * m["success"], 100 * m["truth_baseline_success"]),
          "", "95th-percentile \\|dock error\\|: lateral %.2f cm, longitudinal %.2f cm, heading %.2f°. "
          "Mean time to dock %.1f s. Saturated samples %.1f%%. EKF RMS position error %.2f cm, NEES inside 95%% bound %.0f%%."
          % (m["dock_abs_p95"]["lat_cm"], m["dock_abs_p95"]["lon_cm"], m["dock_abs_p95"]["head_deg"], e["t_dock"][0],
             100 * e["sat_frac"][0], 100 * e["rms_est_pos"][0], 100 * e["nees_in95"][0]),
          "", "*Truth-fed baseline: controller given ground truth instead of the EKF. Evaluation only, to measure what estimation costs.", ""]
    L += ["## rho sweep (%s seeds per point)" % "20", "", "| rho | k_y | RMS e_y EKF [cm] | RMS w_fb EKF | RMS e_y truth-fed [cm] | RMS w_fb truth-fed | saturated |", "|---|---|---|---|---|---|---|"]
    for r in res["rho_sweep"]:
        L.append("| %.3g | %.1f | %s | %.3f | %s | %.3f | %.1f%% |" % (r["rho"], r["ky"], cm(r["rms_ey"][0]), r["rms_wfb"][0],
                                                                   cm(r["rms_ey_truth"][0]), r["rms_wfb_truth"][0], 100 * r["sat"]))
    I = res["integral"]
    L += ["", "## Integral action vs worn wheel (10 seeds per point)", "",
          "| wheel error | LQI mean true e_y [cm] | LQI mean estimated e_y [cm] | LQR mean true e_y [cm] |", "|---|---|---|---|"]
    for w, a, c in zip(I["wheel_err"], I["with"], I["without"]):
        L.append("| %+.0f%% | %s | %s | %s |" % (100 * w, cm(a["mean_ey"][0]), cm(a["mean_ey_est"][0]), cm(c["mean_ey"][0])))
    S = res["schedule"]
    L += ["", "## Gain schedule and final-approach length", "",
          "Minimum damping at creep speed: scheduled %.2f, fixed %.2f. Fraction of a normal run spent below 0.1 m/s: %.1f%%."
          % (S["zeta_min_creep_sched"], S["zeta_min_creep_fixed"], 100 * S["frac_time_v_below_0.1"]), "",
          "| approach length [m] | success scheduled | success fixed | mean \\|lat\\| scheduled [cm] | mean \\|head\\| scheduled [deg] |", "|---|---|---|---|---|"]
    st = S["approach_study"]
    for i, la in enumerate(st["approach_len"]):
        a, f = st["sched"][i], st["fixed"][i]
        L.append("| %.1f | %.0f%% | %.0f%% | %s | %.1f |" % (la, 100 * a["success"], 100 * f["success"], cm(a["abs_lat"]), np.rad2deg(a["abs_head"])))
    R = res["robust"]
    L += ["", "## Robustness (10 seeds per point)", "", "| command latency [ms] | RMS e_y [cm] | RMS w_fb | success |", "|---|---|---|---|"]
    for r in R["latency"]:
        L.append("| %.0f | %s | %.3f | %.0f%% |" % (r["delay_ms"], cm(r["rms_ey"][0]), r["rms_wfb"][0], 100 * r["success"]))
    L += ["", "| motor time constant [s] | RMS e_y [cm] | success |", "|---|---|---|"]
    for r in R["motor_tau"]:
        L.append("| %.2f | %s | %.0f%% |" % (r["motor_tau"], cm(r["rms_ey"][0]), 100 * r["success"]))
    L += ["", "Bump test (10 cm / 10° at t = 12 s): max true \\|e_y\\| %s cm, docked: %s." % (cm(R["bump"]["max_ey"]), R["bump"]["success"])]
    with open(os.path.join(OUT, "results.md"), "w") as f:
        f.write("\n".join(L) + "\n")


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rho", type=float, default=None, help="override the chosen rho")
    args = ap.parse_args()
    n_sweep, n_mc = (6, 20) if args.quick else (20, 100)
    rp = RobotParams()
    cp = ControllerParams() if args.rho is None else ControllerParams(rho=args.rho)
    res = {"spec": SPEC, "params": {"robot": vars(rp), "controller": vars(cp)}}
    sweep_seeds = list(range(100, 100 + n_sweep))     # tuning seeds ...
    mc_seeds = list(range(1000, 1000 + n_mc))          # ... disjoint from evaluation seeds
    with Pool() as pool:
        print("E0 design");          res["design"] = e0_design(cp, rp)
        print("E1 nominal run");     res["nominal"] = fig_nominal(cp)
        print("E2 rho sweep");       res["rho_sweep"] = e2_rho_sweep(pool, cp, sweep_seeds)
        print("E3 integral");        res["integral"] = e3_integral(pool, cp, sweep_seeds[:10])
        print("E4 schedule");        res["schedule"] = e4_schedule(pool, cp, sweep_seeds)
        print("E5 robustness");      res["robust"] = e5_robustness(pool, cp, sweep_seeds[:10])
        print("E6 Monte Carlo");     res["mc"] = e6_monte_carlo(pool, cp, mc_seeds)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    write_markdown(res)
    print(open(os.path.join(OUT, "results.md")).read())


if __name__ == "__main__":
    main()
