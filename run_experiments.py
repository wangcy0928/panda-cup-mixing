"""
run_experiments.py — reproduces the project history (v1 to v3) in simulation, then
measures what each improvement in v4 contributes.

    python run_experiments.py            # all experiments + plots  (~1-2 min)
    python run_experiments.py --quick    # only v3 vs v4

Outputs go to results/:  metrics.md, *.png
"""
import os, sys
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim_robot import SimPanda
from controller import Config, run, metrics

Q_DEMO = np.array([0, -1, 0, -2, 0, 1, 1], dtype=float)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def experiment(cfg, payload=0.12):
    robot = SimPanda(Q_DEMO, payload_kg=payload)
    robot.settle(Q_DEMO, seconds=3.0)      # like safe_move_to_position: arrive and sag
    log = run(cfg, robot, Q_DEMO)
    return log, metrics(log)


def history_configs():
    v3 = Config.original_v3()
    return [
        replace(v3, label="v1: P-only, circle-style gains", kz=1.0, ki_z=0.0, null_space=False),
        replace(v3, label="v2: offline plan + replay", offline_replay=True),
        v3,
        Config.improved(),
    ]


def ablation_configs():
    v4 = Config.improved()
    return [
        v4,
        replace(v4, label="v4 but integrate from MEASURED q", integrate_from_command=False),
        replace(v4, label="v4 but kr = 0.05 (as 3.py)", kr=0.05),
        replace(v4, label="v4 but LPF alpha = 0.35", lpf_alpha=0.35),
        replace(v4, label="v4 but no null space", null_space=False),
        replace(v4, label="v4 but no ramp-out (abrupt stop)", ramp_out=0.0, ramp_ff_term=False),
        replace(v4, label="v4 but kp = kz = 1 (as 3.py)", kp=1.0, kz=1.0),
    ]


def table(rows, title):
    keys = list(rows[0][1].keys())
    s = f"### {title}\n\n| config | " + " | ".join(keys) + " |\n"
    s += "|---" * (len(keys) + 1) + "|\n"
    for label, m in rows:
        s += f"| {label} | " + " | ".join(f"{m[k]:.3g}" for k in keys) + " |\n"
    return s + "\n"


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
COLORS = ["#9aa0a6", "#e8a33d", "#d9534f", "#2f7ed8", "#6a3d9a", "#2ca02c", "#8c564b"]


def plot_history(logs):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for (log, c) in zip(logs, COLORS):
        t, x, xd = log["t"], log["x"], log["x_des"]
        ax[0, 0].plot(1000 * (x[:, 0] - log["x0"][0]), 1000 * (x[:, 1] - log["x0"][1]),
                      color=c, lw=1, label=log["label"])
        ax[0, 1].plot(t, 1000 * (x[:, 2] - xd[:, 2]), color=c, lw=1, label=log["label"])
        ax[1, 0].plot(t, np.degrees(log["e_R"]), color=c, lw=1, label=log["label"])
        ax[1, 1].plot(t, 1000 * np.linalg.norm((xd - x)[:, :2], axis=1), color=c, lw=1,
                      label=log["label"])
    xd = logs[-1]["x_des"]
    ax[0, 0].plot(1000 * (xd[:, 0] - logs[-1]["x0"][0]), 1000 * (xd[:, 1] - logs[-1]["x0"][1]),
                  "k--", lw=0.8, label="desired")
    ax[0, 0].set(title="Top view of end-effector path", xlabel="x [mm]", ylabel="y [mm]")
    ax[0, 0].axis("equal")
    ax[0, 1].set(title="Height error z - z_des  (sag < 0)", xlabel="t [s]", ylabel="mm")
    ax[1, 0].set(title="Cup tilt |e_R|", xlabel="t [s]", ylabel="deg")
    ax[1, 1].set(title="Horizontal tracking error", xlabel="t [s]", ylabel="mm")
    for a in ax.flat:
        a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Project history reproduced in simulation (same load for all)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "1_history.png"), dpi=130)
    plt.close(fig)


def plot_key_idea(log_meas, log_cmd):
    """Steady-state error for the two choices of integration start."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for log, c, name in [(log_meas, COLORS[2], "q_cmd = q_measured + dt·dq"),
                         (log_cmd, COLORS[3], "q_cmd = q_cmd + dt·dq")]:
        ax[0].plot(log["t"], 1000 * (log["x"][:, 2] - log["x_des"][:, 2]), color=c, label=name)
        ax[1].plot(log["t"], 1000 * np.linalg.norm((log["x_des"] - log["x"])[:, :2], axis=1),
                   color=c, label=name)
    ax[0].set(title="z error", xlabel="t [s]", ylabel="mm")
    ax[1].set(title="xy error", xlabel="t [s]", ylabel="mm")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle("Same gains, only the integration start point differs")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "2_integrate_from_command.png"), dpi=130)
    plt.close(fig)


def plot_nullspace(log_on, log_off):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for log, c, name in [(log_off, COLORS[2], "no null space"), (log_on, COLORS[3], "null space on")]:
        q = log["q"]
        ax.plot(log["t"], np.linalg.norm(q - q[0], axis=1), color=c, label=name)
    ax.set(title="Joint-space drift ‖q(t) − q(0)‖ while the EE repeats the same ellipse",
           xlabel="t [s]", ylabel="rad")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "3_null_space_drift.png"), dpi=130)
    plt.close(fig)


def plot_stop(log_abrupt, log_ramp):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for log, c, name in [(log_abrupt, COLORS[2], "abrupt stop (3.py)"),
                         (log_ramp, COLORS[3], "2 s smooth ramp-out")]:
        t, x = log["t"], log["x"]
        dt = t[1] - t[0]
        v = np.linalg.norm(np.diff(x[:, :2], axis=0), axis=1) / dt
        ax.plot(t[1:], v, color=c, lw=1, label=name)
    ax.axvline(20.0, color="k", ls=":", lw=1)
    ax.set(title="Cup horizontal speed at the end of mixing", xlabel="t [s]", ylabel="m/s",
           xlim=(15, 21))
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "4_ramp_out.png"), dpi=130)
    plt.close(fig)


def plot_filter(log):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    t = log["t"]
    ax.plot(t, log["dq_raw"][:, 3], color=COLORS[0], lw=1, label="raw dq (joint 4)")
    ax.plot(t, log["dq_cmd"][:, 3], color=COLORS[2], lw=1.2, label="filtered dq, alpha=0.35")
    ax.text(8.05, -0.55, "first-order filter time constant  dt·(1−α)/α ≈ 19 ms", fontsize=9)
    ax.set(title="Low-pass filter inside a feedback loop = lag (v3)", xlabel="t [s]",
           ylabel="rad/s", xlim=(8, 11))
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "5_lpf_lag.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    quick = "--quick" in sys.argv
    md = "# Simulation results\n\nGenerated by `run_experiments.py`. Same robot model, " \
         "same load (0.12 kg effective) for every run. Steady-state metrics skip the first " \
         "3 s and last 2 s.\n\n"

    cfgs = history_configs()
    if quick:
        cfgs = cfgs[2:]
    hist = []
    for cfg in cfgs:
        print(f"running: {cfg.label}")
        hist.append(experiment(cfg))
    md += table([(l["label"], m) for l, m in hist], "A. Project history (v1 → v4)")
    plot_history([l for l, _ in hist])
    plot_filter(hist[-2][0])

    if not quick:
        abl = []
        for cfg in ablation_configs():
            print(f"running: {cfg.label}")
            abl.append(experiment(cfg))
        md += table([(l["label"], m) for l, m in abl],
                    "B. Ablation: v4 with ONE improvement removed at a time")
        plot_key_idea(abl[1][0], abl[0][0])
        plot_nullspace(abl[0][0], abl[4][0])
        plot_stop(abl[5][0], abl[0][0])

    with open(os.path.join(OUT, "metrics.md"), "w") as f:
        f.write(md)
    print(md)
    print(f"plots + metrics written to {OUT}")
