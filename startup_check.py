"""
startup_check.py — why the ellipse centre x0 must come from forward kinematics.

Runs the course controller (v3 gains) twice:
  1. x0 = FK(actual joint angles)          -> the correct way
  2. x0 hard-coded 20 mm off in z          -> what happens when the handoff pose is not exact
and plots the end-effector error and the largest joint-velocity command over time.

If a wrong x0 is the cause of the startup jerk, case 2 should show one spike at t = 0
that then decays, after which both curves should look the same.

It also repeats the run with several offsets and prints the peak cup acceleration
in the first second, to show that the jerk grows with the offset.

    python startup_check.py      # -> results/6_startup_x0_check.png
"""
import os
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import controller
from sim_robot import SimPanda
from controller import Config, run

Q_DEMO = np.array([0, -1, 0, -2, 0, 1, 1], dtype=float)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DT = 0.01


def simulate(offset_m, base=None, duration=6.0):
    """
    Run the controller with the ellipse centre x0 shifted by offset_m (metres).
    run() computes x0 from FK of the actual pose; shifting it here mimics a hard-coded
    x0 that does not match where the arm really stopped.
    """
    offset = np.asarray(offset_m, dtype=float)
    original = controller.ellipse
    controller.ellipse = lambda t, x0, cfg: original(t, x0 + offset, cfg)
    try:
        cfg = replace(base or Config.original_v3(), duration=duration)
        robot = SimPanda(Q_DEMO)
        robot.settle(Q_DEMO, 3.0)      # arrive at the start pose, like safe_move_to_position
        return run(cfg, robot, Q_DEMO, dt=DT)
    finally:
        controller.ellipse = original


def peak_accel(log, until=1.0):
    """Peak horizontal+vertical cup acceleration in the first `until` seconds (m/s^2)."""
    m = log["t"] < until
    acc = np.linalg.norm(np.diff(log["x"][m], n=2, axis=0), axis=1) / DT**2
    return np.convolve(acc, np.ones(10) / 10, mode="same").max()   # 0.1 s smoothing


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---- 1. the two curves ----
    runs = {
        "x0 from FK (actual pose)": simulate([0, 0, 0]),
        "hard-coded x0, 20 mm off in z": simulate([0, 0, 0.020]),
    }
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for (name, log), c in zip(runs.items(), ["#2f7ed8", "#d9534f"]):
        m = log["t"] <= 6.0
        t = log["t"][m]
        ax[0].plot(t, 1000 * np.linalg.norm(log["x_des"][m] - log["x"][m], axis=1), color=c, label=name)
        ax[1].plot(t, np.abs(log["dq_cmd"][m]).max(axis=1), color=c, label=name)
    ax[0].set(title="End-effector error |x_des - x|", xlabel="t [s]", ylabel="mm")
    ax[1].set(title="Largest joint velocity command", xlabel="t [s]", ylabel="rad/s")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
        a.axvline(0, color="k", lw=0.6)
    fig.suptitle("Startup check: a wrong x0 shows up as a spike at t = 0 that then decays")
    fig.tight_layout()
    path = os.path.join(OUT, "6_startup_x0_check.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)

    for name, log in runs.items():
        e = 1000 * np.linalg.norm(log["x_des"] - log["x"], axis=1)
        print(f"{name:32s} error at start {e[0]:5.1f} mm | "
              f"max joint-velocity command in 1st s {np.abs(log['dq_cmd'][log['t'] < 1]).max():.2f} rad/s")

    # ---- 2. does the jerk grow with the offset? ----
    print("\nPeak cup acceleration in the first second [m/s^2]")
    print(f"{'gains':12s}{'axis':6s}" + "".join(f"{mm:>8d} mm" for mm in (0, 20, 50)))
    for gains, base in [("v3", Config.original_v3()), ("v4", Config.improved())]:
        for axis, k in [("x", 0), ("z", 2)]:
            row = []
            for mm in (0, 20, 50):
                off = np.zeros(3)
                off[k] = mm / 1000
                row.append(peak_accel(simulate(off, base, duration=4.0)))
            print(f"{gains:12s}{axis:6s}" + "".join(f"{a:11.2f}" for a in row))
    print(f"\nplot written to {path}")


if __name__ == "__main__":
    main()
