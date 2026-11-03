"""
plot_log.py — compare CSV logs written by mixing_v4.py (Gazebo VM and real robot).

    python3 plot_log.py logs/mixing_command_*.csv logs/mixing_measured_*.csv

Needs only numpy + matplotlib (no ROS), so it also runs outside the VM.
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    t = d["t"]
    xd = np.stack([d["xdes_x"], d["xdes_y"], d["xdes_z"]], 1)
    x = np.stack([d["x_x"], d["x_y"], d["x_z"]], 1)
    q = np.stack([d[f"q{i}"] for i in range(7)], 1)
    return dict(name=os.path.basename(path), t=t, xd=xd, x=x, eR=d["eR"], q=q)


def metrics(L, skip_start=3.0, skip_end=2.0):
    t = L["t"]
    m = (t > skip_start) & (t < t[-1] - skip_end)
    e = L["xd"] - L["x"]
    xy = np.linalg.norm(e[:, :2], axis=1)
    return {
        "xy RMS err [mm]": 1000 * np.sqrt(np.mean(xy[m] ** 2)),
        "z mean err [mm]": 1000 * np.mean(e[m, 2]),
        "z max |err| [mm]": 1000 * np.abs(e[m, 2]).max(),
        "tilt max [deg]": np.degrees(L["eR"][m].max()),
        "joint drift ||q_end - q_0|| [rad]": np.linalg.norm(L["q"][-1] - L["q"][0]),
        "loop rate [Hz]": (len(t) - 1) / (t[-1] - t[0]),
    }


def main(paths):
    logs = [load(p) for p in paths]
    for L in logs:
        print(f"\n{L['name']}")
        for k, v in metrics(L).items():
            print(f"  {k:36s}{v:9.3f}")

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for L in logs:
        t, e = L["t"], L["xd"] - L["x"]
        x0 = L["xd"][0]
        ax[0, 0].plot(1000 * (L["x"][:, 0] - x0[0]), 1000 * (L["x"][:, 1] - x0[1]), lw=1, label=L["name"])
        ax[0, 1].plot(t, -1000 * e[:, 2], lw=1, label=L["name"])
        ax[1, 0].plot(t, np.degrees(L["eR"]), lw=1, label=L["name"])
        ax[1, 1].plot(t, 1000 * np.linalg.norm(e[:, :2], axis=1), lw=1, label=L["name"])
    ax[0, 0].set(title="Top view", xlabel="x [mm]", ylabel="y [mm]"); ax[0, 0].axis("equal")
    ax[0, 1].set(title="z - z_des (sag < 0)", xlabel="t [s]", ylabel="mm")
    ax[1, 0].set(title="Cup tilt", xlabel="t [s]", ylabel="deg")
    ax[1, 1].set(title="Horizontal error", xlabel="t [s]", ylabel="mm")
    for a in ax.flat:
        a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout()
    out = "log_comparison.png"
    fig.savefig(out, dpi=130)
    print(f"\nplot saved to {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
