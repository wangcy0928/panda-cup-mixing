"""
animate.py — renders the improved controller as a GIF (stick-figure Panda + cup path).

    python animate.py            # -> results/mixing.gif   (~1 min)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from kinematics import fk_all
from sim_robot import SimPanda
from controller import Config, run

Q_DEMO = np.array([0, -1, 0, -2, 0, 1, 1], dtype=float)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main(every=10, fps=20):
    os.makedirs(OUT, exist_ok=True)
    robot = SimPanda(Q_DEMO)
    robot.settle(Q_DEMO, 3.0)
    log = run(Config.improved(), robot, Q_DEMO)
    q, x = log["q"], log["x"]
    frames = range(0, len(q), every)

    fig = plt.figure(figsize=(9, 4.5))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)

    arm_line, = ax3.plot([], [], [], "o-", color="#2f7ed8", lw=3, ms=4)
    trace3, = ax3.plot([], [], [], color="#e8a33d", lw=1)
    cup, = ax3.plot([], [], [], "s", color="#8c564b", ms=9)
    ax3.set(xlim=(-0.3, 0.4), ylim=(-0.35, 0.35), zlim=(0, 0.9),
            xlabel="x", ylabel="y", zlabel="z")
    ax3.set_box_aspect((0.7, 0.7, 0.9))
    ax3.view_init(elev=18, azim=-60)
    ax3.set_title("Panda arm (simplified sim)")

    xd = log["x_des"]
    ax2.plot(1000 * (xd[:, 0] - log["x0"][0]), 1000 * (xd[:, 1] - log["x0"][1]),
             "k--", lw=0.8, label="desired")
    trace2, = ax2.plot([], [], color="#e8a33d", lw=1.5, label="actual")
    dot2, = ax2.plot([], [], "o", color="#8c564b")
    ax2.set(xlim=(-150, 150), ylim=(-100, 100), xlabel="x [mm]", ylabel="y [mm]",
            title="Top view of the cup")
    ax2.set_aspect("equal"); ax2.grid(alpha=0.3); ax2.legend(loc="upper right", fontsize=8)
    txt = ax2.text(-140, -90, "", fontsize=9)
    fig.tight_layout()

    def update(k):
        pts = np.array([[0, 0, 0]] + [T[:3, 3] for T in fk_all(q[k])])
        arm_line.set_data(pts[:, 0], pts[:, 1]); arm_line.set_3d_properties(pts[:, 2])
        trace3.set_data(x[:k, 0], x[:k, 1]); trace3.set_3d_properties(x[:k, 2])
        cup.set_data([x[k, 0]], [x[k, 1]]); cup.set_3d_properties([x[k, 2]])
        trace2.set_data(1000 * (x[:k, 0] - log["x0"][0]), 1000 * (x[:k, 1] - log["x0"][1]))
        dot2.set_data([1000 * (x[k, 0] - log["x0"][0])], [1000 * (x[k, 1] - log["x0"][1])])
        err = 1000 * np.linalg.norm(log["x_des"][k] - x[k])
        txt.set_text(f"t = {log['t'][k]:5.1f} s   |error| = {err:4.1f} mm")
        return arm_line, trace3, cup, trace2, dot2, txt

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    path = os.path.join(OUT, "mixing.gif")
    anim.save(path, writer=PillowWriter(fps=fps), dpi=70)
    print("saved", path)


if __name__ == "__main__":
    main()
