"""
export_viewer.py — run v3 and v4 in the simulation and write a self-contained 3D viewer.

    python export_viewer.py        # -> docs/index.html (open it in a browser)

The page replays the saved joint trajectories with three.js (loaded from a CDN, so it needs
an internet connection). The viewer code lives in viewer_template.html.
"""
import json
import os
import numpy as np

from kinematics import fk_all
from sim_robot import SimPanda
from controller import Config, run, metrics

HERE = os.path.dirname(os.path.abspath(__file__))
Q_DEMO = np.array([0, -1, 0, -2, 0, 1, 1], dtype=float)
STEP = 2                      # keep every 2nd tick (50 Hz) to keep the file small
STAT_KEYS = ["xy RMS err [mm]", "z mean err [mm]", "z max |err| [mm]",
             "tilt max [deg]", "max horiz acc [m/s^2]"]


def r(a, n=4):
    return [round(float(v), n) for v in a]


def export_run(cfg):
    robot = SimPanda(Q_DEMO)
    robot.settle(Q_DEMO, 3.0)
    log = run(cfg, robot, Q_DEMO)
    frames = []
    for q in log["q"][::STEP]:
        frames.append(r([v for T in fk_all(q) for v in (*T[:3, 3], *T[:3, 0], *T[:3, 1], *T[:3, 2])]))
    m = metrics(log)
    return {
        "frames": frames,
        "x_des": [r(p) for p in log["x_des"][::STEP]],
        "x": [r(p) for p in log["x"][::STEP]],
        "x0": r(log["x0"]),
        "dt": 0.01 * STEP,
        "stats": {k: float(m[k]) for k in STAT_KEYS},
    }


def build(template_path=os.path.join(HERE, "viewer_template.html")):
    data = {"v3": export_run(Config.original_v3()), "v4": export_run(Config.improved())}
    html = open(template_path, encoding="utf-8").read()
    return html.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))


def main():
    page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '</head>\n<body>\n' + build() + '\n</body>\n</html>\n')
    out = os.path.join(HERE, "docs")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    print("saved", path, f"({len(page) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
