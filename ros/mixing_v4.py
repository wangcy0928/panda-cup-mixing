"""
mixing_v4.py — the v4 mixing controller on the MEAM 5200 ROS interface.

Run next to the course code, with Gazebo launched:
    python3 mixing_v4.py                   # integrate from the previous command
    python3 mixing_v4.py --from-measured   # 3.py-style, for comparison
Each run writes a CSV to logs/; compare runs with plot_log.py.
"""
import os
import sys
import csv
import time
import numpy as np
import rospy
import geometry_msgs.msg
import visualization_msgs.msg

from core.interfaces import ArmController
from core.utils import time_in_seconds
from lib.calculateFK import FK
from lib.calcJacobian import calcJacobian

Q_DEMO = np.array([0, -1, 0, -2, 0, 1, 1], dtype=float)
Q_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_UPPER = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

# ---------------- parameters (same as Config.improved() in the simulation) ----------------
DURATION = 20.0     # s
W = 2.5             # rad/s
RX, RY = 0.12, 0.07 # m
RAMP_IN, RAMP_OUT = 2.0, 2.0
KP, KZ, KR = 8.0, 8.0, 4.0
DLS_LAMBDA = 0.02
K_NULL = 1.0
DQ_LIMIT = 1.5      # rad/s, whole vector scaled so the largest joint hits this
MAX_LEAD = 0.10     # rad: safety bound on how far q_cmd may run ahead of measured q
RATE_HZ = 100


def rot_error(R_des, R_cur):
    """World-frame rotation vector taking R_cur to R_des (same as _calc_ang_diff in 3.py)."""
    R_rel = R_cur.T @ R_des
    S = 0.5 * (R_rel - R_rel.T)
    return R_cur @ np.array([S[2, 1], S[0, 2], S[1, 0]])


def ramp(t):
    """Amplitude s(t) and ds/dt: smoothstep in over RAMP_IN, smoothstep out over RAMP_OUT."""
    def sstep(u):
        u = min(max(u, 0.0), 1.0)
        return 3 * u * u - 2 * u ** 3, 6 * u - 6 * u * u
    s_in, d_in = sstep(t / RAMP_IN)
    s_out, d_out = sstep((DURATION - t) / RAMP_OUT)
    d_in, d_out = d_in / RAMP_IN, -d_out / RAMP_OUT
    return s_in * s_out, d_in * s_out + s_in * d_out


def ellipse(t, x0):
    s, ds = ramp(t)
    c, sn = np.cos(W * t), np.sin(W * t)
    shape = np.array([RX * c, RY * sn, 0.0])
    dshape = np.array([-RX * W * sn, RY * W * c, 0.0])
    return x0 + s * shape, s * dshape + ds * shape


def mixing_motion_v4(arm, from_measured=False, log_dir="logs"):
    arm.set_arm_speed(0.5)
    print("[mixing_v4] moving to start pose...")
    arm.safe_move_to_position(Q_DEMO)

    fk = FK()
    q_home = np.array(arm.get_positions(), dtype=float)
    _, T0 = fk.forward(q_home)
    x0, R_home = T0[:3, 3].copy(), T0[:3, :3].copy()
    print(f"[mixing_v4] x0 from FK = {np.round(x0, 4)}   mode = "
          f"{'from MEASURED q (3.py style)' if from_measured else 'from COMMAND (v4)'}")

    # ---- logging ----
    os.makedirs(log_dir, exist_ok=True)
    mode = "measured" if from_measured else "command"
    path = os.path.join(log_dir, f"mixing_{mode}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    f = open(path, "w", newline="")
    log = csv.writer(f)
    log.writerow(["t"] + [f"xdes_{a}" for a in "xyz"] + [f"x_{a}" for a in "xyz"] + ["eR"]
                 + [f"q{i}" for i in range(7)] + [f"dq{i}" for i in range(7)]
                 + [f"qcmd{i}" for i in range(7)])

    # ---- RViz trace (10 Hz, at most 1500 points) ----
    pub = rospy.Publisher('/vis/trace', visualization_msgs.msg.Marker, queue_size=10)
    marker = visualization_msgs.msg.Marker()
    marker.header.frame_id = "world"
    marker.ns, marker.id = "mixing_trace_v4", 0
    marker.type = visualization_msgs.msg.Marker.LINE_STRIP
    marker.action = visualization_msgs.msg.Marker.ADD
    marker.scale.x = 0.003
    marker.color.g, marker.color.b, marker.color.a = 0.8, 1.0, 1.0
    marker.pose.orientation.w = 1.0

    rate = rospy.Rate(RATE_HZ)
    q_cmd = q_home.copy()
    t0 = time_in_seconds()
    t_last = t0
    k = 0
    lead_warned = False
    while not rospy.is_shutdown():
        now = time_in_seconds()
        t = now - t0
        if t >= DURATION:
            break
        dt = min(max(now - t_last, 1e-4), 2.0 / RATE_HZ)
        t_last = now

        x_des, v_des = ellipse(t, x0)
        q = np.array(arm.get_positions(), dtype=float)
        _, T = fk.forward(q)
        x, R = T[:3, 3], T[:3, :3]

        # 1) task-space feedback (feed-forward + P), 2) orientation hold
        e = x_des - x
        v = v_des + np.array([KP * e[0], KP * e[1], KZ * e[2]])
        e_R = rot_error(R_home, R)
        xi = np.hstack([v, KR * e_R])

        # 3) damped least squares IK
        J = calcJacobian(q)
        J_pinv = J.T @ np.linalg.inv(J @ J.T + DLS_LAMBDA ** 2 * np.eye(6))
        dq = J_pinv @ xi

        # 4) null space: pull every joint weakly toward the start configuration
        dq += (np.eye(7) - J_pinv @ J) @ (K_NULL * (q_home - q))

        # 5) uniform scaling keeps the direction of motion
        ratio = np.max(np.abs(dq)) / DQ_LIMIT
        if ratio > 1.0:
            dq /= ratio

        # 6) integrate to a position command
        if from_measured:
            q_cmd = q + dt * dq                       # 3.py
        else:
            q_cmd = q_cmd + dt * dq                   # v4
            lead = q_cmd - q
            if np.any(np.abs(lead) > MAX_LEAD):       # safety: never let the command run away
                if not lead_warned:
                    print("[mixing_v4] WARNING: command lead limited — robot not following?")
                    lead_warned = True
                q_cmd = q + np.clip(lead, -MAX_LEAD, MAX_LEAD)
        q_cmd = np.clip(q_cmd, Q_LOWER, Q_UPPER)

        arm.safe_set_joint_positions_velocities(q_cmd, dq)

        log.writerow([f"{t:.4f}"] + [f"{a:.6f}" for a in (*x_des, *x)]
                     + [f"{np.linalg.norm(e_R):.6f}"]
                     + [f"{a:.6f}" for a in (*q, *dq, *q_cmd)])

        if k % 10 == 0 and len(marker.points) < 1500:
            pt = geometry_msgs.msg.Point()
            pt.x, pt.y, pt.z = float(x[0]), float(x[1]), float(x[2])
            marker.points.append(pt)
            marker.header.stamp = rospy.Time.now()
            pub.publish(marker)
        k += 1
        rate.sleep()

    f.close()
    print(f"[mixing_v4] done, {k} ticks logged to {path}")
    # the ramp-out already brought the cup to rest; hold briefly, then go home
    arm.safe_set_joint_positions_velocities(q_cmd, np.zeros(7))
    rospy.sleep(0.5)
    arm.safe_move_to_position(Q_DEMO)
    return path


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--duration" in args:
        DURATION = float(args[args.index("--duration") + 1])
    rospy.init_node("mixing_v4")
    arm2 = ArmController(id=2)        # same arm as 3.py; change if your setup differs
    mixing_motion_v4(arm2, from_measured="--from-measured" in args)
