"""
kinematics.py — Franka Panda forward kinematics (FK) and geometric Jacobian.
正运动学 + 几何雅可比，只用 numpy。

Convention: Craig's *modified* DH, the one Franka publishes officially.
约定：Craig 改进型 DH 参数（Franka 官方文档使用的版本）。

"""
import numpy as np

# (a, d, alpha) for joints 1..7   —  theta = q_i
DH = [
    (0.0,     0.333,  0.0),
    (0.0,     0.0,   -np.pi / 2),
    (0.0,     0.316,  np.pi / 2),
    (0.0825,  0.0,    np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0,     0.0,    np.pi / 2),
    (0.088,   0.0,    np.pi / 2),
]
D_FLANGE = 0.107     # flange offset along z7
D_TCP = 0.1034       # Franka Hand: flange -> fingertip centre (tool centre point)

Q_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_UPPER = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
DQ_MAX  = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])   # rad/s


def _mdh(a, d, alpha, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct,      -st,      0.0,  a],
        [st * ca,  ct * ca, -sa, -d * sa],
        [st * sa,  ct * sa,  ca,  d * ca],
        [0.0,      0.0,      0.0, 1.0],
    ])


def fk_all(q):
    """
    Returns a list of 4x4 transforms: frames 1..7 (each joint's frame), then the
    end-effector (TCP).  frames[i] is where joint i+1's axis lives (its z axis).
    返回每个关节坐标系 + 末端坐标系，用于画图和算雅可比。
    """
    T = np.eye(4)
    frames = []
    for (a, d, alpha), qi in zip(DH, q):
        T = T @ _mdh(a, d, alpha, qi)
        frames.append(T.copy())
    # flange, then hand rotated -45 deg about z (how the Franka Hand is mounted)
    T = T @ _mdh(0.0, D_FLANGE, 0.0, 0.0)
    T = T @ _mdh(0.0, D_TCP, 0.0, -np.pi / 4)
    frames.append(T.copy())
    return frames


def fk(q):
    """End-effector pose T_0e (4x4).  末端位姿。"""
    return fk_all(q)[-1]


def jacobian(q):
    """
    Geometric Jacobian J (6x7) in the world frame, at the TCP.
    几何雅可比：前三行线速度 Jv，后三行角速度 Jw。

    Column i (revolute joint about axis z_i through point p_i):
        Jv_i = z_i × (p_e − p_i)      linear velocity contributed by joint i
        Jw_i = z_i                    angular velocity contributed by joint i
    """
    frames = fk_all(q)
    p_e = frames[-1][:3, 3]
    J = np.zeros((6, 7))
    for i in range(7):
        z_i = frames[i][:3, 2]
        p_i = frames[i][:3, 3]
        J[:3, i] = np.cross(z_i, p_e - p_i)
        J[3:, i] = z_i
    return J


def rot_error(R_des, R_cur):
    """
    Orientation error as a world-frame rotation vector (axis * angle, small-angle form).
    姿态误差：从 R_cur 转到 R_des 所需的旋转向量（世界系）。
    Same idea as _calc_ang_diff in the original 3.py.
    """
    R_rel = R_cur.T @ R_des                  # rotation still needed, in the current EE frame
    S = 0.5 * (R_rel - R_rel.T)              # skew part = sin(angle) * [axis]x
    u = np.array([S[2, 1], S[0, 2], S[1, 0]])
    return R_cur @ u                         # express in world frame


def manipulability(q):
    """Yoshikawa manipulability sqrt(det(J J^T)) — goes to 0 at a singularity. 可操作度。"""
    J = jacobian(q)
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))


# --------------------------------------------------------------------------- #
# Self-test:  python kinematics.py
# Checks the analytic Jacobian against finite differences of FK.
# 自检：用数值差分验证雅可比。
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(20):
        q = rng.uniform(Q_LOWER, Q_UPPER)
        J = jacobian(q)
        T0 = fk(q)
        eps = 1e-6
        Jn = np.zeros((6, 7))
        for i in range(7):
            dq = np.zeros(7); dq[i] = eps
            T1 = fk(q + dq)
            Jn[:3, i] = (T1[:3, 3] - T0[:3, 3]) / eps
            Jn[3:, i] = rot_error(T1[:3, :3], T0[:3, :3]) / eps
        worst = max(worst, np.abs(J - Jn).max())
    print(f"max |J_analytic - J_numeric| over 20 random configs = {worst:.2e}")

    q_demo = np.array([0, -1, 0, -2, 0, 1, 1], dtype=float)
    T = fk(q_demo)
    np.set_printoptions(precision=3, suppress=True)
    print("Q_DEMO end-effector position:", T[:3, 3])
    print("Q_DEMO end-effector z-axis (should point roughly down):", T[:3, 2])
    print("manipulability at Q_DEMO:", round(manipulability(q_demo), 4))
