"""
controller.py — the mixing controller. Every design choice is a flag in Config,
so each one can be switched off and measured (ablation study).

Each 10 ms tick:
  1. desired pose on the ellipse        x_des, v_des
  2. measure q, run FK                  x, R
  3. task-space feedback                v = v_des + K e
  4. orientation feedback               ω = kr · e_R
  5. differential IK                    dq = J⁺ ξ
  6. null-space secondary task          + (I − J⁺J) dq0
  7. limit and optionally filter        scale / LPF
  8. integrate to a position command    q_cmd += dt · dq
  9. send (q_cmd, dq) to the robot

Config.original_v3() follows the choices of the course controller;
Config.improved() is the final version.
"""
from dataclasses import dataclass, replace
import numpy as np
from kinematics import fk, jacobian, rot_error, manipulability, DQ_MAX


@dataclass
class Config:
    # trajectory 
    duration: float = 20.0
    w: float = 2.5              # rad/s along the ellipse
    rx: float = 0.12
    ry: float = 0.07
    ramp_in: float = 2.0        # s
    ramp_out: float = 0.0       # s (0 = stop abruptly, like 3.py)
    smooth_ramp: bool = False   # False: linear ramp (3.py) ; True: smoothstep
    ramp_ff_term: bool = False  # include d(ramp)/dt in v_des (3.py omits it)
    # feedback gains 
    kp: float = 1.0             # x, y
    kz: float = 6.0             # z
    ki_z: float = 0.8           # integral on z
    kr: float = 0.05            # orientation
    integral_clamp_state: bool = False   # True = anti-windup on the state itself
    # IK
    dls_lambda: float = 0.0     # 0 = plain pseudoinverse ; >0 = damped least squares
    # null space 
    null_space: bool = True
    k_null: float = 3.0
    null_all_joints: bool = False   # False: only joint 0 (3.py) ; True: all joints to home
    # limits & filter 
    dq_limit: float = 1.5
    scale_instead_of_clip: bool = False
    lpf_alpha: float = 0.35     # 1.0 = no filter

    # integration (the key choice, see README)
    integrate_from_command: bool = False   # False: q_new = q_measured + dt*dq (3.py)
    # mode
    offline_replay: bool = False    # True: precompute q(t) kinematically, then replay (like 2.py)
    label: str = ""

    @staticmethod
    def original_v3():
        return Config(label="v3 (original 3.py)")

    @staticmethod
    def improved():
        return Config(
            label="v4 (improved)",
            kp=8.0, kz=8.0, ki_z=0.0, kr=4.0,
            ramp_out=2.0, smooth_ramp=True, ramp_ff_term=True,
            integral_clamp_state=True,
            dls_lambda=0.02,
            null_all_joints=True, k_null=1.0,
            scale_instead_of_clip=True,
            lpf_alpha=1.0,
            integrate_from_command=True,
        )


# --------------------------------------------------------------------------- #
# trajectory
# --------------------------------------------------------------------------- #
def _ramp(t, cfg):
    """Amplitude scale s(t) in [0,1] and its time-derivative ds/dt."""
    T_end = cfg.duration
    def shape(u):              # u in [0,1]
        u = min(max(u, 0.0), 1.0)
        if cfg.smooth_ramp:    # smoothstep 3u²−2u³ : zero slope at both ends
            return 3*u*u - 2*u**3, 6*u - 6*u*u
        return u, (1.0 if 0.0 < u < 1.0 else 0.0)
    s_in, ds_in = shape(t / cfg.ramp_in)
    ds_in /= cfg.ramp_in
    if cfg.ramp_out > 0:
        s_out, ds_out = shape((T_end - t) / cfg.ramp_out)
        ds_out = -ds_out / cfg.ramp_out
    else:
        s_out, ds_out = 1.0, 0.0
    return s_in * s_out, ds_in * s_out + s_in * ds_out


def ellipse(t, x0, cfg):
    s, ds = _ramp(t, cfg)
    c, sn = np.cos(cfg.w * t), np.sin(cfg.w * t)
    shape = np.array([cfg.rx * c, cfg.ry * sn, 0.0])
    dshape = np.array([-cfg.rx * cfg.w * sn, cfg.ry * cfg.w * c, 0.0])
    x_des = x0 + s * shape
    # product rule: d/dt (s·shape) = s·dshape + ds·shape
    # the course controller omitted the ds·shape term (small, but a real feed-forward error)
    v_des = s * dshape + (ds * shape if cfg.ramp_ff_term else 0.0)
    return x_des, v_des


# --------------------------------------------------------------------------- #
# one IK step (shared by online control and offline precompute)
# --------------------------------------------------------------------------- #
def ik_step(q, x_des, v_des, R_home, q_home, z_int, dt, cfg):
    T = fk(q)
    x, R = T[:3, 3], T[:3, :3]
    e = x_des - x
    if cfg.ki_z > 0:
        z_int += e[2] * dt
        if cfg.integral_clamp_state:
            z_int = np.clip(z_int, -1.0 / cfg.ki_z, 1.0 / cfg.ki_z)
    z_i = np.clip(cfg.ki_z * z_int, -1.0, 1.0)
    v = v_des + np.array([cfg.kp * e[0], cfg.kp * e[1], cfg.kz * e[2] + z_i])
    e_R = rot_error(R_home, R)
    omega = cfg.kr * e_R
    xi = np.hstack([v, omega])

    J = jacobian(q)
    if cfg.dls_lambda > 0:
        # damped least squares: J⁺ = Jᵀ (J Jᵀ + λ² I)⁻¹  — stays bounded near singularities
        J_pinv = J.T @ np.linalg.inv(J @ J.T + cfg.dls_lambda**2 * np.eye(6))
    else:
        J_pinv = np.linalg.pinv(J)
    dq = J_pinv @ xi

    if cfg.null_space:
        dq0 = np.zeros(7)
        if cfg.null_all_joints:
            dq0 = cfg.k_null * (q_home - q)
        else:
            dq0[0] = cfg.k_null * (q_home[0] - q[0])
        N = np.eye(7) - J_pinv @ J
        dq = dq + N @ dq0

    if cfg.scale_instead_of_clip:
        # shrink the whole vector so direction (= EE direction) is preserved
        ratio = np.max(np.abs(dq) / np.minimum(cfg.dq_limit, DQ_MAX))
        if ratio > 1.0:
            dq = dq / ratio
    else:
        dq = np.clip(dq, -cfg.dq_limit, cfg.dq_limit)
    return dq, z_int, x, e, e_R


# --------------------------------------------------------------------------- #
# run one experiment
# --------------------------------------------------------------------------- #
def run(cfg, robot, q_home, dt=0.01):
    """
    Runs the mixing routine on `robot` (a SimPanda) and returns a log dict of numpy arrays.
    The robot must already be resting at q_home (sagged, as on hardware).
    """
    q_meas = robot.get_positions()
    T0 = fk(q_meas)
    x0, R_home = T0[:3, 3].copy(), T0[:3, :3].copy()   # FK of the *actual* start (as in 3.py)

    n = int(round(cfg.duration / dt))
    log = {k: [] for k in ["t", "x_des", "x", "e_R", "q", "dq_cmd", "dq_raw", "manip"]}

    # ---------- offline replay (2.py style): plan in a perfect kinematic world ----------
    if cfg.offline_replay:
        q_plan = q_meas.copy()
        z_int = 0.0
        plan_q, plan_dq = [], []
        for k in range(n):
            t = (k + 1) * dt
            x_des, v_des = ellipse(t, x0, cfg)
            dq, z_int, *_ = ik_step(q_plan, x_des, v_des, R_home, q_home, z_int, dt, cfg)
            q_plan = q_plan + dt * dq          # a perfect robot: measured == commanded
            plan_q.append(q_plan.copy()); plan_dq.append(dq)

    q_cmd = q_meas.copy()
    dq_prev = np.zeros(7)
    z_int = 0.0
    for k in range(n):
        t = (k + 1) * dt
        x_des, v_des = ellipse(t, x0, cfg)
        q_meas = robot.get_positions()

        if cfg.offline_replay:
            dq_raw = plan_dq[k]
            dq = dq_raw
            q_cmd = plan_q[k]
            T = fk(q_meas); x = T[:3, 3]; e_R = rot_error(R_home, T[:3, :3])
        else:
            dq_raw, z_int, x, e, e_R = ik_step(q_meas, x_des, v_des, R_home, q_home,
                                               z_int, dt, cfg)
            dq = cfg.lpf_alpha * dq_raw + (1 - cfg.lpf_alpha) * dq_prev
            dq_prev = dq
            if cfg.integrate_from_command:
                q_cmd = q_cmd + dt * dq          # the command keeps its own state
            else:
                q_cmd = q_meas + dt * dq         # 3.py: restart from the measurement every tick

        robot.set_joint_positions_velocities(q_cmd, dq)
        robot.step(dt)

        log["t"].append(t); log["x_des"].append(x_des); log["x"].append(x)
        log["e_R"].append(np.linalg.norm(e_R)); log["q"].append(q_meas)
        log["dq_cmd"].append(dq); log["dq_raw"].append(dq_raw)
        log["manip"].append(manipulability(q_meas))

    # ---------- stop: hold the last command for 1 s (3.py then calls safe_move_to_position) ----------
    # Without ramp_out the arm is still moving at full speed here -> the stop is a jolt.
    for k in range(int(1.0 / dt)):
        t = cfg.duration + (k + 1) * dt
        q_meas = robot.get_positions()
        robot.set_joint_positions_velocities(q_cmd, np.zeros(7))
        robot.step(dt)
        T = fk(q_meas)
        log["t"].append(t); log["x_des"].append(log["x_des"][-1]); log["x"].append(T[:3, 3])
        log["e_R"].append(np.linalg.norm(rot_error(R_home, T[:3, :3]))); log["q"].append(q_meas)
        log["dq_cmd"].append(np.zeros(7)); log["dq_raw"].append(np.zeros(7))
        log["manip"].append(manipulability(q_meas))

    out = {k: np.array(v) for k, v in log.items()}
    out["x0"] = x0
    out["label"] = cfg.label
    return out


def metrics(log, settle=3.0):
    """Summary metrics for one run. 'settle' skips the first seconds (ramp-in)."""
    t = log["t"]
    T_end = t[-1] - 1.0                         # end of mixing (1 s hold follows)
    m = (t > settle) & (t < T_end - 2.0)        # steady part of the run
    err = log["x_des"] - log["x"]
    xy = np.linalg.norm(err[:, :2], axis=1)
    # end-effector acceleration from measured positions (2nd difference), 10-sample smoothing
    dt = t[1] - t[0]
    acc = np.diff(log["x"], n=2, axis=0) / dt**2
    k = np.ones(10) / 10
    acc_h = np.convolve(np.linalg.norm(acc[:, :2], axis=1), k, mode="same")
    jerk_dq = np.abs(np.diff(log["dq_cmd"], axis=0)).max() / dt
    return {
        "xy RMS err [mm]":       1000 * np.sqrt(np.mean(xy[m] ** 2)),
        "z mean err [mm]":       1000 * np.mean(err[m, 2]),
        "z max |err| [mm]":      1000 * np.abs(err[m, 2]).max(),
        "tilt max [deg]":        np.degrees(log["e_R"][m].max()),
        "joint0 drift [rad]":    log["q"][t <= T_end][-1, 0] - log["q"][0, 0],
        "max horiz acc [m/s^2]": acc_h.max(),
        "max cmd joint acc [rad/s^2]": jerk_dq,
        "min manipulability":    log["manip"].min(),
    }
