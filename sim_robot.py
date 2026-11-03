"""


On the real Panda, the controller sends a joint position command q_cmd (and a
velocity command dq_cmd) every 10 ms. A low-level joint controller then pulls each
joint toward q_cmd, like a spring. The spring is not infinitely stiff, so an
unmodeled load (the cup of trail mix) makes the arm sag a little. The team saw
this sag on the real robot.
真机的底层关节控制器有限刚度 → 末端有未建模载荷时会"下沉"(sag)。

Calibration: the effective load (0.12 kg) is chosen so that the v1-style controller
(P-only, integrate-from-measured) sags ~15-20 mm in z — the number the team measured
on hardware. 标定：让 v1 控制器在仿真里下沉 15–20 mm，与真机观测一致。

Model per joint (decoupled, unit-free but tuned to look realistic):
    I · q̈ = Kp (q_cmd − q) + Kd (dq_cmd − q̇) + τ_payload
    τ_payload = Jvᵀ · [0, 0, −m g]          (unmodeled load: cup + imperfect gravity
                                            compensation, lumped as an "effective mass")
Plus: one control-cycle of communication delay and small encoder noise.
另外加了一拍通信延迟和少量编码器噪声。

This is a simplified model (no inertia coupling, no Coriolis), not full Panda dynamics.
"""
import numpy as np
from kinematics import jacobian, Q_LOWER, Q_UPPER

G = 9.81


class SimPanda:
    def __init__(self, q0, payload_kg=0.12, kp=150.0, inertia=0.15,
                 sim_dt=0.001, delay_steps=1, noise_std=2e-5, seed=0):
        self.q = np.array(q0, dtype=float)
        self.dq = np.zeros(7)
        self.m = payload_kg
        self.Kp = kp
        self.I = inertia
        self.Kd = 2.0 * np.sqrt(kp * inertia)        # critically damped per joint
        self.sim_dt = sim_dt
        self.delay = [(self.q.copy(), np.zeros(7))] * delay_steps
        self.q_cmd = self.q.copy()
        self.dq_cmd = np.zeros(7)
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    # ---- what the controller is allowed to see / call (same as ArmController) ----
    def get_positions(self):
        return self.q + self.rng.normal(0.0, self.noise_std, 7)

    def set_joint_positions_velocities(self, q_cmd, dq_cmd):
        self.delay.append((np.array(q_cmd, float), np.array(dq_cmd, float)))
        self.q_cmd, self.dq_cmd = self.delay.pop(0)

    # ---- physics ----
    def step(self, ctrl_dt):
        """Advance the plant by one control period using small physics sub-steps."""
        n = int(round(ctrl_dt / self.sim_dt))
        for _ in range(n):
            Jv = jacobian(self.q)[:3]
            tau_load = Jv.T @ np.array([0.0, 0.0, -self.m * G])
            ddq = (self.Kp * (self.q_cmd - self.q)
                   + self.Kd * (self.dq_cmd - self.dq)
                   + tau_load) / self.I
            self.dq += ddq * self.sim_dt
            self.q += self.dq * self.sim_dt
            self.q = np.clip(self.q, Q_LOWER, Q_UPPER)

    def settle(self, q_cmd, seconds=2.0, ctrl_dt=0.01):
        """Hold a commanded pose until the arm comes to rest (like safe_move_to_position)."""
        for _ in range(int(seconds / ctrl_dt)):
            self.set_joint_positions_velocities(q_cmd, np.zeros(7))
            self.step(ctrl_dt)


if __name__ == "__main__":
    from kinematics import fk
    q_demo = np.array([0, -1, 0, -2, 0, 1, 1], dtype=float)
    for m in [0.0, 0.12]:
        r = SimPanda(q_demo, payload_kg=m, noise_std=0.0)
        r.settle(q_demo, seconds=3.0)
        sag = fk(r.q)[2, 3] - fk(q_demo)[2, 3]
        print(f"payload {m:.2f} kg -> static z sag holding Q_DEMO: {sag*1000:+.1f} mm")
