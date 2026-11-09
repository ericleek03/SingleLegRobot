import matplotlib.pyplot as plt


class PDController:
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd
        self.prev_error = 0.0
        self.prev_time = None

    def update(self, setpoint: float, measurement: float, current_time: float):
        error = setpoint - measurement

        if self.prev_time is None:
            self.prev_error = error
            self.prev_time = current_time
            return self.kp * error

        delta_time = current_time - self.prev_time

        if delta_time > 0:
            derivative = (error - self.prev_error) / delta_time
        else:
            derivative = 0.0

        control_signal = self.kp * error + self.kd * derivative
        self.prev_error = error
        self.prev_time = current_time
        return control_signal

    def reset(self):
        self.prev_error = 0.0
        self.prev_time = None


class Joint:
    def __init__(self, inertia, torque_limit, angle=0.0):
        self.J = inertia
        self.torque_limit = torque_limit
        self.angle = angle  # [rad]
        self.velocity = 0.0  # [rad/s]

    def step(self, torque, dt):
        torque = max(min(torque, self.torque_limit), -self.torque_limit)
        self.velocity += (torque / self.J) * dt
        self.angle += self.velocity * dt
        return torque


def run(kp, kd, inertia=0.0219, torque_limit=23.5, target=0.3, dt=0.005, duration=0.50):
    pd = PDController(kp, kd)
    joint = Joint(inertia, torque_limit)
    angles, t, torques = [], [], []
    current_time = 0.0

    while current_time < duration:
        command = pd.update(target, joint.angle, current_time)
        torque = joint.step(command, dt)

        angles.append(joint.angle)
        torques.append(torque)
        t.append(current_time)

        current_time += dt
    return t, angles, torques


def summary(name, t, angles, torques, target):
    overshoot = max(0.0, (max(angles) - target) / target * 100)
    print(f"{name}: final angle {angles[-1]:.3f} rad | overshoot {overshoot:.1f} % | peak torque {max(map(abs, torques)):.1f} Nm")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(t, angles, label="angle")
    ax1.axhline(target, color="gray", linestyle="--", label="target")
    ax1.set_ylabel("Angle [rad]")
    ax1.set_title(f"{name} PD Controller Response")
    ax1.legend()
    ax2.plot(t, torques, color="tab:orange")
    ax2.set_ylabel("Torque [Nm]")
    ax2.set_xlabel("Time [s]")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    kp = 40.0
    kd = 2.75
    target_angle = 0.3 # radians
    hip_inertia = 0.0219
    summary_name = "Hip Joint"
    t, angles, torques = run(kp, kd, inertia=hip_inertia, target=target_angle)
    summary(summary_name, t, angles, torques, target_angle)