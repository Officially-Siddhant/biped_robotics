# plot_contact_forces.py
import matplotlib.pyplot as plt

class ContactForcePlotter:
    def __init__(self, duration, impulse_time, impulse_duration):
        self.duration = duration
        self.impulse_time = impulse_time
        self.impulse_duration = impulse_duration

        self.fig, self.ax = plt.subplots()
        self.lines = [self.ax.plot([], [], label=f"Contact {i}")[0] for i in range(4)]
        self.time_vals = []
        self.force_vals = [[] for _ in range(4)]

        self.ax.set_xlim(0, duration)
        self.ax.set_ylim(0, 100)
        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("Contact Force [N]")
        self.ax.set_title("Real-time Contact Forces")

    def update(self, time, frame_forces):
        self.time_vals.append(time)
        for i in range(4):
            self.force_vals[i].append(frame_forces[i])
            self.lines[i].set_data(self.time_vals, self.force_vals[i])
        plt.pause(0.001)

    def finalize_and_save(self, filename):
        self.ax.axvline(x=self.impulse_time, color='red', linestyle='--', label='Impulse start')
        self.ax.axvline(x=self.impulse_time + self.impulse_duration, color='red', linestyle='--', label='Impulse end')
        self.ax.legend(loc='upper right')
        self.fig.savefig(filename, dpi=300, bbox_inches="tight")