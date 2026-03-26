# dicUtils/velocity.py
import numpy as np
import matplotlib.pyplot as plt


def compute_velocity(trajectory, frame_step, fps):
    if len(trajectory) < 2:
        return None, None

    vx = np.diff(trajectory[:, 0]) / frame_step * fps
    vy = np.diff(trajectory[:, 1]) / frame_step * fps

    return vx, vy


def plot_velocity_fixed_scale(vx, vy, total_frames, frame_step, fps):

    x_axis = np.arange(len(vx)) * frame_step / fps  # seconds

    plt.figure()
    plt.plot(x_axis, vx)
    plt.xlim(0, total_frames / fps)
    plt.ylim(-100, 100)
    plt.xlabel("Time (s)")
    plt.ylabel("Vx (pixels/s)")
    plt.title("Vx vs Time")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(x_axis, vy)
    plt.xlim(0, total_frames / fps)
    plt.ylim(-20, 20)
    plt.xlabel("Time (s)")
    plt.ylabel("Vy (pixels/s)")
    plt.title("Vy vs Time")
    plt.grid(True)
    plt.show()
